"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_utils.vocab import get_vocab_by_strategy, token_wrapper


def load_file(filename):
    "Loads a single jsonl file"
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def load_json_file(filename):
    "Loads a single json file"
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def load_files(basedir, relations, jsonl=True):
    data = []
    for relation in relations:
        try:
            if jsonl:
                data.extend(load_file(os.path.join(basedir, relation)))
            else:
                data.extend(load_json_file(os.path.join(basedir, relation)))
        except FileNotFoundError:
            print(f"Cannot load relation: {relation}")
    return data


def load_relations(relation_id):
    """
    Loads relations names (e.g. P101) from comma-separated list (relation_id is this string) or
    from newline-separated file (relation_id is filename).
    """
    if "," in relation_id:
        return relation_id.split(",")
    elif os.path.isfile(relation_id):
        with open(relation_id) as f:
            return [l.strip() for l in f.readlines()]
    else:
        return [relation_id]


def load_relations_templates(path):
    """
    Loads the templates (patterns) for each template from json or jsonl files
    (jsonl means one template per relation)
    (json maps from relation to list of templates)
    Each template is a dict with a "template" key that stores the string representation of the template

    Returns
    dict[str, list[str]]: dictionary mapping relation id to list of templates
    """
    ext = os.path.splitext(path)[1]
    if ext == ".jsonl":
        return dict((d['relation'], [d['template']]) for d in load_file(path))
    elif ext == ".json":
        rel_map = {rel: [t['template'] for t in templates] for rel, templates in load_json_file(path).items()}
        return rel_map


class LAMADataset(Dataset):
    def __init__(self, dataset_type, data, tokenizer, relation_templates, args):
        super().__init__()
        self.args = args
        self.data = list()
        self.dataset_type = dataset_type
        self.relations = load_relations_templates(self.args.data[dataset_type].template_path)
        self.x_hs, self.x_ts = [], []

        vocab = get_vocab_by_strategy(args, tokenizer)
        for d in data:
            if token_wrapper(args, d['obj_label']) not in vocab:
                continue
            for template in self.relations[d['predicate_id']]:
                self.data.append((
                    d['sub_label'],
                    d['obj_label'],
                    d['predicate_id'],
                    template,
                ))
                self.x_ts.append(d['obj_label'])
                self.x_hs.append(d['sub_label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class RelationBatchDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        assert self.batch_size % 2 == 0
        self.current = 0

        # let's precalculate the indices for each batch
        self.batch_idxs = []

        i = 0
        while i < len(self.dataset) - 1:
            if self.dataset[i + 1][:3] == self.dataset[i][:3]:
                self.batch_idxs.append((i, i+1))
                i += 2
            else:
                # skip the last one if there is an extra triple
                i += 1

        if shuffle:
            np.random.shuffle(self.batch_idxs)
            self.batch_idxs = np.array(self.batch_idxs).flatten()
        else:
            self.batch_idxs = np.array(self.batch_idxs).flatten()

        self._len = len(self.batch_idxs) // self.batch_size + int(len(self.batch_idxs) % self.batch_size != 0)

        # last_triple = None
        # self.batch_idx = -1
        # curr_batch_count = 0
        # for i, d in enumerate(self.dataset):
        #     curr_triple = d[:3]  # (d['sub_label'], d['obj_label'], d['predicate_id'])
        #     if last_triple is None or curr_triple != last_triple:
        #         self.batch_idx += 1
        #         last_triple = curr_triple
        #         self.batch_idxs.append(self.batch_idx)
        #         curr_batch_count = 1
        #     elif curr_triple == last_triple:
        #         if curr_batch_count == batch_size: # if we get to the end of the batch, update the batch_idx
        #             self.batch_idx += 1
        #             curr_batch_count = 0
        #         self.batch_idxs.append(self.batch_idx)
        #         curr_batch_count += 1
        # self._len = self.batch_idx + 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > len(self.batch_idxs):
            # reset counter for next round
            self.current = 0
            raise StopIteration
        batch = [self.dataset[idx] for idx in self.batch_idxs[self.current: self.current + self.batch_size]]
        self.current += self.batch_size
        # current_batch_idx = self.batch_idxs[self.current]
        # while self.batch_idxs[self.current] == current_batch_idx:
        #     batch.append(self.dataset[self.current])
        #     self.current += 1

        # construct the batch
        return list(zip(*batch))

    def __len__(self):
        """Actual lenght might be shorter..."""
        return self._len


class LAMADatasetRelClf(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.dataset_type = dataset_type
        self.queries, self.labels = [], []

        relations_template = load_relations_templates(self.args.data[dataset_type].template_path)

        vocab = get_vocab_by_strategy(args, tokenizer)
        for d in data:
            if token_wrapper(args, d['obj_label']) not in vocab:
                continue

            templates = relations_template[d['predicate_id']]
            for template in templates:
                template = template.replace("[X]", d['sub_label'])
                template = template.replace("[X]", tokenizer.mask_token)
                self.queries.append(
                    template
                )
                self.labels.append(RelationMap.rel2lab[d['predicate_id']])

        print("Starting tokenization")
        self.encodings = tokenizer(self.queries, padding=True)
        print("Finishing tokenization")

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item

    def __len__(self):
        return len(self.labels)



class RelationMap:
    """Used for RelClf to associate relations with an id"""
    rel2lab = {rel: lab for lab, rel in enumerate(load_relations("data/LAMA/relations_subsets/all_relations.txt"))}
    lab2rel = {v: k for k, v in rel2lab.items()}
