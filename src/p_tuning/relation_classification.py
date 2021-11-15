"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
import os

import numpy as np
import sklearn
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from src.data_utils.dataset import load_file, LAMADatasetRelClf, load_files, load_relations, RelationMap, load_relations_templates
from src.data_utils.vocab import init_vocab


def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=-1)
    micro_p, micro_r, micro_f1, support = sklearn.metrics.precision_recall_fscore_support(labels, predictions, average="micro")
    macro_p, macro_r, macro_f1, support = sklearn.metrics.precision_recall_fscore_support(labels, predictions, average="macro")
    acc = (predictions == labels).sum() / len(labels)
    metrics = {
        "acc": acc,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "support": support,
    }
    return metrics

class RelationClassifier:
    def __init__(self, args):
        self.args = args
        # load in tokenizer
        self.config = AutoConfig.from_pretrained(self.args.model.name)
        if not self.config._name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model.name, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path, use_fast=False)

        # load in relations
        train_relation_ids = load_relations(self.args.data.train.relations_path)
        dev_relation_ids = load_relations(self.args.data.dev.relations_path)
        test_relation_ids = load_relations(self.args.data.test.relations_path)
        self.num_relations = len(set(train_relation_ids + dev_relation_ids + test_relation_ids))

        # load in data if we're going to train (otherwise the Trainer in cli will handle dataloading
        if args.train or args.model.evaluate_relclf:
            init_vocab(args)

            self.train_data = load_files(self.args.data.train.pairs_path, map(lambda relation: f'{relation}/train.jsonl', train_relation_ids))
            self.dev_data = load_files(self.args.data.dev.pairs_path, map(lambda relation: f'{relation}/dev.jsonl', dev_relation_ids))
            self.test_data = load_files(self.args.data.test.pairs_path, map(lambda relation: f'{relation}/test.jsonl', test_relation_ids))

            if self.args.debug:
                self.train_data = self.train_data[:32]
                self.dev_data = self.dev_data[:32]
                self.test_data = self.test_data[:32]

            self.train_set = LAMADatasetRelClf('train', self.train_data, self.tokenizer, self.args)
            self.dev_set = LAMADatasetRelClf('dev', self.dev_data, self.tokenizer, self.args)
            self.test_set = LAMADatasetRelClf('test', self.test_data, self.tokenizer, self.args)

        # load in model
        print(self.num_relations, len(RelationMap.lab2rel))
        assert self.num_relations == len(RelationMap.lab2rel)
        model_path = args.model.name
        if not args.get("train", False):
            model_path = self.get_save_path(None)
            # checkpoint directory names are in the form of `checkpoint-{step_number}`
            # we want to load the checkpoint of the fully trained model (which has the highest step_number),
            # so we sort by the dirnames by the step_number and take the last one
            print(model_path)
            checkpoint_dirname = sorted([(int(dirname.split("-")[1]), dirname) for dirname in os.listdir(model_path) if "checkpoint" in dirname])[-1]
            model_path = os.path.join(model_path, checkpoint_dirname[1])

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,  # either path or model name
                                                                        num_labels=self.num_relations,
                                                                        id2label=RelationMap.lab2rel,
                                                                        label2id=RelationMap.rel2lab)
        self.model.eval()
        print(self.args.device)
        self.model.to(self.args.device)

    def get_save_path(self, sub_dir):
        out_dir = f"{self.args.model.id}-{self.args.data.train.id}"
        if self.args.get("debug", False):
            out_dir += "-debug"
        if sub_dir is None:
            out_dir = os.path.join(self.args.model.out_path_prefix, out_dir)
        else:
            out_dir = os.path.join(self.args.model.out_path_prefix, out_dir, sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def train(self):
        self.model.train()
        training_args = TrainingArguments(
            output_dir=self.get_save_path(None),
            seed=self.args.model.seed,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            max_steps=(10 if self.args.debug else -1),
            save_steps=(10 if self.args.debug else 500),
            logging_first_step=self.args.debug,
            load_best_model_at_end=True,
            no_cuda=(self.args.device == "cpu")  # self.args.no_cuda,
        )
        trainer = LoggingTrainer(model=self.model,
                                 args=training_args,
                                 train_dataset=self.train_set,
                                 eval_dataset=self.dev_set,
                                 compute_metrics=compute_metrics,
                                 log_file=os.path.join(self.get_save_path("logs"), "train.log"),
                                 )
        if self.args.train:
            trainer.train()
        self.model.eval()
        eval_predictions = trainer.predict(self.test_set)
        with open(os.path.join(self.get_save_path(f"{self.args.data.test.id}"), "test.log"), "w") as f:
            json.dump(eval_predictions.metrics, f)

    @torch.no_grad()
    def predict(self, queries):
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True)
        inputs = {k: t.to(self.args.device) for k, t in inputs.items()}
        predictions = self.model(**inputs)[0]
        predictions = np.argmax(predictions.cpu().numpy(), axis=-1)
        return [self.model.config.id2label[pred] for pred in predictions]


class LoggingTrainer(Trainer):
    def __init__(self, log_file=None, **kwargs):
        super().__init__(**kwargs)
        # clear the log
        self.log_file = log_file
        f = open(self.log_file, "w")
        f.close()

    def log(self, logs):
        super().log(logs)
        output = {**logs, **{"step": self.state.global_step}}
        with open(self.log_file, "a") as f:
            f.write(f"{json.dumps(output)}\n")
