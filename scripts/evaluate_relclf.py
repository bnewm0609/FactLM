"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import shlex
import sys
import subprocess


def main():
    arg_str = " ".join(sys.argv[1:])

    cmd_eval = f"python src/run_config.py data=id_subsample {arg_str}"
    print(cmd_eval)
    subprocess.run(shlex.split(cmd_eval))

    cmd_eval = f"python src/run_config.py data=ood_human_prompts {arg_str}"
    print(cmd_eval)
    subprocess.run(shlex.split(cmd_eval))

    cmd_eval = f"python src/run_config.py data=ood_object_distribution {arg_str}"
    print(cmd_eval)
    subprocess.run(shlex.split(cmd_eval))

    cmd_eval = f"python src/run_config.py data=ood_keyboard {arg_str}"
    print(cmd_eval)
    subprocess.run(shlex.split(cmd_eval))


if __name__ == "__main__":
    main()
