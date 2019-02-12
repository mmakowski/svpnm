#!/usr/bin/env python
from __future__ import print_function

import json
import logging
import os
import shutil
import sys
import uuid

import crnn


# how many times to run for each split
runs = 10


# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def main(data_dir: str, output_dir: str, hyperparam_file: str):
    with open(hyperparam_file) as f:
        hyperparams = crnn.Hyperparameters(**json.load(f))
    # weight keys are loaded from json as strings, convert them to integers
    hyperparams = hyperparams._replace(class_weight={int(c): w for c, w in hyperparams.class_weight.items()})
    splits = sorted([fn.split("_")[0] for fn in os.listdir(data_dir) if fn.endswith("_train.csv")])
    for split in splits:
        _run_split(data_dir, output_dir, split, hyperparams)


def _run_split(data_dir: str, output_dir: str, split: str, hyperparams: crnn.Hyperparameters):
    split_output_dir = os.path.join(output_dir, split)
    if os.path.exists(os.path.join(split_output_dir, "run_ids.txt")):
        log.info("split %s already completed, skipping", split)
        return
    if not os.path.exists(split_output_dir):
        os.makedirs(split_output_dir)
    log.info("running split %s", split)
    train_file = os.path.join(data_dir, "{}_train.csv".format(split))
    test_file = os.path.join(data_dir, "{}_test.csv".format(split))
    run_ids = [uuid.uuid4().hex[:8] for _ in range(runs)]
    data = crnn.preprocess_data(train_file, test_file, hyperparams, random_state=1)
    for run in range(runs):
        log.info("run %d (%s)", run+1, run_ids[run])
        results = crnn.train_and_evaluate_preprocessed(data, hyperparams)
        with open(os.path.join(split_output_dir, "results_%s.json" % run_ids[run]), 'w') as f:
            f.write(json.dumps(results, indent=2))
    with open(os.path.join(split_output_dir, "run_ids.txt"), 'w') as f:
        f.write("\n".join(run_ids))


if __name__ == '__main__':
    main(*sys.argv[1:])
