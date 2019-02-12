#!/usr/bin/env python
import configparser
import json
import logging
import os
import shutil
import sys
import uuid

import numpy as np
import pandas as pd

import lmc


# how many times to run for each split
runs = 10

# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# configuration
config = configparser.ConfigParser()
config.read('config.ini')
PARALLELISM = int(config['environment']['parallelism'])


def main(data_dir: str, hyperparams_file: str, vectors_file: str, output_dir: str):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    vectors = pd.read_pickle(vectors_file)
    with open(hyperparams_file) as f:
        model_spec = json.load(f)
    splits = sorted([fn.split("_")[0] for fn in os.listdir(data_dir) if fn.endswith("_train.csv")])
    for split in splits:
        _run_split(data_dir, output_dir, split, vectors, model_spec)


def _run_split(data_dir: str, output_dir: str, split: str, vectors, model_spec: dict):
    split_output_dir = os.path.join(output_dir, split)
    if not os.path.exists(split_output_dir):
        os.makedirs(split_output_dir)
    log.info("running split %s", split)
    run_ids = [uuid.uuid4().hex[:8] for _ in range(runs)]
    data = lmc.load_datasets(data_dir, split, vectors, model_spec['pooling'])
    for run in range(runs):
        log.info("run %d (%s)", run+1, run_ids[run])
        results = lmc.train_and_evaluate(data, model_spec, n_jobs=PARALLELISM)
        with open(os.path.join(split_output_dir, "results_%s.json" % run_ids[run]), 'w') as f:
            f.write(json.dumps(results, indent=2))

    with open(os.path.join(split_output_dir, "run_ids.txt"), 'w') as f:
        f.write("\n".join(run_ids))


if __name__ == '__main__':
    main(*sys.argv[1:])
