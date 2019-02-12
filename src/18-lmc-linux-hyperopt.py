#!/usr/bin/env python
from __future__ import print_function

import configparser
import json
import logging
import os
import shutil
import sys
import uuid

import hyperopt as hp
import numpy as np
import pandas as pd
import sklearn.ensemble

import lmc

HP_SPACE = hp.hp.choice('all', [{
        'pos_class_weight': hp.hp.uniform('1', 1, 30),
        'pooling': hp.hp.choice('pooling', lmc.POOLING_TYPES),
        'classifier': hp.hp.choice('classifier', [
            # {
            #     'type': 'NaiveBayes'
            # },
            # {
            #     'type': 'SVM',
            #     'C': hp.hp.lognormal('svm_C', 0, 1),
            #     'kernel': hp.hp.choice('svm_kernel', [
            #         {'type': 'rbf', 'gamma': hp.hp.lognormal('svm_rbf_gamma', 0, 1)}
            #     ])
            # },
            {
                'type': 'RandomForest',
                'n_estimators': hp.hp.qloguniform('rf_n_estimators', 2, 6, 10),
                'criterion': hp.hp.choice('rf_criterion', ['gini', 'entropy'])
            },
            # {
            #     'type': 'AdaBoost',
            #     'n_estimators': hp.hp.qloguniform('ab_n_estimators', 2, 6, 10)
            # }
        ])
}])

MAX_EVALS = 100
RNG_SEED = 8798547

# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# configuration
config = configparser.ConfigParser()
config.read('config.ini')
PARALLELISM = int(config['environment']['parallelism'])


def main(index_file: str, vectors_file: str, output_dir: str):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    pooling_data = _preprocess_data(index_file, vectors_file, rng_seed=RNG_SEED)
    hp.fmin(_objective_fn(output_dir, pooling_data), HP_SPACE, hp.tpe.suggest, max_evals=MAX_EVALS, rstate=np.random.RandomState(RNG_SEED)) 
    best_hyperparams = _find_best_hyperparams(output_dir)
    with open(os.path.join(output_dir, "best_hyperparameters.json"), 'w') as f:
        f.write(json.dumps(best_hyperparams, indent=2))
    

def _preprocess_data(index_file: str, vectors_file: str, rng_seed: int) -> dict:
    vectors = pd.read_pickle(vectors_file)
    return {pooling_type: _preprocess_data_set_pooling(index_file, vectors, pooling_type, rng_seed)
            for pooling_type in lmc.POOLING_TYPES}


def _preprocess_data_set_pooling(index_file: str, vectors, pooling_type: str, rng_seed: int) -> lmc.Data:
    x_dev, y_dev, id_dev = lmc.load_dataset(index_file, vectors, pooling_type)
    devset = pd.DataFrame(dict(file=id_dev, vector=x_dev.tolist(), label=y_dev))
    log.info("loaded %d examples with %s pooling", len(devset), pooling_type)
    rng = np.random.RandomState(rng_seed)
    devset['group'] = rng.randint(0, 100, devset.shape[0])
    train = devset[devset['group'] < 80]
    test = devset[devset['group'] >= 80]
    x_train, y_train, _ = lmc.df_to_x_y_id(train)
    x_test, y_test, id_test = lmc.df_to_x_y_id(test)
    return lmc.Data(x_train, y_train, x_test, y_test, id_test)


def _objective_fn(output_dir: str, pooling_data: dict):
    def objective(sampled: dict) -> dict:
        log.info("evaluating %s", sampled)
        try:
            data = pooling_data[sampled['pooling']]
            results = lmc.train_and_evaluate(data, sampled, n_jobs=PARALLELISM)
            del(results['test']['examples'])
            results['hyperparameters'] = sampled
            with open(os.path.join(output_dir, "{}.json".format(uuid.uuid4().hex[:8])), 'w') as f:
                f.write(json.dumps(results, indent=2))
            log.info("average precision: %s", results['test']['average_precision'])
            return {'status': hp.STATUS_OK, 'loss': 1.0 - results['test']['average_precision']}
        except:
            log.exception("error while evaluating")
            return {'status': hp.STATUS_FAIL}
    return objective


def _find_best_hyperparams(output_dir: str) -> dict:
    best_avg_precision = 0
    result = None
    for file in os.listdir(output_dir):
        with open(os.path.join(output_dir, file)) as f:
            results = json.load(f)
            avg_precision = results['test']['average_precision']
            if avg_precision > best_avg_precision:
                best_avg_precision = avg_precision
                result = results['hyperparameters']
    log.info("best average precision: %f for hyperparameters %s", best_avg_precision, result)
    return result


if __name__ == '__main__':
    main(*sys.argv[1:])
