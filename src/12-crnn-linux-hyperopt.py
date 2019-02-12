#!/usr/bin/env python
from __future__ import print_function

import json
import logging
import os
import shutil
import sys
import uuid

import hyperopt as hp
import numpy as np

import crnn
import preprocess


# Nones will be replaced with sampled values; see below for hyperparameter space definition
hyperparam_template = crnn.Hyperparameters(
    min_token_occurrences=1,
    min_token_documents=1,
    class_weight=None,
    embed_dim=None,
    conv_filters=None,
    conv_kernel_size=None,
    conv_padding=None,
    conv_strides=None,
    conv_pool_size=None,
    lstm_size=None,
    batch_size=16,
    max_length=2048,
    dropout=None,
    es_min_delta=0.0001,
    es_patience=2,
    max_epochs=50,
)

HP_SPACE = hp.hp.choice('all', [{
    'class_weight': {0: 1, 1: hp.hp.uniform('1', 1, 30)},
    'embed_dim': 2**hp.hp.quniform('2', 2, 6, 2),
    'conv_filters': 2**hp.hp.quniform('3', 3, 8, 1),
    'conv_kernel_size': hp.hp.quniform('4', 2, 10, 1),
    'conv_padding': hp.hp.choice('5', ['valid', 'same']),
    'conv_strides': hp.hp.quniform('6', 1, 8, 1),
    'conv_pool_size': 2**hp.hp.quniform('7', 1, 5, 1),
    'lstm_size': 2**hp.hp.quniform('8', 3, 11, 1),
    'dropout': hp.hp.choice('10', [0.0, 0.5])
}])

MAX_EVALS = 100
RNG_SEED = 1214

# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def main(data_file: str, output_dir: str):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    data = _preprocess_data(data_file, hyperparam_template, RNG_SEED)
    hp.fmin(_objective_fn(output_dir, data), HP_SPACE, hp.tpe.suggest, max_evals=MAX_EVALS, rstate=np.random.RandomState(RNG_SEED)) 
    best_hyperparams = _find_best_hyperparams(output_dir)
    with open(os.path.join(output_dir, "best_hyperparameters.json"), 'w') as f:
        f.write(json.dumps(best_hyperparams._asdict(), indent=2))


def _preprocess_data(data_file: str, hyperparams: crnn.Hyperparameters, rng_seed: int) -> crnn.PreprocessedData:
    datasets, vocab_size, _ = preprocess.tokenise_data([preprocess.DatasetSpec(data_file, None)],
                                                       hyperparams.min_token_occurrences,
                                                       hyperparams.min_token_documents,
                                                       random_state=rng_seed)
    devset = datasets['train']
    log.info("loaded %d examples", len(devset))
    rng = np.random.RandomState(rng_seed)
    devset['group'] = rng.randint(0, 100, devset.shape[0])
    train = devset[devset['group'] < 80]
    test = devset[devset['group'] >= 80]
    x_train, y_train, _ = crnn.format_data(train, int(hyperparams.max_length))
    x_test, y_test, id_test = crnn.format_data(test, int(hyperparams.max_length))
    return crnn.PreprocessedData(x_train, y_train, x_test, y_test, id_test, vocab_size)


def _objective_fn(output_dir: str, data: crnn.PreprocessedData):
    def objective(sampled: dict) -> dict:
        hyperparams = hyperparam_template._replace(**sampled)
        log.info("evaluating %s", hyperparams)
        try:
            results = crnn.train_and_evaluate_preprocessed(data, hyperparams)
            del(results['test']['examples'])
            results['hyperparameters'] = hyperparams._asdict()
            with open(os.path.join(output_dir, "{}.json".format(uuid.uuid4().hex[:8])), 'w') as f:
                f.write(json.dumps(results, indent=2))
            return {'status': hp.STATUS_OK, 'loss': 1.0 - results['test']['average_precision']}
        except:
            log.error("error while evaluating: %s", sys.exc_info()[1])
            return {'status': hp.STATUS_FAIL}
    return objective


def _find_best_hyperparams(output_dir: str) -> crnn.Hyperparameters:
    best_avg_precision = 0
    result = None
    for file in os.listdir(output_dir):
        with open(os.path.join(output_dir, file)) as f:
            results = json.load(f)
            avg_precision = results['test']['average_precision']
            if avg_precision > best_avg_precision:
                best_avg_precision = avg_precision
                result = crnn.Hyperparameters(**results['hyperparameters'])
    log.info("best average precision: %f for hyperparameters %s", best_avg_precision, result)
    return result


if __name__ == '__main__':
    main(*sys.argv[1:])
