#!/usr/bin/env python
from __future__ import print_function

import json
import logging
import math
import os
import pprint
import sys
import uuid

import keras # type: ignore
import numpy as np # type: ignore
import sklearn.metrics # type: ignore

import lstm
import preprocess


# network parameters, picked through interactive tuning
min_token_occurrences = 1
min_token_documents = 1
embed_dim = 16
lstm_size = 64
batch_size = 32
max_length = 256
dropout = 0.0

# training and evaluation parameters
runs = 50
train_example_counts = [100, 500, 1000, 2000, 5000, 10000]
test_examples = 20000
# -- early stopping:
es_min_delta = 0.0001 # stop training when improvement in loss compared to the previous epoch is below this value
es_patience = 2 # allow this many epochs without improvement
max_epochs = 50


# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def main(train_file: str, test_file: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    run_ids = [uuid.uuid4().hex[:8] for _ in range(runs)]
    for run in range(runs):
        run_ids[run]
        log.info("run %d (%s)", run+1, run_ids[run])
        learning_curve = []
        for train_examples in train_example_counts:
            results = _train_and_evaluate(train_file, test_file, train_examples, run)
            learning_curve.append(dict(training_examples=train_examples, results=results))
        with open(os.path.join(output_dir, "learning_curve_%s.json" % run_ids[run]), 'w') as f:
            f.write(json.dumps(learning_curve, indent=2))
    with open(os.path.join(output_dir, "run_ids.txt"), 'w') as f:
        f.write("\n".join(run_ids))


def _train_and_evaluate(train_file: str, test_file: str, train_examples: int, random_state: int) -> dict:
    log.info("evaluating with %d training examples", train_examples)
    datasets, vocab_size, _ = preprocess.tokenise_data([preprocess.DatasetSpec(train_file, train_examples),
                                                        preprocess.DatasetSpec(test_file, test_examples)],
                                                        min_token_occurrences,
                                                        min_token_documents,
                                                        random_state=random_state)
    x_train, y_train = _format_data(datasets['train'], max_length)
    x_test, y_test = _format_data(datasets['test'], max_length)
    model = _create_model(embed_dim, lstm_size, vocab_size, max_length, dropout)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=es_min_delta,
                                                   patience=es_patience)
    fit_result = model.fit(x_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=max_epochs,
                           validation_split=0.1,
                           callbacks=[early_stopping])
    log.info("evaluating on %d test examples", test_examples)
    eval_result = _evaluate(model, x_test, y_test)
    log.info("test accuracy: %f", eval_result['accuracy'])
    return dict(train=fit_result.history, test=eval_result)


def _format_data(dataset, max_length: int):
    x = keras.preprocessing.sequence.pad_sequences(dataset['vectors'], maxlen=max_length)
    y = dataset['label']
    return x, y


def _create_model(embed_dim: int,
                  lstm_size: int,
                  vocab_size: int,
                  input_length: int,
                  dropout: float):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size,
                               embed_dim,
                               input_length=input_length),

        keras.layers.Conv1D(filters=64,
                            kernel_size=5,
                            padding='valid',
                            activation='relu',
                            strides=1),
        keras.layers.MaxPooling1D(pool_size=4),

        keras.layers.LSTM(lstm_size, dropout=dropout, recurrent_dropout=dropout),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def _evaluate(model,
              x_test,
              y_test) -> dict:
    y_prob = model.predict(x_test).flatten()
    y_pred = np.round(y_prob)
    return dict(
        roc_auc=sklearn.metrics.roc_auc_score(y_test, y_prob),
        precision=sklearn.metrics.precision_score(y_test, y_pred),
        recall=sklearn.metrics.recall_score(y_test, y_pred),
        f1=sklearn.metrics.f1_score(y_test, y_pred),
        accuracy=sklearn.metrics.accuracy_score(y_test, y_pred),
        confusion_matrix=sklearn.metrics.confusion_matrix(y_test, y_pred).tolist()
    )


if __name__ == '__main__':
    main(*sys.argv[1:])
