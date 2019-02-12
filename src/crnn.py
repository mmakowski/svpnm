"""
A CRNN model implementation. The earlier experiments, 07-uaf-lstm and 10-crnn-uaf-noisy use their own copy;
They have not been refactored to use this implementation because that would trigger a lengthy rerun of the experiments.
"""
import collections
import logging
import sys

import keras # type: ignore
import numpy as np # type: ignore
import sklearn.metrics # type: ignore

import preprocess


# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


Hyperparameters = collections.namedtuple('Hyperparameters',
    [# preprocessing
     'min_token_occurrences',
     'min_token_documents',
     'class_weight',
     # network architecture
     'embed_dim',
     # - convolution:
     'conv_filters',
     'conv_kernel_size',
     'conv_padding',
     'conv_strides',
     'conv_pool_size',
     'lstm_size',
     'batch_size',
     'max_length',  # input will be truncated/padded to this number of tokens
     'dropout',
     # early stopping:
     'es_min_delta', # stop training when improvement in loss compared to the previous epoch is below this value
     'es_patience',  # allow this many epochs without improvement
     'max_epochs'
    ])


PreprocessedData = collections.namedtuple('PreprocessedData',
    ['x_train', 'y_train', 'x_test', 'y_test', 'id_test', 'vocab_size'])


def train_and_evaluate_files(train_file: str,
                             test_file: str,
                             hyperparams: Hyperparameters,
                             random_state: int,
                             train_examples: int = None,
                             test_examples: int = None) -> dict:
    log.info("evaluating %s with training example limit %s", train_file, train_examples)
    data = preprocess_data(train_file, test_file, hyperparams, random_state, train_examples, test_examples)
    return train_and_evaluate_preprocessed(data, hyperparams)


def preprocess_data(train_file: str,
                    test_file: str,
                    hyperparams: Hyperparameters,
                    random_state: int,
                    train_examples: int = None,
                    test_examples: int = None) -> PreprocessedData:
    datasets, vocab_size, _ = preprocess.tokenise_data([preprocess.DatasetSpec(train_file, train_examples),
                                                        preprocess.DatasetSpec(test_file, test_examples)],
                                                       int(hyperparams.min_token_occurrences),
                                                       int(hyperparams.min_token_documents),
                                                       random_state=random_state)
    log.info("loaded %d training examples", len(datasets['train']))
    x_train, y_train, _ = format_data(datasets['train'], int(hyperparams.max_length))
    x_test, y_test, id_test = format_data(datasets['test'], int(hyperparams.max_length))
    return PreprocessedData(x_train, y_train, x_test, y_test, id_test, vocab_size)


def train_and_evaluate_preprocessed(data: PreprocessedData,
                                    hyperparams: Hyperparameters) -> dict:
    model = _create_model(data.vocab_size, hyperparams)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=hyperparams.es_min_delta,
                                                   patience=int(hyperparams.es_patience))
    fit_result = model.fit(data.x_train,
                           data.y_train,
                           batch_size=int(hyperparams.batch_size),
                           epochs=int(hyperparams.max_epochs),
                           validation_split=0.1,
                           class_weight=hyperparams.class_weight,
                           shuffle=True,
                           callbacks=[early_stopping])
    log.info("evaluating on %d test examples", len(data.y_test))
    eval_result = _evaluate(model, data.x_test, data.y_test, data.id_test)
    log.info("test accuracy: %f", eval_result['accuracy'])
    keras.backend.clear_session()
    return dict(train=fit_result.history, test=eval_result, vocabulary_size=data.vocab_size)


def format_data(dataset, max_length: int):
    x = keras.preprocessing.sequence.pad_sequences(dataset['vectors'], maxlen=max_length)
    y = dataset['label']
    ids = dataset['file']
    return x, y, ids


def _create_model(vocab_size: int, hyperparams: Hyperparameters):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size,
                               int(hyperparams.embed_dim),
                               input_length=int(hyperparams.max_length)),

        keras.layers.Conv1D(filters=int(hyperparams.conv_filters),
                            kernel_size=int(hyperparams.conv_kernel_size),
                            padding=hyperparams.conv_padding,
                            activation='relu',
                            strides=int(hyperparams.conv_strides)),
        keras.layers.MaxPooling1D(pool_size=int(hyperparams.conv_pool_size)),

        keras.layers.CuDNNLSTM(int(hyperparams.lstm_size)),
        keras.layers.Dropout(hyperparams.dropout),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def _evaluate(model,
              x_test,
              y_test,
              id_test) -> dict:
    y_prob = model.predict(x_test).flatten()
    y_pred = np.round(y_prob)
    return dict(
        roc_auc=sklearn.metrics.roc_auc_score(y_test, y_prob),
        average_precision=sklearn.metrics.average_precision_score(y_test, y_prob),
        precision=sklearn.metrics.precision_score(y_test, y_pred),
        recall=sklearn.metrics.recall_score(y_test, y_pred),
        f1=sklearn.metrics.f1_score(y_test, y_pred),
        accuracy=sklearn.metrics.accuracy_score(y_test, y_pred),
        confusion_matrix=sklearn.metrics.confusion_matrix(y_test, y_pred).tolist(),
        examples=[(id_test.tolist()[i], y_test.tolist()[i], y_prob.tolist()[i]) for i in range(len(id_test))]
    )
