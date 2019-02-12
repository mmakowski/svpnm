"""C language model"""

import collections

import keras

Hyperparameters = collections.namedtuple('Hyperparameters',
    [# network architecture
     'vocab_size',
     'embed_dim',
     'lstm_size',
     'batch_size',
     'max_length',  # input will be truncated/padded to this number of tokens
     'dropout'
    ])


def read_vocabulary(vocabulary_file: str) -> dict:
    with open(vocabulary_file) as f:
        return {tok.strip(): i+1 for i, tok in enumerate(f.readlines())}


def vectorise(token_str: str, vocabulary: dict) -> list:
    return [vocabulary.get(tok, len(vocabulary)+1) for tok in token_str.split()]


def make_model(hyperparams: Hyperparameters):
    model = keras.models.Sequential([
        keras.layers.Embedding(int(hyperparams.vocab_size),
                               int(hyperparams.embed_dim),
                               input_length=int(hyperparams.max_length)),
        keras.layers.CuDNNLSTM(int(hyperparams.lstm_size), return_sequences=True, stateful=False),
        keras.layers.Dropout(hyperparams.dropout),
        keras.layers.CuDNNLSTM(int(hyperparams.lstm_size), return_sequences=True, stateful=False, name='final_lstm'),
        keras.layers.TimeDistributed(keras.layers.Dense(hyperparams.vocab_size, activation='softmax'))
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy'])
    return model
