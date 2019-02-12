#!/usr/bin/env python
import abc
import collections
import json
import os
import shutil
import sys

import keras
import numpy as np
from tqdm import tqdm

import c_lm

RNG_SEED = 141511

VOCAB_SIZE = 10000
BATCH_LENGTH = 128
BATCH_SIZE = 128
ES_MIN_DELTA = 0.000001
ES_PATIENCE = 5 
MAX_EPOCHS = 200
VALID_PERCENT = 5 # the percentage of data to use for validation


HYPERPARAMS = c_lm.Hyperparameters(vocab_size=VOCAB_SIZE,
                                   embed_dim=64,
                                   lstm_size=128,
                                   batch_size=BATCH_SIZE,
                                   max_length=BATCH_LENGTH,
                                   dropout=0.2)


def main(corpus_file: str, output_dir: str):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    vocabulary_file = os.path.join(output_dir, 'vocabulary.txt')
    vocabulary = _make_vocabulary(corpus_file, vocabulary_file)
    model = c_lm.make_model(HYPERPARAMS)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=ES_MIN_DELTA,
                                                   patience=ES_PATIENCE)
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'model.hdf5'),
                                                       save_best_only=True)
    sampler = Sampler(vocabulary)
    history = model.fit_generator(TrainSequence(corpus_file, vocabulary),
                                  validation_data=ValidSequence(corpus_file, vocabulary),
                                  epochs=MAX_EPOCHS,
                                  shuffle=False,
                                  callbacks=[early_stopping, model_checkpoint, sampler])
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f)


def _make_vocabulary(corpus_file: str, vocabulary_file: str) -> dict:
    vocab_counter = collections.Counter()
    with open(corpus_file) as f:
        for line in tqdm(f, desc="calculating vocabulary"):
            vocab_counter.update(line.rstrip().split(" "))
    vocab_list = [t for t, _ in vocab_counter.most_common(VOCAB_SIZE-2)] # -2 to make space for unk and pad
    with open(vocabulary_file, 'w') as f:
        f.write("\n".join(vocab_list))
    return c_lm.read_vocabulary(vocabulary_file)



class BatchSequence(keras.utils.Sequence):
    __metaclass__ = abc.ABCMeta

    def __init__(self, corpus_file: str, vocabulary: dict):
        self.read_batches(corpus_file, vocabulary)

    def read_batches(self, corpus_file: str, vocabulary: dict):
        lines = []
        cur_line_x = np.zeros((BATCH_LENGTH))
        cur_line_y = np.zeros((BATCH_LENGTH))
        tok_idx = 0
        with open(corpus_file) as f:
            for i, l in tqdm(enumerate(f)):
                if self.include_example_idx(i):
                    x_tokens = c_lm.vectorise(l.rstrip(), vocabulary)
                    y_tokens = x_tokens[1:] + [0]
                    for x_tok, y_tok in zip(x_tokens, y_tokens):
                        if tok_idx == BATCH_LENGTH:
                            lines.append([cur_line_x, cur_line_y])
                            cur_line_x = np.zeros((BATCH_LENGTH))
                            cur_line_y = np.zeros((BATCH_LENGTH))
                            tok_idx = 0
                        cur_line_x[tok_idx] = x_tok
                        cur_line_y[tok_idx] = y_tok
                        tok_idx += 1
        lines.append((cur_line_x, cur_line_y))
        self.lines = np.array(lines)


    def __len__(self):
        return len(self.lines) // BATCH_SIZE

    def __getitem__(self, batch_idx: int):
        x, y = np.swapaxes(self.lines[BATCH_SIZE * batch_idx : BATCH_SIZE * (batch_idx+1)], 0, 1)
        # y needs an additional dimension for sparse categorical cross-entropy
        return x, y.reshape(y.shape[0], y.shape[1], 1)

    @abc.abstractmethod
    def include_example_idx(self, i: int) -> bool:
        return False


class TrainSequence(BatchSequence):
    def __init__(self, corpus_file: str, vocabulary: dict):
        super(TrainSequence, self).__init__(corpus_file, vocabulary)
        self.rng = np.random.RandomState(RNG_SEED)
        self.rng.shuffle(self.lines)
    
    def include_example_idx(self, i: int) -> bool:
        return i % 100 >= VALID_PERCENT

    def on_epoch_end(self):
        self.rng.shuffle(self.lines)


class ValidSequence(BatchSequence):
    def include_example_idx(self, i: int) -> bool:
        return i % 100 < VALID_PERCENT


# for debugging:
class Sampler(keras.callbacks.Callback):
    def __init__(self, vocabulary):
        self.vocabulary = _invert_vocabulary(vocabulary)

    def on_epoch_end(self, epoch, logs=None):
        inp = keras.preprocessing.sequence.pad_sequences([[np.random.randint(0, VOCAB_SIZE)]],
                                                         maxlen=BATCH_LENGTH,
                                                         padding='post')
        for i in range(BATCH_LENGTH-1):
            outp = self.model.predict(inp, batch_size=1)
            # do not predict pad and unk
            outp[0][i][0] = 0.0
            outp[0][i][VOCAB_SIZE-1] = 0.0
            inp[0][i+1] = np.argmax(outp[0][i])
        print(" ".join([self.vocabulary[i] for i in inp[0]]))


# for debugging:
def _print_batch(batch, vocabulary: dict):
    inv_vocab = _invert_vocabulary(vocabulary)
    x, y = batch
    for i in range(len(x)):
        print("x: " + " ".join([inv_vocab[t] for t in x[i]]))
        print("y: " + " ".join([inv_vocab[t[0]] for t in y[i]]))
        print("----")
    print("====")


def _invert_vocabulary(vocabulary: dict) -> dict:
    inv_vocabulary = {i: w for w, i in vocabulary.items()}
    inv_vocabulary[0] = '<PAD>'
    inv_vocabulary[len(vocabulary)+1] = '<UNK>'
    return inv_vocabulary


if __name__ == '__main__':
    main(*sys.argv[1:])
