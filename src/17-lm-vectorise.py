#!/usr/bin/env python
from __future__ import print_function

import logging
import os
import sys

import keras
import numpy as np
import pandas as pd
from tqdm import tqdm

import c_lm
import preprocess


# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def main(data_dir: str, lm_dir: str, output_file: str):
    vectors = _vectorise_files(data_dir, lm_dir)
    vectors.to_pickle(output_file)


def _vectorise_files(data_dir: str, lm_dir: str):
    log.info("loading vocabulary...")
    vocabulary = c_lm.read_vocabulary(os.path.join(lm_dir, 'vocabulary.txt'))
    log.info("loading model...")
    full_model = keras.models.load_model(os.path.join(lm_dir, 'model.hdf5'))
    # we do not need the top, classification layer -- we will only be using the final activations of the top LSTM layer
    language_model = keras.models.Sequential(full_model.layers[:-1])
    content_dir = os.path.join(data_dir, 'content')
    vectors = [(file_name, mean_vector, sd_vector)
               for file_name in tqdm(sorted(os.listdir(content_dir)), desc="vectorising")
               for mean_vector, sd_vector in [_vectorise_file(os.path.join(content_dir, file_name), vocabulary, language_model)]]
    return pd.DataFrame(vectors, columns=('file', 'mean_vector', 'sd_vector'))


def _vectorise_file(c_file: str, vocabulary: dict, language_model):
    with open(c_file) as f:
        denoised_content = preprocess.denoise_c(f.read())
    lm_input = c_lm.vectorise(denoised_content, vocabulary)
    x = _make_batch(lm_input, language_model.inputs[0].shape)
    activations = language_model.predict(x, batch_size=x.shape[0])
    mean_vector = np.mean(activations, axis=(0, 1))
    sd_vector = np.sqrt(np.mean((activations - mean_vector) ** 2, axis=(0, 1)))
    return mean_vector, sd_vector


def _make_batch(token_vectors: list, input_shape):
    batch_length = int(input_shape[1])
    batch_size = max(int(np.ceil(len(token_vectors) / batch_length)), 1)
    pad_size = batch_size * batch_length - len(token_vectors)
    padded = np.zeros(batch_size * batch_length)
    padded[pad_size:] = token_vectors
    return padded.reshape((batch_size, batch_length))


if __name__ == '__main__':
    main(*sys.argv[1:])
