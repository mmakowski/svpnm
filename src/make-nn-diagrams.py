#!/usr/bin/env python
import json
import os
import sys

import keras

import crnn
import c_lm
import report_resources

# language model hyperarparameters
LM_VOCAB_SIZE = 10000
LM_BATCH_LENGTH = 128
LM_BATCH_SIZE = 128
LM_HYPERPARAMS = c_lm.Hyperparameters(vocab_size=LM_VOCAB_SIZE,
                                      embed_dim=64,
                                      lstm_size=128,
                                      batch_size=LM_BATCH_SIZE,
                                      max_length=LM_BATCH_LENGTH,
                                      dropout=0.2)

def main(crnn_hyperparam_file: str):
    _draw_crnn_diagram(crnn_hyperparam_file)
    _draw_lm_diagram()


def _draw_crnn_diagram(hyperparam_file: str):
    with open(hyperparam_file) as f:
        hyperparams = crnn.Hyperparameters(**json.load(f))
    # weight keys are loaded from json as strings, convert them to integers
    hyperparams = hyperparams._replace(class_weight={int(c): w for c, w in hyperparams.class_weight.items()})
    model = crnn._create_model(10000, hyperparams)
    with open(os.path.join(report_resources.OUTPUT_DIR, 'crnn-diagram.eps'), 'wb') as f:
        f.write(keras.utils.vis_utils.model_to_dot(model, show_shapes=True).create())


def _draw_lm_diagram():
    model = c_lm.make_model(LM_HYPERPARAMS)
    with open(os.path.join(report_resources.OUTPUT_DIR, 'lm-diagram.eps'), 'wb') as f:
        f.write(keras.utils.vis_utils.model_to_dot(model, show_shapes=True).create())


if __name__ == '__main__':
    main(*sys.argv[1:])