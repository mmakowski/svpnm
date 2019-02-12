#!/usr/bin/env python

import configparser
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import sklearn.feature_extraction
from tqdm import tqdm

# truncate training set to make discretisation feasible
MAX_NUM_TRAIN_EXAMPLES = 60000

# logging
logging.basicConfig(format='%(asctime)s %(process)s %(levelname)-8s %(message)s', stream=sys.stdout)
log = logging.getLogger()
log.setLevel(logging.INFO)


def main(train_file, test_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_set = _load_dataset(train_file)
    test_set = _load_dataset(test_file)
    vectoriser = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1, 1))
    X_train_text, y_train = _create_inputs(train_set, limit=MAX_NUM_TRAIN_EXAMPLES)
    X_test_text, y_test = _create_inputs(test_set)
    log.info("vectorising...")
    X_train_vec = vectoriser.fit_transform(X_train_text)
    X_test_vec = vectoriser.transform(X_test_text)
    _dump_arff_file(X_train_vec, y_train, os.path.join(output_dir, "train.arff"), vectoriser.get_feature_names())
    _dump_arff_file(X_test_vec, y_test, os.path.join(output_dir, "test.arff"), vectoriser.get_feature_names())


def _load_dataset(dataset_file):
    log.info("loading %s...", dataset_file)
    content_dir = os.path.join(os.path.dirname(dataset_file), 'content')
    data = pd.read_csv(dataset_file)
    data['content'] = data['file'].map(lambda file: _load_content(os.path.join(content_dir, file)))
    return data


def _load_content(file):
    with open(file, 'r') as f:
        return f.read()


def _create_inputs(df, limit=None):
    X = df['content']
    y = df['label'].map({'VULNERABLE': 1, 'NOT_VULNERABLE': 0})
    if limit:
        X = X[:limit]
        y = y[:limit]
    return X, y


def _dump_arff_file(X, y, file_path, feature_names):
    log.info("writing %s...", file_path)
    label_index = X.shape[1]
    y = np.array(y)
    assert len(feature_names) == label_index
    with open(file_path, 'w') as output_file:
        output_file.write("@relation %s\n\n" % file_path.split('/')[-1].replace(".arff", ""))
        for index, name in enumerate(feature_names):
            output_file.write("@attribute %s numeric\n" % _arff_attr_id(index, name))
        output_file.write("@attribute class {0,1}\n\n")
        output_file.write("@data\n\n")
        for row_index in tqdm(range(X.shape[0])):
            output_file.write(_arff_format_row(X[row_index], y[row_index], label_index))


def _arff_attr_id(index, feature_name):
    return "feat_%s_%d" % (re.sub(r"\W", "_", feature_name), index)


def _arff_format_row(row, label, label_index):
    result = "{"
    nonzero_inds = list(row.nonzero()[1])
    nonzero_inds.sort()
    for i in nonzero_inds:
        result += "%d %d," % (i, row[0, i])
    result += "%d %d" % (label_index, label)
    result += "}\n"
    return result


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
