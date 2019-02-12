import collections
import os

import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.svm


Data = collections.namedtuple('Data', ['x_train', 'y_train', 'x_test', 'y_test', 'id_test'])

POOLING_TYPES = ['mean', 'sd', 'mean+sd']


def load_datasets(data_dir: str, split: str, vectors, pooling_type: str) -> Data:
    train_file = os.path.join(data_dir, "{}_train.csv".format(split))
    test_file = os.path.join(data_dir, "{}_test.csv".format(split))
    x_train, y_train, _ = load_dataset(train_file, vectors, pooling_type)
    x_test, y_test, id_test = load_dataset(test_file, vectors, pooling_type)
    return Data(x_train, y_train, x_test, y_test, id_test)


def load_dataset(index_file: str, vectors, pooling_type: str):
    index = pd.read_csv(index_file)
    dataset = pd.merge(index, vectors, on='file')
    dataset['label'] = dataset['label'].map(lambda x: 1 if x == 'VULNERABLE' else 0)
    if pooling_type == 'mean':
        dataset['vector'] = dataset['mean_vector']
    elif pooling_type == 'sd':
        dataset['vector'] = dataset['sd_vector']
    elif pooling_type == 'mean+sd':
        dataset['vector'] = dataset.apply(lambda row: np.concatenate([row['mean_vector'], row['sd_vector']]).tolist(), axis=1)
    else:
        raise ValueError("unsupported pooling type: %s", pooling_type)
    return df_to_x_y_id(dataset)


def df_to_x_y_id(dataset):
    ids = list(dataset['file'])
    x_tmp = np.array(dataset['vector'])
    x_concat = np.concatenate(x_tmp)
    x = np.reshape(x_concat, (x_tmp.shape[0], x_concat.shape[0] // x_tmp.shape[0]))
    y = np.array(dataset['label'])
    return x, y, ids


def train_and_evaluate(data: Data, spec: dict, n_jobs: int = None) -> dict:
    classifier = _make_classifier(spec['classifier'], n_jobs)
    sample_weight = data.y_train * spec['pos_class_weight'] + np.ones(data.y_train.shape[0]) - data.y_train
    classifier.fit(data.x_train, data.y_train, sample_weight=sample_weight)
    return dict(test=evaluate_classifier(classifier, data.x_test, data.y_test, data.id_test))


def _make_classifier(spec: dict, n_jobs: int):
    if spec['type'] == 'NaiveBayes':
        return sklearn.naive_bayes.GaussianNB()
    elif spec['type'] == 'LR':
        return sklearn.linear_model.LogisticRegression(penalty=spec['penalty'],
                                                       C=spec['C'],
                                                       solver='liblinear')
    elif spec['type'] == 'SVM':
        gamma = spec['kernel']['gamma'] if spec['kernel']['type'] == 'RBF' else 'auto'
        return sklearn.svm.SVC(C=spec['C'],
                               kernel=spec['kernel']['type'],
                               gamma=gamma,
                               probability=True)
    elif spec['type'] == 'RandomForest':
        return sklearn.ensemble.RandomForestClassifier(n_estimators=int(spec['n_estimators']),
                                                       criterion=spec['criterion'],
                                                       n_jobs=n_jobs)
    elif spec['type'] == 'AdaBoost':
        return sklearn.ensemble.AdaBoostClassifier(n_estimators=int(spec['n_estimators']))
    else:
        raise ValueError("invalid classifier type in spec: %s" % str(spec))    


def evaluate_classifier(classifier, x_test, y_test, id_test) -> dict:
    y_prob = classifier.predict_proba(x_test)[:,1] # probability of class 1, i.e. vulnerable
    y_pred = np.round(y_prob)
    return dict(
        roc_auc=sklearn.metrics.roc_auc_score(y_test, y_prob),
        average_precision=sklearn.metrics.average_precision_score(y_test, y_prob),
        precision=sklearn.metrics.precision_score(y_test, y_pred),
        recall=sklearn.metrics.recall_score(y_test, y_pred),
        f1=sklearn.metrics.f1_score(y_test, y_pred),
        accuracy=sklearn.metrics.accuracy_score(y_test, y_pred),
        confusion_matrix=sklearn.metrics.confusion_matrix(y_test, y_pred).tolist(),
        examples=[(id_test[i], int(y_test[i]), float(y_prob[i])) for i in range(len(id_test))]
    )
