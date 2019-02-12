#!/usr/bin/env python
import json
import os
import sys

import pandas as pd
import seaborn as sns
import sklearn.metrics

import report_resources
import matplotlib.pyplot as plt

SCORES_COLS = ('id', 'label', 'score')
METRICS_COLS = ('Release', 'Model', 'Metric', 'Score')


def main(drf_results_dir: str,
         crnn_results_dir: str,
         lmrf_results_dir: str,
         output_file: str):
    report_resources.ensure_output_dir_exists()
    drf_split_metrics = _read_drf_split_metrics(drf_results_dir)
    crnn_scores = _read_scores(crnn_results_dir)
    crnn_split_precisions = _model_pr_at_set_reference(drf_split_metrics, crnn_scores, 'CRNN', reference_metric='recall')
    crnn_split_recalls = _model_pr_at_set_reference(drf_split_metrics, crnn_scores, 'CRNN', reference_metric='precision')
    lmrf_scores = _read_scores(lmrf_results_dir)
    lmrf_split_precisions = _model_pr_at_set_reference(drf_split_metrics, lmrf_scores, 'LMRF', reference_metric='recall')
    lmrf_split_recalls = _model_pr_at_set_reference(drf_split_metrics, lmrf_scores, 'LMRF', reference_metric='precision')
    _write_pr_plot(pd.concat([crnn_split_precisions,
                              crnn_split_recalls,
                              lmrf_split_precisions,
                              lmrf_split_recalls,
                              drf_split_metrics]),
                   output_file)
 

def _write_pr_plot(metrics, output_file: str):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.lineplot(data=metrics,
                 x='Release',
                 y='Score',
                 hue='Metric',
                 style='Model',
                 ci=None,
                 ax=ax)
    fig.savefig(os.path.join(output_file), bbox_inches='tight')


def _model_pr_at_set_reference(drf_split_metrics, crnn_scores: dict, model_name: str, reference_metric: str):
    variable_metric = 'precision' if reference_metric == 'recall' else 'recall'
    return pd.DataFrame([(release, model_name, variable_metric, _pr_at_set_reference(run_scores,
                                                                                     reference_metric,
                                                                                     variable_metric,
                                                                                     reference_score))
                         for release in crnn_scores.keys()
                         for reference_score in [drf_split_metrics[(drf_split_metrics['Release'] == release) &\
                                                                   (drf_split_metrics['Metric'] == reference_metric)]['Score'].values[0]]
                         for run_scores in crnn_scores[release].values()
                        ],
                        columns=METRICS_COLS)


def _pr_at_set_reference(scores, reference_metric: str, variable_metric: str, reference_score: float) -> float:
    y_true = scores['label']
    y_scores = scores['score']
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_scores)
    curve = dict(precision=precision, recall=recall)
    i_for_reference_score = min(range(len(curve[reference_metric])), key=lambda i: abs(curve[reference_metric][i] - reference_score))
    return curve[variable_metric][i_for_reference_score]
    

def _read_drf_split_metrics(results_dir: str):
    results = pd.DataFrame([], columns=METRICS_COLS)
    for i, split in enumerate(sorted(os.listdir(results_dir))):
        split_stats = _read_stats(os.path.join(results_dir, split, 'stats.txt'))
        split_results = pd.DataFrame([(i+1, 'DRF', metric, float(split_stats[metric])) for metric in ['precision', 'recall']],
                                     columns=METRICS_COLS)
        results = pd.concat([results, split_results])
    return results


def _read_stats(stats_file: str) -> dict:
    with open(stats_file) as f:
        return {key: value
                for line in f.readlines()
                for key, value in [line.rstrip().split(": ")]}


def _read_scores(results_dir: str) -> dict:
    return {release+1: _read_split_scores(results_dir, split)
            for release, split in enumerate(sorted(os.listdir(results_dir)))}


def _read_split_scores(results_dir: str, split: str) -> dict:
    with open(os.path.join(results_dir, split, 'run_ids.txt')) as f:
        runs = [line.rstrip() for line in f.readlines()]
    return {run: _read_run_scores(results_dir, split, run) for run in runs}


def _read_run_scores(results_dir: str, split: str, run: str):
    with open(os.path.join(results_dir, split, "results_{}.json".format(run))) as f:
        examples = json.load(f)['test']['examples']
    return pd.DataFrame(examples, columns=SCORES_COLS)


if __name__ == '__main__':
    main(*sys.argv[1:])
