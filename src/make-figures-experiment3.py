#!/usr/bin/env python
import json
import os
import sys

import pandas as pd
import seaborn as sns
import sklearn.metrics

import report_resources
import matplotlib.pyplot as plt

METRICS_COLS = ('Release', 'Model', 'Metric', 'Score')


def main(drf_results_dir: str, crnn_results_dir: str, output_file: str, model_name: str):
    report_resources.ensure_output_dir_exists()
    drf_split_metrics = _read_drf_split_metrics(drf_results_dir)
    crnn_split_metrics = _read_crnn_split_metrics(crnn_results_dir, model_name)
    _write_pr_plot(pd.concat([crnn_split_metrics, drf_split_metrics]), output_file)
 

def _write_pr_plot(metrics, output_file: str):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.lineplot(data=metrics,
                 x='Release',
                 y='Score',
                 hue='Metric',
                 style='Model',
                 ci=95,
                 ax=ax)
    fig.savefig(os.path.join(output_file), bbox_inches='tight')


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


def _read_crnn_split_metrics(results_dir: str, model_name: str):
    results = pd.DataFrame([], columns=METRICS_COLS)
    for i, split in enumerate(sorted(os.listdir(results_dir))):
        with open(os.path.join(results_dir, split, 'run_ids.txt')) as f:
            runs = [line.rstrip() for line in f.readlines()]
        for run in runs:
            run_stats = _read_run(os.path.join(results_dir, split, "results_{}.json".format(run)))
            run_results = pd.DataFrame([(i+1, model_name, metric, float(run_stats[metric])) for metric in ['precision', 'recall']],
                                       columns=METRICS_COLS)
            results = pd.concat([results, run_results])
    return results


def _read_run(run_file: str) -> dict:
    with open(run_file) as f:
        return json.load(f)['test']


if __name__ == '__main__':
    main(*sys.argv[1:])
