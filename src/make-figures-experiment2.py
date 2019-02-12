#!/usr/bin/env python
import json
import os
import sys

import pandas as pd
import seaborn as sns

import report_resources
import matplotlib.pyplot as plt

RNG_SEED = 1319


def main(data_dir: str, results_dir: str):
    report_resources.ensure_output_dir_exists()
    _write_pr_plot(_read_split_metrics(results_dir), _read_train_fractions(data_dir))
 

def _write_pr_plot(split_metrics, train_fractions):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.lineplot(data=split_metrics,
                 x='Release',
                 y='Score',
                 hue='Metric',
                 style='Metric', markers=['o', 'o'], dashes=False, # a hack to get markers unaffected by metric
                 ax=ax)
    sns.barplot(data=train_fractions,
                x='Release',
                y='Fraction of data used for training',
                facecolor='#eeeeee88',
                edgecolor='#eeeeee88',
                ax=ax)
    ax.set(ylabel='Score / Fraction of data used for training')
    fig.savefig(os.path.join(report_resources.OUTPUT_DIR, '02-drf-linux-splits-pr.pdf'), bbox_inches='tight')


def _read_train_fractions(data_dir: str):
    FRACTION_COLS = ('Release', 'Fraction of data used for training')
    fractions = pd.DataFrame([], columns=FRACTION_COLS)
    splits = [f.split("_")[0]
              for f in os.listdir(data_dir)
              if f.endswith("_train.csv")]
    for i, split in enumerate(sorted(splits)):
        train_examples = _count_examples(data_dir, split, 'train')
        test_examples = _count_examples(data_dir, split, 'test')
        train_fraction = train_examples / (train_examples + test_examples)
        fractions = pd.concat([fractions, pd.DataFrame([(i+1, train_fraction)], columns=FRACTION_COLS)])
    return fractions


def _read_split_metrics(results_dir: str):
    METRICS_COLS = ('Release', 'Metric', 'Score')
    results = pd.DataFrame([], columns=METRICS_COLS)
    for i, split in enumerate(sorted(os.listdir(results_dir))):
        split_stats = _read_stats(os.path.join(results_dir, split, 'stats.txt'))
        # note: we start counting releases from 0, because due to some seaborn weirdness the metrics are shifted
        # by 1 when combined with training fractions
        split_results = pd.DataFrame([(i, metric, float(split_stats[metric])) for metric in ['precision', 'recall']],
                                     columns=METRICS_COLS)
        results = pd.concat([results, split_results])
    return results


def _read_stats(stats_file: str) -> dict:
    with open(stats_file) as f:
        return {key: value
                for line in f.readlines()
                for key, value in [line.rstrip().split(": ")]}


def _count_examples(data_dir: str, split: str, file_type: str) -> int:
    with open(os.path.join(data_dir, "{}_{}.csv".format(split, file_type))) as f:
        return sum(1 for line in f) - 1


if __name__ == '__main__':
    main(*sys.argv[1:])
