#!/usr/bin/env python
from report_resources import *

DRF_UAF_RESULTS_FILE = 'results/01-uaf-drf/stats.txt'
CRNN_UAF_RESULTS_DIR = 'results/02-uaf-lstm'


def main():
    ensure_output_dir_exists()
    _write_pr_boxplot(_load_baseline(), load_learning_curve(CRNN_UAF_RESULTS_DIR))


def _write_pr_boxplot(baseline, results):
    cmap = matplotlib.cm.get_cmap(PALETTE)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hlines(y=baseline['precision'], xmin=-1.0, xmax=7.0, color=cmap(0),
              linewidth=1, linestyles='dashed', zorder=0)
    ax.hlines(y=baseline['recall'], xmin=-1.0, xmax=7.0, color=cmap(1),
              linewidth=1, linestyles='dashed', zorder=0)
    sns.boxplot(data=results, x='Training set size', y='Score', hue='Metric', ax=ax)
    fig.savefig(os.path.join(OUTPUT_DIR, '01-crnn-uaf-pr.pdf'), bbox_inches='tight')


def _load_baseline():
    with open(DRF_UAF_RESULTS_FILE) as f:
        return {kv[0]: float(kv[1].strip())
                for line in f.readlines()
                for kv in [line.split(":")]}


if __name__ == '__main__':
    main()
