#!/usr/bin/env python
import json
import os
import sys

import numpy as np
import pandas as pd
import sklearn.metrics

import report_resources
import matplotlib.pyplot as plt
import matplotlib


SCORES_COLS = ('id', 'label', 'score')
LINUX_C_FiLES = 20630 # in release 3.18
VULNERABLE_FILE_PROPORTION = 0.03
MIN_PRECISION = 0.5


def main(run_ids_file: str):
    report_resources.ensure_output_dir_exists()
    run_scores = _read_split_scores(run_ids_file)
    run_pr_curves = {run: _pr_curve(scores) for run, scores in run_scores.items()}
    vulnerable_c_files = VULNERABLE_FILE_PROPORTION * LINUX_C_FiLES
    all_metrics = _aggregate(pd.concat([_add_alert_volume(pr_curve, vulnerable_c_files) for pr_curve in run_pr_curves.values()]))
    _write_plot(all_metrics, output_file=os.path.join(report_resources.OUTPUT_DIR, 'lmc-alert-volume.pdf'))
 

def _write_plot(metrics, output_file: str):
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    ax0 = plt.subplot(gs[0])
    ax0.plot(metrics.index, metrics['Precision'], label='Precision')
    ax0.plot(metrics.index, metrics['Recall'], label='Recall')
    ax0.set_ylabel('Score')
    ax0.legend(loc='best')

    ax1 = plt.subplot(gs[1], sharex=ax0)
    tp_color = plt.get_cmap(report_resources.PALETTE)(2)
    fp_color = plt.get_cmap(report_resources.PALETTE)(3)
    ax1.bar(metrics.index, metrics['True positives'],
            width=0.001, color=tp_color, edgecolor=tp_color,
            label='True positives')
    ax1.bar(metrics.index, metrics['False positives'], bottom=metrics['True positives'],
            width=0.001, color=fp_color, edgecolor=fp_color,
            label='False positives')
    ax1.set_ylabel('Alerted files')
    ax1.set_xlabel('Threshold')
    ax1.legend(loc='best')

    plt.subplots_adjust(hspace=0.0)

    fig.savefig(os.path.join(output_file), bbox_inches='tight')


def _pr_curve(scores):
    y_true = scores['label']
    y_scores = scores['score']
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(y_true, y_scores)
    threshold = list(threshold) + [1.0]
    return pd.DataFrame(dict(Threshold=threshold,
                             Precision=precision,
                             Recall=recall))


def _add_alert_volume(pr_curve, vulnerable_c_files: int):
    def volume_tp(row):
        if row['Precision'] < MIN_PRECISION:
            return None
        else:
            return row['Recall'] * vulnerable_c_files

    def volume_fp(row):
        if row['Precision'] < MIN_PRECISION:
            return None
        else:
            return row['True positives'] * (1/row['Precision'] - 1)

    result = pr_curve
    result['True positives'] = result.apply(volume_tp, axis=1)
    result['False positives'] = result.apply(volume_fp, axis=1)
    return result


def _aggregate(metrics):
    return metrics.groupby('Threshold').agg({'Precision': np.mean, 
                                             'Recall': np.mean, 
                                             'True positives': np.mean,
                                             'False positives': np.mean})


def _read_split_scores(run_ids_file: str) -> dict:
    split_dir = os.path.dirname(run_ids_file)
    with open(run_ids_file) as f:
        runs = [line.rstrip() for line in f.readlines()]
    return {run: _read_run_scores(split_dir, run) for run in runs}


def _read_run_scores(split_dir: str, run: str):
    with open(os.path.join(split_dir, "results_{}.json".format(run))) as f:
        examples = json.load(f)['test']['examples']
    return pd.DataFrame(examples, columns=SCORES_COLS)


if __name__ == '__main__':
    main(*sys.argv[1:])
