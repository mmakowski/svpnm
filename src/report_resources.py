import json
import os

import matplotlib
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['figure.figsize'] = [9, 6]
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = 'Set2'
CMAP_HEATMAP = 'Blues'

import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette(PALETTE)

OUTPUT_DIR = 'results/report'


def ensure_output_dir_exists():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def load_learning_curve(results_dir):
    lc_files = [os.path.join(results_dir, fn)
                for fn in os.listdir(results_dir)
                if fn.startswith('learning_curve_')]
    rows = [row 
            for lc_file in lc_files
            for row in _learning_curve_run_rows(lc_file)]
    return pd.DataFrame(rows, columns=['Training set size', 'Metric', 'Score'])


def _learning_curve_run_rows(file: str) -> list:
    rows = []
    with open(file) as f:
        train_sizes = json.load(f)
    for ts in train_sizes:
        for metric in ['precision', 'recall']:
            rows.append((ts['training_examples'], metric, ts['results']['test'][metric]))
    return rows


