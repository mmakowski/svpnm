#!/usr/bin/env python

import os
import sys

import pandas as pd

import uaf_stats


def main(test_file: str, output_dir: str):
    content_dir = os.path.join(os.path.dirname(test_file), 'content')
    data = pd.read_csv(test_file).query("label == 'VULNERABLE'")
    data['min_uaf_distance'] = data['file'].apply(lambda f: uaf_stats.min_uaf_distance(content_dir, f))
    data['is_last_stmt_assignment'] = data['file'].apply(lambda f: uaf_stats.is_last_stmt_assignment(content_dir, f))
    data['length_in_statements'] = data['file'].apply(lambda f: uaf_stats.length_in_statements(content_dir, f))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(str(data.groupby('is_last_stmt_assignment').agg({
                'min_uaf_distance': ['count', 'mean', 'std'],
                'length_in_statements': ['mean', 'std']
            })))


if __name__ == '__main__':
    main(*sys.argv[1:])
