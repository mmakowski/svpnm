#!/usr/bin/env python

from collections import deque, namedtuple
import os
import shutil
import string
import sys

import numpy as np
from tqdm import tqdm

import project
from c_ast import *

CONTENT_DIR = os.path.join(project.UAF_DISTANT_DIR, 'content')
TRAIN_FILE = os.path.join(project.UAF_DISTANT_DIR, 'train.csv')
VALID_FILE = os.path.join(project.UAF_DISTANT_DIR, 'valid.csv')
TEST_FILE = os.path.join(project.UAF_DISTANT_DIR, 'test.csv')


NUM_EXAMPLES = 100
RNG_SEED = 14914
GEN_PARAMS = dict(
    vuln_fraction=0.5,
    num_vars=dict(n=20, p=0.2), # binomial distribution parameters
    var_name_length=dict(n=10, p=0.4), # binomial distribution parameters
    min_uaf_distance=5, # minimum distance in statements between free and subsequent use
    extra_uaf_distance=dict(p=0.2) # geometric distribution parameters
)


def main():
    if os.path.exists(project.UAF_DISTANT_DIR):
        shutil.rmtree(project.UAF_DISTANT_DIR)
    os.makedirs(CONTENT_DIR)
    rng = np.random.RandomState(RNG_SEED)

    metadata_file = _start_metadata_file(TRAIN_FILE)
    for i in tqdm(range(NUM_EXAMPLES)):
        if i == int(0.6 * NUM_EXAMPLES):
            metadata_file.close()
            metadata_file = _start_metadata_file(VALID_FILE)
        elif i == int(0.8 * NUM_EXAMPLES):
            metadata_file.close()
            metadata_file = _start_metadata_file(TEST_FILE)
        _generate_new_file(i, rng, metadata_file)
    metadata_file.close()


def _generate_new_file(i: int, rng, metadata_file):
    func, vulnerable = _random_func(rng)
    file_name_prefix = "{:08d}".format(i)
    c_file_name = "%s.c" % file_name_prefix
    with open(os.path.join(CONTENT_DIR, c_file_name), 'w') as cf:
        cf.write(full_code(func))
    ast_file_name = "%s.ast" % file_name_prefix
    with open(os.path.join(CONTENT_DIR, ast_file_name), 'w') as af:
        af.write(ast(func))
    metadata_file.write("%s,%s\n" % (c_file_name, _vuln_label(vulnerable)))


def _start_metadata_file(path: str):
    f = open(path, 'w')
    f.write("file,label\n")
    return f


def _random_func(rng):
    vulnerable = rng.choice([True, False], p=[GEN_PARAMS['vuln_fraction'], 1-GEN_PARAMS['vuln_fraction']])
    num_vars = 2 + rng.binomial(**GEN_PARAMS['num_vars'])
    variables = [_var_name(i, rng) for i in range(num_vars)]
    selected_variable = rng.choice(variables, size=1)[0]
    other_variables = [var for var in variables if var != selected_variable]
    decls = [VarDecl(var) for var in variables]
    selected_variable_history = _var_history(selected_variable, vulnerable, rng)
    other_variable_histories = [_var_history(var, False, rng) for var in other_variables]
    stmts = _combine_var_histories(selected_variable_history, other_variable_histories, rng)
    func = Func(decls + stmts)
    return func, vulnerable


def _var_name(seq: int, rng) -> str:
    id_chars = list(string.ascii_lowercase + "_")
    return "".join(rng.choice(id_chars, size=1+rng.binomial(**GEN_PARAMS['var_name_length']))) + str(seq)


def _var_history(var: str, uaf: bool, rng) -> list:
    # ensure there is enough statements available to guarantee the distance between free and use, 
    # and then between use and the end of the function
    num_assigns = 2 * GEN_PARAMS['min_uaf_distance'] +\
                  rng.geometric(**GEN_PARAMS['extra_uaf_distance']) +\
                  rng.geometric(**GEN_PARAMS['extra_uaf_distance'])
    history = [Malloc(var)] +\
              [Assign(var, rng.randint(0, 128)) for _ in range(num_assigns-1)]
    if uaf:
        history += [Free(var), Assign(var, rng.randint(0, 128))]
    else:
        history += [Assign(var, rng.randint(0, 128)), Free(var)]
    return history


def _combine_var_histories(selected_variable_history: list, other_variable_histories: list, rng) -> list:
    """Requirements:
    1. The last two statements in the history of selected variable must be separated by at least
       min_uaf_distance statements (pad);
    2. The last statement in the history of selected variable must be followed by at least
       min_uaf_distance statements (suffix);
    """
    extra_other_stmts = sum([len(h) for h in other_variable_histories]) - 2 * GEN_PARAMS['min_uaf_distance']
    prefix_length = rng.randint(extra_other_stmts//2, extra_other_stmts+1) # use the majority of extra statements for the prefix
    pad_length = GEN_PARAMS['min_uaf_distance'] + rng.randint(0, extra_other_stmts-prefix_length+1)
    suffix_length = GEN_PARAMS['min_uaf_distance'] + (extra_other_stmts - (prefix_length-GEN_PARAMS['min_uaf_distance']) - pad_length)
    assert prefix_length + pad_length + suffix_length == 2*GEN_PARAMS['min_uaf_distance'] + extra_other_stmts
    
    # prepare queueues
    hists = [deque(hist) for hist in other_variable_histories]
    
    # first, select suffix statements
    suffix = []
    while len(suffix) < suffix_length:
        suffix.append(hists[rng.choice(len(hists))].pop())
        hists = [hist for hist in hists if len(hist) > 0]
    suffix.reverse()

    # then, select pad statements
    pad = []
    while len(pad) < pad_length:
        pad.append(hists[rng.choice(len(hists))].pop())
        hists = [hist for hist in hists if len(hist) > 0]
    pad.reverse()

    # finally, select prefix statements, including the benign statements from the selected variable's history
    prefix = []
    hists += [deque(selected_variable_history[:-2])]
    while len(hists) > 0:
        prefix.append(hists[rng.choice(len(hists))].popleft())
        hists = [hist for hist in hists if len(hist) > 0]

    return prefix + [selected_variable_history[-2]] + pad + [selected_variable_history[-1]] + suffix


def _vuln_label(vulnerable: bool) -> str:
    if vulnerable:
        return 'VULNERABLE'
    else:
        return 'NOT_VULNERABLE'


if __name__ == '__main__':
    main()
