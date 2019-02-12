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

CONTENT_DIR = os.path.join(project.UAF_NOISY_DIR, 'content')
TRAIN_FILE = os.path.join(project.UAF_NOISY_DIR, 'train.csv')
VALID_FILE = os.path.join(project.UAF_NOISY_DIR, 'valid.csv')
TEST_FILE = os.path.join(project.UAF_NOISY_DIR, 'test.csv')

# generator state
GenState = namedtuple('GenState', ['unallocated', # variables that have not yet been allocated
                                   'allocated', # variables allocated but not initialised
                                   'initialised', # initialised variables
                                   'freed', # freed variables
                                   'not_yet_vulnerable', # variables that should be made vulnerable but are not yet
                                   'vulnerable', # variables that have been made vulnerable
                                   'depth'])
# generator automaton actions consist of the AST statments plus these:
CondEnd = 'CondEnd'
AssignAfterFree = 'AssignAfterFree'
Finish = 'Finish'

NUM_EXAMPLES = 100000
RNG_SEED = 14914
GEN_PARAMS = dict(
    vuln_fraction=0.5,
    num_vars=dict(n=20, p=0.1), # binomial distribution parameters
    num_vuln_vars=dict(p=0.8), # geometric distribution parameter
    var_name_length=dict(n=10, p=0.4), # binomial distribution parameters
    action_freqs={ # relative frequencies of different actions
        Malloc: 3,
        Free: 3,
        Assign: 8,
        Cond: 6,
        CondEnd: 7,
        AssignAfterFree: 6,
        Finish: 1
    },
    max_cond_depth=3
)


def main():
    if os.path.exists(project.UAF_NOISY_DIR):
        shutil.rmtree(project.UAF_NOISY_DIR)
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


def _generate_new_file(i, rng, metadata_file):
    func, vulnerable = _random_func(rng)
    file_name_prefix = "{:08d}".format(i)
    c_file_name = "%s.c" % file_name_prefix
    with open(os.path.join(CONTENT_DIR, c_file_name), 'w') as cf:
        cf.write(full_code(func))
    ast_file_name = "%s.ast" % file_name_prefix
    with open(os.path.join(CONTENT_DIR, ast_file_name), 'w') as af:
        af.write(ast(func))
    metadata_file.write("%s,%s\n" % (c_file_name, _vuln_label(vulnerable)))


def _start_metadata_file(path):
    f = open(path, 'w')
    f.write("file,label\n")
    return f


def _random_func(rng):
    vulnerable = rng.choice([True, False], p=[GEN_PARAMS['vuln_fraction'], 1-GEN_PARAMS['vuln_fraction']])
    num_vars = 1 + rng.binomial(**GEN_PARAMS['num_vars'])
    variables = [_var_name(i, rng) for i in range(num_vars)]
    if vulnerable:
        vuln_variables = rng.choice(variables,
                                    size=min(len(variables), rng.geometric(**GEN_PARAMS['num_vuln_vars'])),
                                    replace=False)
    else:
        vuln_variables = []
    decls = [VarDecl(var) for var in variables]
    start_state = GenState(unallocated=set(variables),
                           allocated=set(),
                           initialised=set(),
                           freed=set(),
                           not_yet_vulnerable=set(vuln_variables),
                           vulnerable=set(),
                           depth=0)
    stmts, _ = _gen_statements(rng, state=start_state)
    func = Func(decls +
                stmts)
    return func, vulnerable


def _var_name(seq, rng):
    id_chars = list(string.ascii_lowercase + "_")
    return "".join(rng.choice(id_chars, size=rng.binomial(**GEN_PARAMS['var_name_length']))) + str(seq)


def _gen_statements(rng, state, stmts=None):
    if stmts is None:
        stmts = []
    available_actions = _available_actions(state)
    action_probs = _weights_to_probs([GEN_PARAMS['action_freqs'][action] for action in available_actions])
    action = rng.choice(available_actions, p=action_probs)
    if action == Malloc:
        var = rng.choice(list(state.unallocated))
        stmts.append(Malloc(var))
        state.allocated.add(var)
        state.unallocated.remove(var)
        return _gen_statements(rng, state, stmts)
    if action == Assign:
        var = rng.choice(list(state.allocated.union(state.initialised)))
        stmts.append(Assign(var, rng.randint(0, 128)))
        state.initialised.add(var)
        if var in state.allocated: state.allocated.remove(var)
        return _gen_statements(rng, state, stmts)
    elif action == Free:
        var = rng.choice(list(state.allocated.union(state.initialised)))
        stmts.append(Free(var))
        state.freed.add(var)
        if var in state.initialised: state.initialised.remove(var)
        if var in state.allocated: state.allocated.remove(var)
        return _gen_statements(rng, state, stmts)
    elif action == Cond:
        var = rng.choice(list(state.initialised))
        inner_stmts, new_state = _gen_statements(rng, state._replace(depth=state.depth+1))
        stmts.append(Cond(_gen_bool(rng, var), inner_stmts))
        return _gen_statements(rng, new_state, stmts)
    elif action == AssignAfterFree:
        var = rng.choice(list(state.not_yet_vulnerable.intersection(state.freed)))
        stmts.append(Assign(var, rng.randint(0, 128)))
        state.vulnerable.add(var)
        if var in state.not_yet_vulnerable: state.not_yet_vulnerable.remove(var)
        return _gen_statements(rng, state, stmts)
    elif action == CondEnd:
        return stmts, state._replace(depth=state.depth-1)
    elif action == Finish:
        return stmts, state
    else:
        raise ValueError("unsupported action: %s" % str(action))
        

def _gen_bool(rng, var):
    if rng.choice(['pos', 'neg'], p=[0.8, 0.2]) == 'pos':
        return BoolAtom(var)
    else:
        return BoolNeg(_gen_bool(rng, var))


def _available_actions(state):
    actions = set()
    if len(state.unallocated) > 0 and state.depth == 0:
        actions.add(Malloc)
    if len(state.allocated) > 0:
        actions.add(Assign)
        if state.depth == 0:
            actions.add(Free)
    if len(state.initialised) > 0:
        actions.add(Assign)
        if state.depth < GEN_PARAMS['max_cond_depth']:
            actions.add(Cond)
        if state.depth == 0:
            actions.add(Free)
    if len(state.freed.intersection(state.not_yet_vulnerable)) > 0 and \
       state.depth == 0:
        actions.add(AssignAfterFree)
    if len(state.unallocated) == 0 and \
       len(state.allocated) == 0 and \
       len(state.initialised) == 0 and \
       len(state.not_yet_vulnerable) == 0:
        actions.add(Finish)
    if state.depth > 0:
        actions.add(CondEnd)
    return list(actions)


def _weights_to_probs(weights):
    return np.array(weights) / np.sum(weights)


def _vuln_label(vulnerable):
    if vulnerable:
        return 'VULNERABLE'
    else:
        return 'NOT_VULNERABLE'


if __name__ == '__main__':
    main()
