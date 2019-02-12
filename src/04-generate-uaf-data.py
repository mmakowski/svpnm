#!/usr/bin/env python

from collections import deque, namedtuple
import os
import shutil
import string
import sys

import numpy as np
from tqdm import tqdm

import project

CONTENT_DIR = os.path.join(project.UAF_DIR, 'content')
TRAIN_FILE = os.path.join(project.UAF_DIR, 'train.csv')
VALID_FILE = os.path.join(project.UAF_DIR, 'valid.csv')
TEST_FILE = os.path.join(project.UAF_DIR, 'test.csv')

NUM_EXAMPLES = 100000
RNG_SEED = 14914
GEN_PARAMS = dict(
    vuln_fraction=0.5,
    num_vars=dict(n=20, p=0.1), # binomial distribution parameters
    var_name_length=dict(n=3, p=0.8), # binomial distribution parameters
    num_vuln_vars=dict(p=0.8), # geometric distribution parameter
    var_num_ubfs=dict(p=0.3), # geometric distribution parameter
    var_num_uafs=dict(p=0.9) # geometric distribution parameter
)

INDENT_SPACES = 2
PROGRAM_TEMPLATE = """
#include <stdlib.h>

%s

int main() {
  f();
  return 0;
}
"""

# AST nodes

class VarDecl(namedtuple('VarDecl', 'var')):
    """ `char * <var>;`"""
    def code(self, indent_level=0):
        return _indent(indent_level, "char * %s;" % self.var)


class Malloc(namedtuple('Malloc', 'var')):
    """`<var> = malloc(sizeof(char));`"""
    def code(self, indent_level=0):
        return _indent(indent_level, "%s = malloc(sizeof(char));" % self.var)


class Free(namedtuple('Free', 'var')):
    """`free(<var>);`"""
    def code(self, indent_level=0):
        return _indent(indent_level, "free(%s);" % self.var)


class Assign(namedtuple('Assign', ['var', 'num'])):
    """`*<var> = <num>;`"""
    def code(self, indent_level=0):
        return _indent(indent_level, "*%s = %d;" % (self.var, self.num))


class Func(namedtuple('Func', 'stmts')):
    """`void f() { <stmts> }`"""
    def code(self, indent_level=0):
        return "\n".join(
            [_indent(indent_level, "void f() {")] +
            [stmt.code(indent_level+1) for stmt in self.stmts] +
            [_indent(indent_level, "}")]
        )


def main():
    if os.path.exists(project.UAF_DIR):
        shutil.rmtree(project.UAF_DIR)
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
        cf.write(_full_code(func))
    ast_file_name = "%s.ast" % file_name_prefix
    with open(os.path.join(CONTENT_DIR, ast_file_name), 'w') as af:
        af.write(_ast(func))
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
    var_histories = [_var_history(var, var in vuln_variables, rng) for var in variables]
    stmts = _combine_var_histories(var_histories, rng)
    func = Func(decls +
                stmts)
    return func, vulnerable


def _var_name(seq, rng):
    id_chars = list(string.ascii_lowercase)
    return id_chars[seq]


def _var_history(var, uaf, rng):
    history = [Malloc(var)] +\
              [Assign(var, rng.randint(0, 128)) for _ in range(rng.geometric(**GEN_PARAMS['var_num_ubfs']))] +\
              [Free(var)]
    if uaf:
        history += [Assign(var, rng.randint(0, 128)) for _ in range(rng.geometric(**GEN_PARAMS['var_num_uafs']))]
    return history


def _combine_var_histories(var_histories, rng):
    hists = [deque(hist) for hist in var_histories]
    combined = []
    while len(hists) > 0:
        combined.append(hists[rng.choice(len(hists))].popleft())
        hists = [hist for hist in hists if len(hist) > 0]
    return combined


def _vuln_label(vulnerable):
    if vulnerable:
        return 'VULNERABLE'
    else:
        return 'NOT_VULNERABLE'


def _full_code(func):
    return PROGRAM_TEMPLATE % func.code()


def _indent(depth, line):
    return "%s%s" % (" " * (depth*INDENT_SPACES), line)


def _ast(func):
    return str(func)

if __name__ == '__main__':
    main()
