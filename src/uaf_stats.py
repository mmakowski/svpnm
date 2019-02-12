"""
Functions to compute statistics on synthetic use-after-free examples. Used to check if
the generated data is sensible and to show correlation between certain features
of examples and model scores.
"""
import os

from c_ast import *


def length_in_statements(content_dir: str, c_file_name: str) -> int:
    func_ast = _read_ast(content_dir, c_file_name)
    return _ast_size_in_statements(func_ast)


def min_uaf_distance(content_dir: str, c_file_name: str) -> int:
    free_locs = {}
    use_locs = {}
    func_ast = _read_ast(content_dir, c_file_name)
    loc = 1
    for stmt in func_ast.stmts:
        if stmt.__class__ == Free:
            free_locs[stmt.var] = loc
        elif stmt.__class__ == Assign:
            if stmt.var not in use_locs:
                use_locs[stmt.var] = []
            use_locs[stmt.var].append(loc)
        # for conditionals, count all statements in the conditional to approximate token distance rathern than AST distance
        elif stmt.__class__ == Cond:
            loc += _ast_size_in_statements(stmt)
        loc += 1
    uaf_distances = [use_loc - free_locs[var]
                     for var in use_locs
                     for use_loc in use_locs[var]
                     if use_loc > free_locs[var]]
    return min(uaf_distances)


def is_last_stmt_assignment(content_dir: str, c_file_name: str) -> bool:
    func_ast = _read_ast(content_dir, c_file_name)
    return func_ast.stmts[-1].__class__ == Assign


def _ast_size_in_statements(cond_stmt) -> int:
    size = 0
    for stmt in cond_stmt.stmts:
        if stmt.__class__ == Cond:
            size += _ast_size_in_statements(stmt)
        size += 1
    return size


def _read_ast(content_dir: str, c_file_name: str):
    with open(os.path.join(content_dir, c_file_name.replace(".c", ".ast"))) as f:
        return parse_ast_str(f.read())


