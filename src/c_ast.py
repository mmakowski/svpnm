from collections import namedtuple


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

class Cond(namedtuple('Cond', ['guard', 'stmts'])):
    """`if (<cond>) { <stmts> }`"""
    def code(self, indent_level=0):
        return "\n".join(
            [_indent(indent_level, "if (%s) {" % self.guard.code())] +
            [stmt.code(indent_level+1) for stmt in self.stmts] +
            [_indent(indent_level, "}")]
        )

class BoolAtom(namedtuple('BoolAtom', 'var')):
    """`*<var>`"""
    def code(self):
        return "*%s" % self.var

class BoolNeg(namedtuple('BoolNeg', 'bool')):
    """`!(<bool>)`"""
    def code(self):
        return "!(%s)" % self.bool.code()

class Func(namedtuple('Func', 'stmts')):
    """`void f() { <stmts> }`"""
    def code(self, indent_level=0):
        return "\n".join(
            [_indent(indent_level, "void f() {")] +
            [stmt.code(indent_level+1) for stmt in self.stmts] +
            [_indent(indent_level, "}")]
        )


def full_code(func):
    return PROGRAM_TEMPLATE % func.code()


def ast(func):
    return str(func)


def parse_ast_str(ast_str: str):
    return eval(ast_str)


def _indent(depth, line):
    return "%s%s" % (" " * (depth*INDENT_SPACES), line)
