"""Lark grammar + parse() for course-syntax FOL. Identifier case is permissive
so course-style P(x) and TPTP-converted p(X) both parse. Inside a term, a name
is a variable iff bound by an enclosing forall/exists, otherwise a constant.
"""

from lark import Lark, Transformer, v_args
from lark.exceptions import LarkError

from logic.syntax import (
    Term, Formula,
    atom, neg, conj, disj, imp, iff, forall, exists, TOP, BOT,
)

GRAMMAR = r"""
?start: formula
?formula: iff_form
?iff_form: imp_form
         | iff_form "<->" imp_form    -> iff_op
?imp_form: or_form
         | or_form "->" imp_form      -> imp_op
?or_form: and_form
        | or_form "\\/" and_form      -> or_op
?and_form: unary_form
         | and_form "/\\" unary_form  -> and_op
?unary_form: "~" unary_form           -> neg_op
           | "forall" NAME "." formula -> forall_op
           | "exists" NAME "." formula -> exists_op
           | atom_form
?atom_form: "top"                     -> top_op
          | "bot"                     -> bot_op
          | NAME "(" term ("," term)* ")"  -> pred_app
          | NAME                            -> pred_atom_bare
          | "(" formula ")"
?term: NAME "(" term ("," term)* ")"  -> func_app
     | NAME                            -> name_term

NAME: /(?!(forall|exists|top|bot)\b)[A-Za-z][A-Za-z0-9_]*/

%import common.WS
%ignore WS
COMMENT: /%[^\n]*/
%ignore COMMENT
"""


class ParseError(ValueError):
    pass


_PENDING = "__pending__"


@v_args(inline=False)
class _ToTree(Transformer):
    def iff_op(self, args):
        return iff(args[0], args[1])

    def imp_op(self, args):
        return imp(args[0], args[1])

    def or_op(self, args):
        return disj(args[0], args[1])

    def and_op(self, args):
        return conj(args[0], args[1])

    def neg_op(self, args):
        return neg(args[0])

    def forall_op(self, args):
        return forall(str(args[0]), args[1])

    def exists_op(self, args):
        return exists(str(args[0]), args[1])

    def top_op(self, args):
        return TOP

    def bot_op(self, args):
        return BOT

    def pred_app(self, args):
        pred = str(args[0])
        terms = list(args[1:])
        return atom(pred, terms)

    def pred_atom_bare(self, args):
        return atom(str(args[0]), ())

    def func_app(self, args):
        name = str(args[0])
        return Term("func", name, tuple(args[1:]))

    def name_term(self, args):
        return Term(_PENDING, str(args[0]), ())


def _resolve_term(t: Term, bound: frozenset) -> Term:
    if t.kind == _PENDING:
        if t.name in bound:
            return Term("var", t.name, ())
        return Term("const", t.name, ())
    if t.kind == "func":
        return Term("func", t.name, tuple(_resolve_term(a, bound) for a in t.args))
    return t


def _resolve_formula(f: Formula, bound: frozenset) -> Formula:
    if f.kind == "atom":
        pred, args = f.payload
        return Formula("atom", (pred, tuple(_resolve_term(a, bound) for a in args)))
    if f.kind == "neg":
        return Formula("neg", (_resolve_formula(f.payload[0], bound),))
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        return Formula(f.kind, (_resolve_formula(a, bound), _resolve_formula(b, bound)))
    if f.kind in {"forall", "exists"}:
        x, body = f.payload
        return Formula(f.kind, (x, _resolve_formula(body, bound | {x})))
    return f


_PARSER = Lark(GRAMMAR, parser="lalr", maybe_placeholders=False)
_TRANSFORMER = _ToTree()


def parse(s: str) -> Formula:
    try:
        tree = _PARSER.parse(s)
    except LarkError as e:
        raise ParseError(str(e)) from e
    ast = _TRANSFORMER.transform(tree)
    if not isinstance(ast, Formula):
        raise ParseError(f"parse did not yield a Formula: {ast!r}")
    return _resolve_formula(ast, frozenset())
