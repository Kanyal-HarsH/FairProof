import pytest

from logic.parser import parse, ParseError
from logic.pretty import pretty
from logic.syntax import (
    var, const, func, atom, neg, conj, disj, imp, iff, forall, exists, TOP, BOT,
)


CORPUS = [
    "P",
    "top",
    "bot",
    "(P /\\ Q)",
    "(P \\/ Q)",
    "(P -> Q)",
    "(P <-> Q)",
    "~(P)",
    "~~P",
    "(P /\\ (Q /\\ R))",
    "((P /\\ Q) /\\ R)",
    "(P -> (Q -> R))",
    "((P -> Q) -> ((~Q) -> (~P)))",
    "(((P -> Q) -> P) -> P)",
    "~(P /\\ (~P))",
    "((P /\\ Q) -> (Q /\\ P))",
    "((P \\/ Q) -> (Q \\/ P))",
    "(P <-> P)",
    "(top -> P)",
    "(bot -> P)",
    "P(a)",
    "P(a, b)",
    "R(f(a), b)",
    "R(f(g(a)), h(b, c))",
    "forall x. P(x)",
    "exists x. P(x)",
    "(forall x. P(x)) -> P(a)",
    "P(a) -> (exists x. P(x))",
    "forall x. forall y. R(x, y)",
    "forall x. (exists y. R(x, y))",
    "(forall x. (exists y. R(x, y))) -> (exists y. (forall x. R(x, y)))",
    "(exists x. (forall y. R(x, y))) -> (forall y. (exists x. R(x, y)))",
    "(forall x. (P(x) -> Q(x))) -> ((forall x. P(x)) -> (forall x. Q(x)))",
    "exists x. (D(x) -> forall y. D(y))",
    "exists x. ((forall y. D(y)) -> D(x))",
    "(forall x. (P(x) \\/ Q(x))) -> ((forall x. P(x)) \\/ (forall x. Q(x)))",
    "((P -> Q) /\\ (Q -> R)) -> (P -> R)",
    "((P /\\ Q) \\/ R) -> ((P \\/ R) /\\ (Q \\/ R))",
    "forall x. forall y. forall z. ((R(x, y) /\\ R(y, z)) -> R(x, z))",
    "forall x. (P(x) -> P(f(x)))",
    "(forall x. exists y. P(x, y)) -> (forall x. exists z. P(x, z))",
    "((p_1_1 \\/ p_1_2) /\\ (p_2_1 \\/ p_2_2))",
    "(P(a, b) /\\ Q(a, b)) -> (P(a, b) /\\ Q(b, a))",
    "(forall x. P(x, x)) -> (exists x. P(x, x))",
    "((forall x. P(x)) /\\ Q) -> (forall x. (P(x) /\\ Q))",
    "(forall x. (P(x) <-> Q(x))) -> ((forall x. P(x)) <-> (forall x. Q(x)))",
    "(forall x. (forall y. (R(x, y) -> R(y, x))))",
    "(exists x. P(x)) -> (~(forall x. ~P(x)))",
    "((P /\\ (Q /\\ R)) /\\ ((P /\\ Q) /\\ R))",
    "(forall x. (forall y. (forall z. ((R(x, y) /\\ R(y, z)) -> R(x, z)))))",
]


def test_corpus_parses():
    for s in CORPUS:
        parse(s)


def test_corpus_round_trip():
    for s in CORPUS:
        f = parse(s)
        f2 = parse(pretty(f))
        assert f == f2, f"round-trip failed: {s!r}"


def test_associativity_imp_right():
    f = parse("P -> Q -> R")
    expected = imp(atom("P"), imp(atom("Q"), atom("R")))
    assert f == expected


def test_associativity_or_left():
    f = parse("P \\/ Q \\/ R")
    expected = disj(disj(atom("P"), atom("Q")), atom("R"))
    assert f == expected


def test_associativity_and_left():
    f = parse("P /\\ Q /\\ R")
    expected = conj(conj(atom("P"), atom("Q")), atom("R"))
    assert f == expected


def test_negation_binds_tightest():
    f = parse("~P -> Q")
    expected = imp(neg(atom("P")), atom("Q"))
    assert f == expected


def test_quantifier_scope_to_end():
    f = parse("forall x. P(x) -> Q(x)")
    expected = forall("x", imp(atom("P", (var("x"),)), atom("Q", (var("x"),))))
    assert f == expected


def test_var_bound_inside_const_outside():
    f = parse("(forall x. P(x)) /\\ Q(x)")
    inner = forall("x", atom("P", (var("x"),)))
    outer = atom("Q", (const("x"),))
    expected = conj(inner, outer)
    assert f == expected


def test_function_application():
    f = parse("P(f(a), g(b, c))")
    expected = atom("P", (
        func("f", (const("a"),)),
        func("g", (const("b"), const("c"))),
    ))
    assert f == expected


def test_tptp_style_uppercase_variables():
    f = parse("forall X. P(X)")
    assert f == forall("X", atom("P", (var("X"),)))


def test_top_bot_formulas():
    assert parse("top") == TOP
    assert parse("bot") == BOT


def test_reject_unbalanced_parens():
    with pytest.raises(ParseError):
        parse("(P /\\ Q")


def test_reject_lone_connective():
    with pytest.raises(ParseError):
        parse("/\\ P")


def test_reject_quantifier_no_dot():
    with pytest.raises(ParseError):
        parse("forall x P(x)")


def test_reject_empty_argument_list():
    with pytest.raises(ParseError):
        parse("P()")
