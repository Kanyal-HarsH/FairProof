from logic.parser import parse
from logic.pretty import pretty, pretty_term
from logic.syntax import (
    var, const, func, atom, neg, conj, disj, imp, iff, forall, exists, TOP, BOT,
)


def test_pretty_atomic():
    assert pretty(atom("P")) == "P"
    assert pretty(TOP) == "top"
    assert pretty(BOT) == "bot"


def test_pretty_predicate_with_args():
    assert pretty(atom("P", (const("a"),))) == "P(a)"
    assert pretty(atom("R", (var("x"), const("a")))) == "R(x, a)"


def test_pretty_function_term():
    t = func("f", (var("x"), func("g", (const("a"),))))
    assert pretty_term(t) == "f(x, g(a))"


def test_pretty_binary_wraps_each_side():
    f = imp(conj(atom("P"), atom("Q")), atom("R"))
    assert pretty(f) == "((P /\\ Q) -> R)"


def test_pretty_quantifier_wraps_body():
    f = forall("x", atom("P", (var("x"),)))
    assert pretty(f) == "(forall x. P(x))"


def test_pretty_negation_wraps_argument():
    f = neg(atom("P"))
    assert pretty(f) == "~(P)"


def test_pretty_drinker():
    f = exists("x", imp(atom("D", (var("x"),)),
                         forall("y", atom("D", (var("y"),)))))
    s = pretty(f)
    assert s == "(exists x. (D(x) -> (forall y. D(y))))"
    assert parse(s) == f
