from logic.syntax import (
    Term, Formula,
    var, const, func, atom, neg, conj, disj, imp, iff, forall, exists,
    TOP, BOT, free_vars, term_vars, is_atom,
)


def test_helpers_match_raw_construction():
    assert var("x") == Term("var", "x", ())
    assert const("a") == Term("const", "a", ())
    assert func("f", (var("x"),)) == Term("func", "f", (Term("var", "x", ()),))
    assert atom("P", ()) == Formula("atom", ("P", ()))
    assert conj(atom("P"), atom("Q")) == Formula("and", (atom("P"), atom("Q")))


def test_constants_are_singletons():
    assert TOP == Formula("top", ())
    assert BOT == Formula("bot", ())


def test_term_vars():
    assert term_vars(var("x")) == frozenset({"x"})
    assert term_vars(const("a")) == frozenset()
    assert term_vars(func("f", (var("x"), const("a"), var("y")))) == frozenset({"x", "y"})


def test_free_vars_propositional():
    assert free_vars(atom("P", ())) == frozenset()
    assert free_vars(conj(atom("P"), atom("Q"))) == frozenset()


def test_free_vars_quantifier_binds():
    f = forall("x", atom("P", (var("x"),)))
    assert free_vars(f) == frozenset()


def test_free_vars_under_outer_var():
    f = forall("x", atom("R", (var("x"), var("y"))))
    assert free_vars(f) == frozenset({"y"})


def test_hash_equality():
    a = forall("x", atom("P", (var("x"),)))
    b = forall("x", atom("P", (var("x"),)))
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_is_atom():
    assert is_atom(atom("P"))
    assert not is_atom(neg(atom("P")))
