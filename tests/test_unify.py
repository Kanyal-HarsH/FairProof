from logic.syntax import var, const, func, atom
from logic.unify import (
    unify_terms, unify_atoms, apply_subst_term, apply_subst_formula,
)


def test_unify_two_consts_equal():
    sigma = unify_terms(const("a"), const("a"))
    assert sigma == {}


def test_unify_two_consts_unequal():
    assert unify_terms(const("a"), const("b")) is None


def test_unify_var_with_const():
    sigma = unify_terms(var("x"), const("a"))
    assert sigma == {"x": const("a")}


def test_unify_function_terms():
    t1 = func("f", (var("x"), var("y")))
    t2 = func("f", (const("a"), func("g", (var("x"),))))
    sigma = unify_terms(t1, t2)
    assert sigma["x"] == const("a")
    assert sigma["y"] == func("g", (var("x"),))


def test_unify_occurs_check():
    t1 = func("f", (var("x"),))
    t2 = var("x")
    assert unify_terms(t1, t2) is None
    assert unify_terms(t2, t1) is None


def test_unify_occurs_via_indirection():
    t1 = atom("P", (var("x"), func("f", (var("x"),))))
    t2 = atom("P", (func("g", (var("y"),)), var("y")))
    assert unify_atoms(t1, t2) is None


def test_unify_const_with_const_pair():
    a = atom("P", (var("x"), var("x")))
    b = atom("P", (const("a"), const("b")))
    assert unify_atoms(a, b) is None


def test_unify_predicate_arity_mismatch():
    a = atom("P", (var("x"),))
    b = atom("P", (var("x"), var("y")))
    assert unify_atoms(a, b) is None


def test_apply_subst_idempotent():
    sigma = {"x": const("a"), "y": func("f", (var("x"),))}
    t = func("g", (var("x"), var("y")))
    out1 = apply_subst_term(sigma, t)
    out2 = apply_subst_term(sigma, out1)
    assert out1 == out2
    assert out1 == func("g", (const("a"), func("f", (const("a"),))))


def test_apply_subst_formula():
    sigma = {"x": const("a")}
    f = atom("P", (var("x"), var("y")))
    out = apply_subst_formula(sigma, f)
    assert out == atom("P", (const("a"), var("y")))
