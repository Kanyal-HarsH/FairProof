from logic.syntax import (
    var, const, func, atom, neg, conj, disj, imp, forall, exists, free_vars, term_vars,
)
from logic.subst import subst_term, subst_formula, alpha_rename


def test_subst_term_replaces_variable():
    t = func("f", (var("x"), const("a"), var("x")))
    out = subst_term(t, "x", const("b"))
    assert out == func("f", (const("b"), const("a"), const("b")))


def test_subst_term_skips_constants():
    assert subst_term(const("a"), "a", var("x")) == const("a")


def test_subst_formula_replaces_free_var():
    f = atom("P", (var("x"),))
    out = subst_formula(f, "x", const("a"))
    assert out == atom("P", (const("a"),))


def test_subst_formula_bound_shadows():
    f = forall("x", atom("P", (var("x"),)))
    out = subst_formula(f, "x", const("a"))
    assert out == f


def test_subst_formula_alpha_rename_avoids_capture():
    f = forall("y", atom("P", (var("x"),)))
    out = subst_formula(f, "x", var("y"))
    quant_kind, payload = out.kind, out.payload
    assert quant_kind == "forall"
    new_bound, body = payload
    assert new_bound != "y"
    assert body == atom("P", (var("y"),))


def test_subst_no_op_when_var_not_free():
    f = atom("P", (const("a"),))
    out = subst_formula(f, "x", const("b"))
    assert out == f


def test_subst_distributes_through_connectives():
    f = imp(atom("P", (var("x"),)), atom("Q", (var("x"),)))
    out = subst_formula(f, "x", const("a"))
    assert out == imp(atom("P", (const("a"),)), atom("Q", (const("a"),)))


def test_alpha_rename_inside_atom():
    f = atom("P", (var("x"), var("y"), var("x")))
    out = alpha_rename(f, "x", "z")
    assert out == atom("P", (var("z"), var("y"), var("z")))


def test_alpha_rename_skips_bound():
    f = forall("x", atom("P", (var("x"),)))
    out = alpha_rename(f, "x", "z")
    assert out == f


def test_free_vars_after_subst():
    f = imp(atom("P", (var("x"),)), atom("Q", (var("y"),)))
    out = subst_formula(f, "x", func("h", (var("z"),)))
    assert free_vars(out) == frozenset({"y", "z"})
