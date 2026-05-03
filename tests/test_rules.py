import pytest

from logic.syntax import (
    var, const, atom, neg, conj, disj, imp, iff, forall, exists, TOP, BOT,
)
from logic.sequent import make_sequent, is_axiom, add_left, add_right
from prover.rules import (
    RuleContext, RuleError,
    apply_id, apply_top_R, apply_bot_L,
    apply_neg_L, apply_neg_R,
    apply_and_L, apply_and_R, apply_or_L, apply_or_R, apply_imp_L, apply_imp_R,
    apply_forall_L, apply_forall_R, apply_exists_L, apply_exists_R,
)


P = atom("P")
Q = atom("Q")
R = atom("R")


def ctx():
    return RuleContext()


def test_id_closes():
    s = make_sequent([P], [P])
    assert is_axiom(s)
    assert apply_id(s, P, ctx()) == ()


def test_top_R_closes():
    s = make_sequent([], [TOP])
    assert apply_top_R(s, TOP, ctx()) == ()


def test_bot_L_closes():
    s = make_sequent([BOT], [])
    assert apply_bot_L(s, BOT, ctx()) == ()


def test_neg_L_swaps_to_right():
    f = neg(P)
    s = make_sequent([f], [Q])
    (premise,) = apply_neg_L(s, f, ctx())
    assert premise == make_sequent([], [Q, P])


def test_neg_R_swaps_to_left():
    f = neg(P)
    s = make_sequent([Q], [f])
    (premise,) = apply_neg_R(s, f, ctx())
    assert premise == make_sequent([Q, P], [])


def test_and_L_splits_conjuncts():
    f = conj(P, Q)
    s = make_sequent([f], [R])
    (premise,) = apply_and_L(s, f, ctx())
    assert premise == make_sequent([P, Q], [R])


def test_and_R_branches():
    f = conj(P, Q)
    s = make_sequent([R], [f])
    a, b = apply_and_R(s, f, ctx())
    assert a == make_sequent([R], [P])
    assert b == make_sequent([R], [Q])


def test_or_L_branches():
    f = disj(P, Q)
    s = make_sequent([f], [R])
    a, b = apply_or_L(s, f, ctx())
    assert a == make_sequent([P], [R])
    assert b == make_sequent([Q], [R])


def test_or_R_collects_disjuncts():
    f = disj(P, Q)
    s = make_sequent([R], [f])
    (premise,) = apply_or_R(s, f, ctx())
    assert premise == make_sequent([R], [P, Q])


def test_imp_L_branches():
    f = imp(P, Q)
    s = make_sequent([f], [R])
    a, b = apply_imp_L(s, f, ctx())
    assert a == make_sequent([], [R, P])
    assert b == make_sequent([Q], [R])


def test_imp_R_moves_antecedent_left():
    f = imp(P, Q)
    s = make_sequent([R], [f])
    (premise,) = apply_imp_R(s, f, ctx())
    assert premise == make_sequent([R, P], [Q])


def test_forall_L_retains_principal_and_instantiates():
    f = forall("x", atom("P", (var("x"),)))
    s = make_sequent([f], [atom("Q")])
    (premise,) = apply_forall_L(s, f, ctx(), const("a"))
    assert f in premise.antecedent
    assert atom("P", (const("a"),)) in premise.antecedent


def test_exists_R_retains_principal_and_instantiates():
    f = exists("x", atom("P", (var("x"),)))
    s = make_sequent([atom("Q")], [f])
    (premise,) = apply_exists_R(s, f, ctx(), const("a"))
    assert f in premise.succedent
    assert atom("P", (const("a"),)) in premise.succedent


def test_forall_R_uses_fresh_eigenvariable():
    f = forall("x", atom("P", (var("x"),)))
    s = make_sequent([], [f])
    c = ctx()
    (premise,) = apply_forall_R(s, f, c)
    assert f not in premise.succedent
    instantiated = next(iter(premise.succedent))
    pred, args = instantiated.payload
    assert pred == "P"
    assert len(args) == 1
    fresh_const = args[0]
    assert fresh_const.kind == "const"
    assert fresh_const.name not in {"x"}


def test_forall_R_freshness_against_existing_constant():
    f = forall("x", atom("P", (var("x"),)))
    s = make_sequent([atom("Q", (const("a_1"),))], [f])
    c = RuleContext()
    (premise,) = apply_forall_R(s, f, c)
    instantiated = next(iter(premise.succedent))
    _, args = instantiated.payload
    assert args[0].name != "a_1"


def test_exists_L_uses_fresh_eigenvariable():
    f = exists("x", atom("P", (var("x"),)))
    s = make_sequent([f], [])
    c = ctx()
    (premise,) = apply_exists_L(s, f, c)
    assert f not in premise.antecedent


def test_rule_rejects_wrong_principal():
    s = make_sequent([conj(P, Q)], [R])
    with pytest.raises(RuleError):
        apply_or_L(s, conj(P, Q), ctx())


def test_hand_derive_contrapositive():
    """Prove (P -> Q) -> (~Q -> ~P) by hand-driving the rule engine."""
    goal = imp(imp(P, Q), imp(neg(Q), neg(P)))
    s0 = make_sequent([], [goal])
    c = ctx()

    (s1,) = apply_imp_R(s0, goal, c)
    inner = imp(neg(Q), neg(P))
    (s2,) = apply_imp_R(s1, inner, c)
    nq = neg(Q)
    (s3,) = apply_neg_L(s2, nq, c)
    np_ = neg(P)
    (s4,) = apply_neg_R(s3, np_, c)
    pq = imp(P, Q)
    a, b = apply_imp_L(s4, pq, c)
    assert is_axiom(a)
    assert is_axiom(b)
