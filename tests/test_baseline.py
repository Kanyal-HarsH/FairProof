import pytest

from logic.parser import parse
from logic.syntax import (
    var, const, atom, neg, conj, disj, imp, iff, forall, exists, TOP, BOT,
)
from prover.baseline import prove, is_proved, proof_depth


# Sanity: propositional theorems that the baseline must handle.
PROVABLE = [
    ("contrapositive", "(P -> Q) -> ((~Q) -> (~P))"),
    ("identity", "P -> P"),
    ("k_combinator", "P -> (Q -> P)"),
    ("conj_proj_left", "(P /\\ Q) -> P"),
    ("conj_proj_right", "(P /\\ Q) -> Q"),
    ("disj_intro_left", "P -> (P \\/ Q)"),
    ("disj_intro_right", "Q -> (P \\/ Q)"),
    ("modus_ponens_curried", "(P -> Q) -> (P -> Q)"),
    ("excluded_middle", "P \\/ (~P)"),
    ("non_contradiction", "~(P /\\ (~P))"),
    ("peirce_law", "((P -> Q) -> P) -> P"),
    ("hypo_syllogism", "((P -> Q) /\\ (Q -> R)) -> (P -> R)"),
    ("commute_and", "(P /\\ Q) -> (Q /\\ P)"),
    ("commute_or", "(P \\/ Q) -> (Q \\/ P)"),
    ("pelletier_1", "(P -> Q) <-> ((~Q) -> (~P))"),
    ("pelletier_2", "(~(~P)) <-> P"),
    ("pelletier_3", "(~(P -> Q)) -> (Q -> P)"),
    ("pelletier_4", "((~P) -> Q) <-> ((~Q) -> P)"),
    ("pelletier_5", "((P \\/ Q) -> (P \\/ R)) -> (P \\/ (Q -> R))"),
]

# First-order theorems including the drinker variants and quantifier swaps.
PROVABLE_FOL = [
    ("forall_inst", "(forall x. P(x)) -> P(a)"),
    ("exists_intro", "P(a) -> (exists x. P(x))"),
    ("forall_distrib_imp",
     "(forall x. (P(x) -> Q(x))) -> ((forall x. P(x)) -> (forall x. Q(x)))"),
    ("forall_swap",
     "(forall x. forall y. R(x, y)) -> (forall y. forall x. R(x, y))"),
    ("exists_under_imp",
     "(exists x. (forall y. R(x, y))) -> (forall y. (exists x. R(x, y)))"),
    ("drinker_canonical",
     "exists x. (D(x) -> forall y. D(y))"),
    ("drinker_dual",
     "exists x. ((forall y. D(y)) -> D(x))"),
]

# Non-theorems: the prover should NOT find a proof within the budget.
NON_THEOREMS = [
    ("forall_disj_split",
     "(forall x. (P(x) \\/ Q(x))) -> ((forall x. P(x)) \\/ (forall x. Q(x)))"),
    ("quantifier_swap_invalid",
     "(forall x. exists y. R(x, y)) -> (exists y. forall x. R(x, y))"),
    ("affirming_consequent",
     "((forall x. (P(x) -> Q(x))) /\\ Q(a)) -> P(a)"),
    ("converse",
     "(P -> Q) -> (Q -> P)"),
]


@pytest.mark.parametrize("name,src", PROVABLE)
def test_propositional_theorems(name, src):
    f = parse(src)
    proof, _ = prove(f, max_depth=50)
    assert proof is not None, f"{name} should be provable"


@pytest.mark.parametrize("name,src", PROVABLE_FOL)
def test_first_order_theorems(name, src):
    # node_budget=200_000 matches the experimental harness in
    # experiments/_metrics.py. Algorithm 2 with a lexicographic
    # principal tiebreak takes ~100k nodes on forall_swap, which
    # exceeds the 50k default.
    f = parse(src)
    proof, _ = prove(f, max_depth=50, node_budget=200_000)
    assert proof is not None, f"{name} should be provable"


@pytest.mark.parametrize("name,src", NON_THEOREMS)
def test_non_theorems_fail(name, src):
    f = parse(src)
    proof, _ = prove(f, max_depth=10, node_budget=5000)
    assert proof is None, f"{name} should not be provable within budget"


def test_proof_tree_leaves_are_axioms():
    f = parse("(P /\\ Q) -> (Q /\\ P)")
    proof, _ = prove(f)
    assert proof is not None

    def collect_leaves(n):
        if not n.premises:
            yield n
        else:
            for c in n.premises:
                yield from collect_leaves(c)

    from logic.sequent import is_axiom
    for leaf in collect_leaves(proof):
        assert leaf.rule == "axiom"
        assert is_axiom(leaf.sequent)


def test_proof_depth_bounded():
    f = parse("P -> P")
    proof, _ = prove(f)
    assert proof is not None
    assert proof_depth(proof) <= 5


def test_iff_unfold_preserves_provability():
    f = parse("P <-> P")
    proof, _ = prove(f)
    assert proof is not None
