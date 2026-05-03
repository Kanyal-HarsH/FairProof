import pytest

from logic.parser import parse
from prover.improved_algorithm import prove_improved
from prover.config import ImprovedConfig, config_for


PROVABLE = [
    ("contrapositive", "(P -> Q) -> ((~Q) -> (~P))"),
    ("identity", "P -> P"),
    ("k_combinator", "P -> (Q -> P)"),
    ("conj_proj_left", "(P /\\ Q) -> P"),
    ("disj_intro_left", "P -> (P \\/ Q)"),
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
    ("forall_inst", "(forall x. P(x)) -> P(a)"),
    ("exists_intro", "P(a) -> (exists x. P(x))"),
    ("forall_distrib_imp",
     "(forall x. (P(x) -> Q(x))) -> ((forall x. P(x)) -> (forall x. Q(x)))"),
    ("forall_swap",
     "(forall x. forall y. R(x, y)) -> (forall y. forall x. R(x, y))"),
    ("drinker_canonical", "exists x. (D(x) -> forall y. D(y))"),
    ("drinker_dual", "exists x. ((forall y. D(y)) -> D(x))"),
]

NON_THEOREMS = [
    ("forall_disj_split",
     "(forall x. (P(x) \\/ Q(x))) -> ((forall x. P(x)) \\/ (forall x. Q(x)))"),
    ("quantifier_swap_invalid",
     "(forall x. exists y. R(x, y)) -> (exists y. forall x. R(x, y))"),
    ("converse", "(P -> Q) -> (Q -> P)"),
]


@pytest.mark.parametrize("name,src", PROVABLE)
def test_improved_c4_proves(name, src):
    f = parse(src)
    cfg = config_for("C4")
    cfg.max_total_nodes = 30_000
    proof, _ = prove_improved(f, cfg)
    assert proof is not None, f"{name} should be provable under C4"


@pytest.mark.parametrize("cfg_name", ["C1", "C2", "C3", "C4"])
def test_each_ablation_proves_drinker(cfg_name):
    f = parse("exists x. (D(x) -> forall y. D(y))")
    cfg = config_for(cfg_name)
    cfg.max_total_nodes = 10_000
    proof, _ = prove_improved(f, cfg)
    assert proof is not None, f"{cfg_name} should prove drinker"


@pytest.mark.parametrize("name,src", NON_THEOREMS)
def test_improved_fails_on_non_theorems(name, src):
    f = parse(src)
    cfg = config_for("C4")
    cfg.max_total_nodes = 5_000
    cfg.max_depth = 12
    proof, _ = prove_improved(f, cfg)
    assert proof is None, f"{name} should not be provable under budget"


def test_memo_records_hits_on_repeated_subproblem():
    f = parse("(P -> P) /\\ (P -> P)")
    cfg = config_for("C2")
    cfg.max_total_nodes = 10_000
    proof, stats = prove_improved(f, cfg)
    assert proof is not None
    assert stats.cache_hits >= 1


def test_disabling_memo_means_no_hits():
    f = parse("(P -> P) /\\ (P -> P)")
    cfg = config_for("C1")
    proof, stats = prove_improved(f, cfg)
    assert proof is not None
    assert stats.cache_hits == 0


def test_iteration_at_solution_recorded():
    f = parse("(forall x. (P(x) -> Q(x))) -> ((forall x. P(x)) -> (forall x. Q(x)))")
    cfg = config_for("C4")
    cfg.max_total_nodes = 30_000
    proof, stats = prove_improved(f, cfg)
    assert proof is not None
    assert stats.iteration_at_solution >= 1
