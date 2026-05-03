from logic.syntax import (
    var, const, func, atom, imp, forall, exists,
)
from logic.sequent import make_sequent
from prover.relevance import (
    relevance_score, rank_terms, term_depth, pred_overlap, term_reuse,
)


def test_term_depth():
    assert term_depth(const("a")) == 0
    assert term_depth(var("x")) == 0
    assert term_depth(func("f", (const("a"),))) == 1
    assert term_depth(func("f", (func("g", (const("a"),)),))) == 2


def test_term_reuse():
    seq = make_sequent([atom("P", (const("a"),))], [atom("Q", (const("b"),))])
    assert term_reuse(const("a"), seq) == 1
    assert term_reuse(const("b"), seq) == 1
    assert term_reuse(const("c"), seq) == 0


def test_pred_overlap_drinker_witness():
    """For drinker `forall y. (D(y) -> ...)` proven via exists_R, instantiating
    with a constant that already appears in D(c) on the antecedent should
    score higher than an unrelated constant.
    """
    a = const("a")
    b = const("b")
    body = imp(atom("D", (var("y"),)), atom("Q"))
    principal = exists("y", body)
    # Sequent has D(a) on antecedent, exists y. body on succedent.
    seq = make_sequent([atom("D", (a,))], [principal])
    score_a = pred_overlap(a, principal, seq, side="succedent")
    score_b = pred_overlap(b, principal, seq, side="succedent")
    assert score_a >= 1
    assert score_b == 0


def test_relevance_score_orders_by_overlap_then_reuse_then_depth():
    a = const("a")
    deep = func("f", (func("g", (const("c"),)),))
    body = atom("P", (var("x"),))
    principal = forall("x", body)
    seq = make_sequent([principal], [atom("P", (a,))])
    score_a = relevance_score(a, principal, seq, "antecedent")
    score_deep = relevance_score(deep, principal, seq, "antecedent")
    assert score_a > score_deep


def test_rank_terms_picks_overlap_first():
    a = const("a")
    b = const("b")
    body = atom("P", (var("x"),))
    principal = forall("x", body)
    seq = make_sequent([principal], [atom("P", (a,))])
    ordered = rank_terms([b, a], principal, seq, "antecedent")
    assert ordered[0] == a
    assert ordered[1] == b


def test_rank_terms_deterministic_tiebreak():
    a = const("alpha")
    b = const("beta")
    body = atom("Q")  # x doesn't appear in body, so no overlap difference
    principal = forall("x", body)
    seq = make_sequent([principal], [])
    ordered = rank_terms([b, a], principal, seq, "antecedent")
    assert [t.name for t in ordered] == ["alpha", "beta"]
