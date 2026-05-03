"""

FairProof. HFB+M+RG search driver. Entry point: prove_improved.

Five-class rule priority (Class 1, closing, is handled by is_axiom):
    Class 2  deterministic propositional rules and unary movers
    Class 3  eigenvariable quantifier rules (forall R, exists L)
    Class 4  branching propositional rules
    Class 5  retaining quantifier rules (forall L, exists R)

_search tries the classes in order on the current sequent and stops at
the first class that produces a proof. Each _try_class_X returns one of:
    OK    a proof was built
    FAIL  the class fired but the subtree could not be closed
    NONE  no rule of this class is applicable; try the next class

HFB:
    Iterative deepening over the per-iteration node budget. Budget grows
    by budget_growth per iteration. Each iteration restarts from the
    root and reuses memo hits from earlier iterations. The loop stops
    when a proof is found, when max_total_nodes is hit, or when
    timeout_s expires.

M:
    fingerprint(seq, history) is the cache key. A CLOSED hit returns
    the cached proof immediately. An IN_PROGRESS hit on the current
    ancestor path is a cycle and is abandoned. Closed proofs are stored
    on success; failed expansions store nothing (failure under one
    budget does not imply failure under a larger budget).

R:
    For Class 5, candidate terms are ranked by rank_terms with weights
    k1, k2, k3 from the config. The top rg_top_k are tried first; the
    rest go through round-robin under FairnessCounter so no candidate
    is permanently starved.

Soundness: every applied rule is in prover/rules.py. The cache only
forwards CLOSED entries from successful expansions, and cycle detection
on IN_PROGRESS prunes duplicates without turning an unproved goal into
a proved one.

"""

import time
from dataclasses import dataclass, field
from typing import Optional

from logic.syntax import Term, Formula, conj, imp
from logic.sequent import Sequent, make_sequent, is_axiom
from prover.rules import (
    RuleContext, RuleError,
    apply_neg_L, apply_neg_R,
    apply_and_L, apply_and_R, apply_or_L, apply_or_R,
    apply_imp_L, apply_imp_R,
    apply_forall_L, apply_forall_R, apply_exists_L, apply_exists_R,
)
from prover.baseline import ProofNode, unfold_iff, enumerate_terms
from prover.config import ImprovedConfig
from prover.memoisation import MemoisationCache, fingerprint, cache_lookup, CLOSED
from prover.relevance import rank_terms


# Class 1 (closing rules) is handled by is_axiom. Classes 2..5 below.

CLASS_2_DET = (
    ("and_L", "antecedent", "and"),
    ("or_R", "succedent", "or"),
    ("imp_R", "succedent", "imp"),
    ("neg_L", "antecedent", "neg"),
    ("neg_R", "succedent", "neg"),
)

CLASS_3_EIGEN = (
    ("forall_R", "succedent", "forall"),
    ("exists_L", "antecedent", "exists"),
)

CLASS_4_BRANCH = (
    ("and_R", "succedent", "and"),
    ("or_L", "antecedent", "or"),
    ("imp_L", "antecedent", "imp"),
)

CLASS_5_RETAIN = (
    ("forall_L", "antecedent", "forall"),
    ("exists_R", "succedent", "exists"),
)


def _formula_key(f) -> str:
    if f.kind == "atom":
        pred, args = f.payload
        return f"atom:{pred}:{len(args)}"
    if f.kind == "neg":
        return f"neg:{_formula_key(f.payload[0])}"
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        return f"{f.kind}:{_formula_key(a)}|{_formula_key(b)}"
    if f.kind in {"forall", "exists"}:
        x, body = f.payload
        return f"{f.kind}:{x}.{_formula_key(body)}"
    return f.kind


def _formula_size(f) -> int:
    if f.kind == "atom":
        return 1
    if f.kind == "neg":
        return 1 + _formula_size(f.payload[0])
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        return 1 + _formula_size(a) + _formula_size(b)
    if f.kind in {"forall", "exists"}:
        return 1 + _formula_size(f.payload[1])
    return 1


def _principal_sort_key(f):
    return (_formula_size(f), _formula_key(f))


def _bag(seq, side):
    return seq.antecedent if side == "antecedent" else seq.succedent


def find_principal(seq, side, kind):
    candidates = [f for f in _bag(seq, side) if f.kind == kind]
    if not candidates:
        return None
    candidates.sort(key=_principal_sort_key)
    return candidates[0]


def all_principals(seq, side, kind):
    candidates = [f for f in _bag(seq, side) if f.kind == kind]
    candidates.sort(key=_principal_sort_key)
    return candidates


# Every K-th retaining choice falls back to round-robin so no term starves.

@dataclass
class FairnessCounter:
    period: int = 32
    step: int = 0

    def tick(self) -> bool:

        """Advance the counter. Returns True on every period-th call."""

        self.step += 1
        return self.step % self.period == 0


@dataclass
class ImprovedStats:
    nodes_expanded: int = 0
    depth_max_seen: int = 0
    fresh_consts_used: int = 0
    distinct_instantiations: int = 0
    iterations: int = 0
    iteration_at_solution: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_stores: int = 0
    cache_size: int = 0
    eigenvar_count: int = 0
    max_term_depth: int = 0
    axiom_closures: int = 0
    branching_applications: int = 0
    aborted_node_budget: bool = False
    aborted_total_budget: bool = False
    aborted_timeout: bool = False


_APPLY_SIMPLE = {
    "and_L": apply_and_L, "and_R": apply_and_R,
    "or_L": apply_or_L, "or_R": apply_or_R,
    "imp_L": apply_imp_L, "imp_R": apply_imp_R,
    "neg_L": apply_neg_L, "neg_R": apply_neg_R,
    "forall_R": apply_forall_R, "exists_L": apply_exists_L,
}


def _apply_simple(rule_name: str, seq, principal, ctx):
    return _APPLY_SIMPLE[rule_name](seq, principal, ctx)


def _apply_inst(rule_name: str, seq, principal, ctx, term):
    if rule_name == "forall_L":
        return apply_forall_L(seq, principal, ctx, term)[0]
    if rule_name == "exists_R":
        return apply_exists_R(seq, principal, ctx, term)[0]
    raise ValueError(rule_name)


def _try_class_2(seq, depth, ctx, search_args):

    """First applicable deterministic propositional or unary mover rule."""

    for rule_name, side, kind in CLASS_2_DET:
        principal = find_principal(seq, side, kind)
        if principal is None:
            continue
        try:
            premises = _apply_simple(rule_name, seq, principal, ctx)
        except RuleError:
            continue
        children = []
        for p in premises:
            child = _search(p, depth + 1, **search_args)
            if child is None:
                return ("FAIL", None)
            children.append(child)
        return ("OK", ProofNode(rule=rule_name, principal=principal,
                                sequent=seq, premises=tuple(children)))
    return ("NONE", None)


def _try_class_3(seq, depth, ctx, search_args):

    """First applicable eigenvariable quantifier rule (forall R or exists L)."""

    for rule_name, side, kind in CLASS_3_EIGEN:
        principal = find_principal(seq, side, kind)
        if principal is None:
            continue
        try:
            premises = _apply_simple(rule_name, seq, principal, ctx)
        except RuleError:
            continue
        children = []
        for p in premises:
            child = _search(p, depth + 1, **search_args)
            if child is None:
                return ("FAIL", None)
            children.append(child)
        return ("OK", ProofNode(rule=rule_name, principal=principal,
                                sequent=seq, premises=tuple(children)))
    return ("NONE", None)


def _try_class_4(seq, depth, ctx, search_args):

    """First applicable branching propositional rule (and R, or L, imp L)."""

    for rule_name, side, kind in CLASS_4_BRANCH:
        principal = find_principal(seq, side, kind)
        if principal is None:
            continue
        try:
            premises = _apply_simple(rule_name, seq, principal, ctx)
        except RuleError:
            continue
        ctx.branching_applications += 1
        children = []
        ok = True
        for p in premises:
            child = _search(p, depth + 1, **search_args)
            if child is None:
                ok = False
                break
            children.append(child)
        if ok:
            return ("OK", ProofNode(rule=rule_name, principal=principal,
                                    sequent=seq, premises=tuple(children)))
        return ("FAIL", None)
    return ("NONE", None)


def _aborted(stats) -> bool:
    return (stats.aborted_node_budget or stats.aborted_total_budget
            or stats.aborted_timeout)


def _try_class_5(seq, depth, ctx, search_args):

    """Retaining quantifier rules. RG-rank existing terms; fall back to fresh."""

    config = search_args["config"]
    history = search_args["history"]
    stats = search_args["stats"]
    fairness = search_args["fairness"]
    for rule_name, side, kind in CLASS_5_RETAIN:
        for principal in all_principals(seq, side, kind):
            if _aborted(stats):
                return ("FAIL", None)
            used = history.get(principal, frozenset())
            available = list(enumerate_terms(seq) - used)

            rg_active = (config.use_relevance
                         and len(available) >= config.rg_min_candidates)
            if rg_active:
                ordered = rank_terms(
                    available, principal, seq, side,
                    config.rg_k1, config.rg_k2, config.rg_k3,
                )
            else:
                ordered = sorted(available, key=lambda c: c.name)

            top = ordered[: config.rg_top_k] if rg_active else ordered
            tail = ordered[config.rg_top_k:] if rg_active else []

            tries = list(top)
            if rg_active and tail and fairness.tick():
                tries = list(tail) + list(top)
            elif tail:
                tries.extend(tail)

            for t in tries:
                if _aborted(stats):
                    return ("FAIL", None)
                stats.distinct_instantiations += 1
                new_history = dict(history)
                new_history[principal] = used | {t}
                try:
                    premise = _apply_inst(rule_name, seq, principal, ctx, t)
                except RuleError:
                    continue
                child = _search(
                    premise, depth + 1,
                    **{**search_args, "history": new_history},
                )
                if child is not None:
                    return ("OK", ProofNode(
                        rule=rule_name, principal=principal,
                        sequent=seq, premises=(child,),
                    ))

            if _aborted(stats):
                return ("FAIL", None)
            fresh_t = ctx.fresh_const("c")
            stats.fresh_consts_used += 1
            stats.distinct_instantiations += 1
            new_history = dict(history)
            new_history[principal] = used | {fresh_t}
            try:
                premise = _apply_inst(rule_name, seq, principal, ctx, fresh_t)
            except RuleError:
                continue
            child = _search(
                premise, depth + 1,
                **{**search_args, "history": new_history},
            )
            if child is not None:
                return ("OK", ProofNode(
                    rule=rule_name, principal=principal,
                    sequent=seq, premises=(child,),
                ))
    return ("NONE", None)


def _search(seq: Sequent, depth: int, *, config: ImprovedConfig,
            ctx: RuleContext, history: dict, stats: ImprovedStats,
            cache: Optional[MemoisationCache], ancestor_path: frozenset,
            fairness: FairnessCounter, node_budget: int,
            deadline: Optional[float] = None) -> Optional[ProofNode]:

    """One node of the search. Honours budgets, consults the cache,
    then tries Class 2 through Class 5 in order. Returns the proof tree
    or None.
    """

    if stats.aborted_node_budget or stats.aborted_total_budget or stats.aborted_timeout:
        return None
    if stats.nodes_expanded >= node_budget:
        stats.aborted_node_budget = True
        return None
    if stats.nodes_expanded >= config.max_total_nodes:
        stats.aborted_total_budget = True
        return None
    if deadline is not None and time.perf_counter() >= deadline:
        stats.aborted_timeout = True
        return None
    stats.nodes_expanded += 1
    if depth > stats.depth_max_seen:
        stats.depth_max_seen = depth
    if depth > config.max_depth:
        return None

    if is_axiom(seq):
        ctx.axiom_closures += 1
        return ProofNode(rule="axiom", principal=None, sequent=seq, premises=())

    fp = fingerprint(seq, history)
    status, cached = cache_lookup(cache, fp, ancestor_path)
    if status == "CLOSED_HIT":
        return cached
    if status == "CYCLE":
        return None

    if cache is not None:
        cache.mark_in_progress(fp)
    next_ancestor = ancestor_path | {fp}

    search_args = {
        "config": config, "ctx": ctx, "history": history, "stats": stats,
        "cache": cache, "ancestor_path": next_ancestor, "fairness": fairness,
        "node_budget": node_budget, "deadline": deadline,
    }

    proof: Optional[ProofNode] = None
    for klass in (_try_class_2, _try_class_3, _try_class_4, _try_class_5):
        outcome, candidate = klass(seq, depth, ctx, search_args)
        if outcome == "OK":
            proof = candidate
            break
        if outcome == "FAIL":
            proof = None
            break

    if cache is not None:
        cache.unmark_in_progress(fp)
        if proof is not None:
            cache.store_closed(fp, proof)

    return proof


def _finalise_stats(stats: ImprovedStats, ctx: RuleContext, cache) -> None:
    stats.eigenvar_count = ctx.eigenvar_count
    stats.max_term_depth = ctx.max_term_depth
    stats.axiom_closures = ctx.axiom_closures
    stats.branching_applications = ctx.branching_applications
    if cache is not None:
        stats.cache_hits = cache.hits
        stats.cache_misses = cache.misses
        stats.cache_stores = cache.stores
        stats.cache_size = cache.closed_size()


def prove_improved(formula: Formula, config: Optional[ImprovedConfig] = None,
              timeout_s: Optional[float] = None) -> tuple:

    """
    Returns (proof, ImprovedStats). proof is None on failure.
    config selects which of memo, relevance, iterative deepening are active.
    """
    if config is None:
        config = ImprovedConfig()

    f = unfold_iff(formula)
    seq = make_sequent([], [f])
    ctx = RuleContext()
    stats = ImprovedStats()
    cache = MemoisationCache(config.memo_max_entries) if config.use_memo else None
    fairness = FairnessCounter(period=config.fairness_period)
    deadline = time.perf_counter() + timeout_s if timeout_s is not None else None

    if not config.iterative_deepening:
        proof = _search(
            seq, 0,
            config=config, ctx=ctx, history={}, stats=stats,
            cache=cache, ancestor_path=frozenset(),
            fairness=fairness, node_budget=config.max_total_nodes,
            deadline=deadline,
        )
        stats.iterations = 1
        if proof is not None:
            stats.iteration_at_solution = 1
        _finalise_stats(stats, ctx, cache)
        return proof, stats

    budget = config.base_node_budget
    proof = None
    for i in range(1, config.max_iterations + 1):
        stats.iterations = i
        if stats.nodes_expanded >= config.max_total_nodes:
            stats.aborted_total_budget = True
            break
        if stats.aborted_timeout:
            break
        per_iteration_budget = stats.nodes_expanded + budget
        if per_iteration_budget > config.max_total_nodes:
            per_iteration_budget = config.max_total_nodes
        proof = _search(
            seq, 0,
            config=config, ctx=ctx, history={}, stats=stats,
            cache=cache, ancestor_path=frozenset(),
            fairness=fairness, node_budget=per_iteration_budget,
            deadline=deadline,
        )
        if proof is not None:
            stats.iteration_at_solution = i
            _finalise_stats(stats, ctx, cache)
            return proof, stats
        if stats.aborted_total_budget or stats.aborted_timeout:
            break
        stats.aborted_node_budget = False
        budget = int(budget * config.budget_growth)

    _finalise_stats(stats, ctx, cache)
    return None, stats
