""" Baseline Algorithm 2 from Prof. Hou """

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


@dataclass
class ProofNode:
    rule: str
    principal: object
    sequent: Sequent
    premises: tuple = ()


def unfold_iff(f: Formula) -> Formula:

    """Rewrite iff(A, B) as conj(imp(A, B), imp(B, A))."""

    if f.kind == "iff":
        a, b = f.payload
        a2 = unfold_iff(a)
        b2 = unfold_iff(b)
        return conj(imp(a2, b2), imp(b2, a2))
    if f.kind == "neg":
        return Formula("neg", (unfold_iff(f.payload[0]),))
    if f.kind in {"and", "or", "imp"}:
        a, b = f.payload
        return Formula(f.kind, (unfold_iff(a), unfold_iff(b)))
    if f.kind in {"forall", "exists"}:
        x, body = f.payload
        return Formula(f.kind, (x, unfold_iff(body)))
    return f


def _unfold_iff_seq(seq: Sequent) -> Sequent:
    return Sequent(
        frozenset(unfold_iff(f) for f in seq.antecedent),
        frozenset(unfold_iff(f) for f in seq.succedent),
    )


DETERMINISTIC_RULES = (
    ("and_L", "antecedent"),
    ("or_R", "succedent"),
    ("imp_R", "succedent"),
    ("neg_L", "antecedent"),
    ("neg_R", "succedent"),
    ("forall_R", "succedent"),
    ("exists_L", "antecedent"),
)

BRANCHING_RULES = (
    ("and_R", "succedent"),
    ("or_L", "antecedent"),
    ("imp_L", "antecedent"),
)

RETAINING_RULES = (
    ("forall_L", "antecedent"),
    ("exists_R", "succedent"),
)


_KIND_FOR = {
    "and_L": "and", "and_R": "and",
    "or_L": "or", "or_R": "or",
    "imp_L": "imp", "imp_R": "imp",
    "neg_L": "neg", "neg_R": "neg",
    "forall_L": "forall", "forall_R": "forall",
    "exists_L": "exists", "exists_R": "exists",
}


def _formula_key(f: Formula) -> str:
    if f.kind == "atom":
        pred, args = f.payload
        return f"atom:{pred}:{len(args)}:" + ",".join(_term_key(a) for a in args)
    if f.kind == "neg":
        return "neg:" + _formula_key(f.payload[0])
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        return f"{f.kind}:" + _formula_key(a) + "|" + _formula_key(b)
    if f.kind in {"forall", "exists"}:
        x, body = f.payload
        return f"{f.kind}:{x}." + _formula_key(body)
    return f.kind


def _term_key(t: Term) -> str:
    if t.kind == "func":
        return f"f:{t.name}({','.join(_term_key(a) for a in t.args)})"
    return f"{t.kind}:{t.name}"


def _find_principal(seq: Sequent, rule_name: str, side: str):
    # lex tiebreak; frozenset iteration is hash-randomised across processes
    target = _KIND_FOR[rule_name]
    bag = seq.antecedent if side == "antecedent" else seq.succedent
    matches = [f for f in bag if f.kind == target]
    if not matches:
        return None
    return min(matches, key=_formula_key)


def _all_principals(seq: Sequent, rule_name: str, side: str):
    target = _KIND_FOR[rule_name]
    bag = seq.antecedent if side == "antecedent" else seq.succedent
    return sorted((f for f in bag if f.kind == target), key=_formula_key)


def _consts_in_term(t: Term) -> frozenset:
    if t.kind == "const":
        return frozenset({t})
    if t.kind == "func":
        out = set()
        for a in t.args:
            out |= _consts_in_term(a)
        return frozenset(out)
    return frozenset()


def _consts_in_formula(f: Formula) -> frozenset:
    if f.kind == "atom":
        _, args = f.payload
        out = set()
        for a in args:
            out |= _consts_in_term(a)
        return frozenset(out)
    if f.kind == "neg":
        return _consts_in_formula(f.payload[0])
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        return _consts_in_formula(a) | _consts_in_formula(b)
    if f.kind in {"forall", "exists"}:
        return _consts_in_formula(f.payload[1])
    return frozenset()


def enumerate_terms(seq: Sequent) -> frozenset:
    out = set()
    for f in seq.antecedent | seq.succedent:
        out |= _consts_in_formula(f)
    return frozenset(out)


def _apply_simple(rule_name, seq, principal, ctx):
    table = {
        "and_L": apply_and_L, "and_R": apply_and_R,
        "or_L": apply_or_L, "or_R": apply_or_R,
        "imp_L": apply_imp_L, "imp_R": apply_imp_R,
        "neg_L": apply_neg_L, "neg_R": apply_neg_R,
        "forall_R": apply_forall_R, "exists_L": apply_exists_L,
    }
    return table[rule_name](seq, principal, ctx)


def _apply_inst(rule_name, seq, principal, ctx, term):
    if rule_name == "forall_L":
        return apply_forall_L(seq, principal, ctx, term)[0]
    if rule_name == "exists_R":
        return apply_exists_R(seq, principal, ctx, term)[0]
    raise ValueError(rule_name)


@dataclass
class BaselineStats:
    nodes_expanded: int = 0
    depth_max_seen: int = 0
    fresh_consts_used: int = 0
    aborted_node_budget: bool = False
    aborted_timeout: bool = False


def prove_baseline(seq: Sequent, depth: int, max_depth: int,
                   history: dict, ctx: RuleContext, stats: BaselineStats,
                   node_budget: int,
                   deadline: Optional[float] = None) -> Optional[ProofNode]:
    if stats.aborted_node_budget or stats.aborted_timeout:
        return None
    if stats.nodes_expanded >= node_budget:
        stats.aborted_node_budget = True
        return None
    if deadline is not None and time.perf_counter() >= deadline:
        stats.aborted_timeout = True
        return None
    stats.nodes_expanded += 1
    if depth > stats.depth_max_seen:
        stats.depth_max_seen = depth
    if depth > max_depth:
        return None
    if is_axiom(seq):
        ctx.axiom_closures += 1
        return ProofNode(rule="axiom", principal=None, sequent=seq, premises=())

    for rule_name, side in DETERMINISTIC_RULES:
        principal = _find_principal(seq, rule_name, side)
        if principal is None:
            continue
        try:
            premises = _apply_simple(rule_name, seq, principal, ctx)
        except RuleError:
            continue
        children = []
        for p in premises:
            child = prove_baseline(p, depth + 1, max_depth, history, ctx, stats, node_budget, deadline)
            if child is None:
                return None
            children.append(child)
        return ProofNode(rule=rule_name, principal=principal, sequent=seq,
                         premises=tuple(children))

    for rule_name, side in BRANCHING_RULES:
        principal = _find_principal(seq, rule_name, side)
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
            child = prove_baseline(p, depth + 1, max_depth, history, ctx, stats, node_budget, deadline)
            if child is None:
                ok = False
                break
            children.append(child)
        if ok:
            return ProofNode(rule=rule_name, principal=principal, sequent=seq,
                             premises=tuple(children))
        return None

    for rule_name, side in RETAINING_RULES:
        for principal in _all_principals(seq, rule_name, side):
            if stats.aborted_node_budget or stats.aborted_timeout:
                return None
            used = history.get(principal, frozenset())
            available = enumerate_terms(seq) - used
            for t in sorted(available, key=lambda c: c.name):
                if stats.aborted_node_budget or stats.aborted_timeout:
                    return None
                new_history = dict(history)
                new_history[principal] = used | {t}
                try:
                    premise = _apply_inst(rule_name, seq, principal, ctx, t)
                except RuleError:
                    continue
                child = prove_baseline(premise, depth + 1, max_depth, new_history, ctx, stats, node_budget, deadline)
                if child is not None:
                    return ProofNode(rule=rule_name, principal=principal,
                                     sequent=seq, premises=(child,))
            if stats.aborted_node_budget or stats.aborted_timeout:
                return None
            fresh_t = ctx.fresh_const("c")
            stats.fresh_consts_used += 1
            new_history = dict(history)
            new_history[principal] = used | {fresh_t}
            try:
                premise = _apply_inst(rule_name, seq, principal, ctx, fresh_t)
            except RuleError:
                continue
            child = prove_baseline(premise, depth + 1, max_depth, new_history, ctx, stats, node_budget, deadline)
            if child is not None:
                return ProofNode(rule=rule_name, principal=principal,
                                 sequent=seq, premises=(child,))

    return None


def prove(formula: Formula, max_depth: int = 50, node_budget: int = 50_000,
          timeout_s: Optional[float] = None) -> tuple:

    """Returns (proof, BaselineStats). proof is None on failure.
    node_budget caps total expansions; timeout_s is wall-clock seconds.
    """

    f = unfold_iff(formula)
    seq = make_sequent([], [f])
    ctx = RuleContext()
    stats = BaselineStats()
    deadline = time.perf_counter() + timeout_s if timeout_s is not None else None
    proof = prove_baseline(seq, 0, max_depth, {}, ctx, stats, node_budget, deadline)
    return proof, stats


def is_proved(formula: Formula, max_depth: int = 50, node_budget: int = 50_000,
              timeout_s: Optional[float] = None) -> bool:
    proof, _ = prove(formula, max_depth, node_budget, timeout_s)
    return proof is not None


def proof_depth(node: ProofNode) -> int:
    if not node.premises:
        return 1
    return 1 + max(proof_depth(p) for p in node.premises)


def proof_size(node: ProofNode) -> int:
    return 1 + sum(proof_size(p) for p in node.premises)
