"""Per-cell metric wrapper. Returns a dict ready for json.dumps."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import time
from datetime import datetime, timezone
from typing import Optional

import psutil

from logic.parser import parse, ParseError
from prover.baseline import (
    prove_baseline as _prove_baseline_inner,
    BaselineStats, ProofNode, proof_depth,
    unfold_iff, enumerate_terms,
)
from prover.improved_algorithm import prove_improved
from prover.config import ImprovedConfig, config_for
from prover.rules import RuleContext
from prover.memoisation import MemoisationCache
from logic.sequent import make_sequent


_PROCESS = psutil.Process()
PROVER_VERSION = "1.0.0"
_PYTHON_HASH_SEED = os.environ.get("PYTHONHASHSEED", "random")
_MACHINE_ID = f"{platform.node()}-{platform.system()}-{platform.machine()}"


def _peak_memory_mb() -> float:
    return _PROCESS.memory_info().rss / 1_048_576.0


def _derive_seed(label: str, config_name: str, run_index: int) -> int:
    # Same (label, config, run_index) -> same seed across runs.
    h = hashlib.sha256(f"{label}|{config_name}|{run_index}".encode("ascii")).hexdigest()
    return int(h[:16], 16)


def _make_config(config_name: str, node_budget: int) -> Optional[ImprovedConfig]:
    if config_name == "C0":
        return None
    cfg = config_for(config_name)
    cfg.max_total_nodes = node_budget
    cfg.iterative_deepening = True
    return cfg


def _classify_status(proof, stats, config_name: str) -> str:
    if proof is not None:
        return "PROVED"
    if getattr(stats, "aborted_timeout", False):
        return "TIMEOUT"
    if config_name == "C0":
        if getattr(stats, "aborted_node_budget", False):
            return "NODE_LIMIT"
    else:
        if getattr(stats, "aborted_total_budget", False) or getattr(stats, "aborted_node_budget", False):
            return "NODE_LIMIT"
    return "FAIL_OPEN"


def _run_baseline(formula, node_budget: int, max_depth: int,
                  timeout_s: Optional[float]):
    f = unfold_iff(formula)
    seq = make_sequent([], [f])
    ctx = RuleContext()
    stats = BaselineStats()
    deadline = time.perf_counter() + timeout_s if timeout_s is not None else None
    proof = _prove_baseline_inner(seq, 0, max_depth, {}, ctx, stats,
                                  node_budget, deadline)
    return proof, stats, ctx


def _run_improved(formula, cfg: ImprovedConfig, timeout_s: Optional[float]):
    proof, stats = prove_improved(formula, cfg, timeout_s=timeout_s)
    return proof, stats


def run_cell(formula_str: str, label: str, config_name: str,
             run_index: int, node_budget: int = 50_000,
             max_depth: int = 200,
             timeout_s: Optional[float] = 30.0) -> dict:
    """Runs one (label, config, run_index) cell. Returns a JSONL-ready dict.
    On parse failure the status field is ERROR and the metrics are zero.
    """
    seed = _derive_seed(label, config_name, run_index)
    random.seed(seed)
    record = {
        "label": label,
        "config": config_name,
        "run_index": run_index,
        "status": "ERROR",
        "wall_time_s": 0.0,
        "node_count": 0,
        "peak_memory_mb": 0.0,
        "proof_depth": None,
        "cache_hit_rate": None,
        "distinct_instantiations": 0,
        "eigenvariables_introduced": 0,
        "max_term_depth": 0,
        "iteration_at_solution": None,
        "fraction_closed_before_branch": 0.0,
        "seed": seed,
        "python_hash_seed": _PYTHON_HASH_SEED,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine_id": _MACHINE_ID,
        "prover_version": PROVER_VERSION,
    }
    try:
        formula = parse(formula_str)
    except ParseError:
        return record

    t0 = time.perf_counter()
    if config_name == "C0":
        proof, stats, ctx = _run_baseline(formula, node_budget, max_depth, timeout_s)
        eigenvar_count = ctx.eigenvar_count
        max_term_depth_seen = ctx.max_term_depth
        axiom_closures = ctx.axiom_closures
        branching_applications = ctx.branching_applications
        cache_hit_rate = None
        iteration_at_solution = None
    else:
        cfg = _make_config(config_name, node_budget)
        cfg.max_depth = max_depth
        proof, stats = _run_improved(formula, cfg, timeout_s)
        eigenvar_count = stats.eigenvar_count
        max_term_depth_seen = stats.max_term_depth
        axiom_closures = stats.axiom_closures
        branching_applications = stats.branching_applications
        if cfg.use_memo:
            total_lookups = stats.cache_hits + stats.cache_misses
            cache_hit_rate = (stats.cache_hits / total_lookups) if total_lookups > 0 else 0.0
        else:
            cache_hit_rate = None
        iteration_at_solution = stats.iteration_at_solution if proof is not None else None
    wall_time_s = time.perf_counter() - t0

    record["wall_time_s"] = wall_time_s
    record["node_count"] = stats.nodes_expanded
    record["peak_memory_mb"] = _peak_memory_mb()
    record["proof_depth"] = proof_depth(proof) if proof is not None else None
    record["cache_hit_rate"] = cache_hit_rate
    record["distinct_instantiations"] = getattr(stats, "distinct_instantiations", 0)
    record["iteration_at_solution"] = iteration_at_solution
    record["eigenvariables_introduced"] = eigenvar_count
    record["max_term_depth"] = max_term_depth_seen
    denom = axiom_closures + branching_applications
    record["fraction_closed_before_branch"] = (
        (axiom_closures / denom) if denom > 0 else 1.0
    )

    record["status"] = _classify_status(proof, stats, config_name)
    return record


JSONL_FIELDS = (
    "label", "config", "run_index", "status",
    "wall_time_s", "node_count", "peak_memory_mb", "proof_depth",
    "cache_hit_rate", "distinct_instantiations",
    "eigenvariables_introduced", "max_term_depth",
    "iteration_at_solution", "fraction_closed_before_branch",
    "seed", "python_hash_seed", "timestamp_iso", "machine_id",
    "prover_version",
)
