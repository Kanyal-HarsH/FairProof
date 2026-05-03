"""
ImprovedConfig knobs and the C1..C4 factory.
C0 is not here; it lives in prover.baseline.prove.

"""

from dataclasses import dataclass


@dataclass
class ImprovedConfig:
    # B
    iterative_deepening: bool = True
    base_node_budget: int = 1_000
    budget_growth: float = 2.0
    max_iterations: int = 12

    max_depth: int = 200
    max_total_nodes: int = 200_000

    base_term_depth: int = 0
    term_depth_growth: int = 1

    # M
    use_memo: bool = True
    memo_max_entries: int = 10_000_000

    # R
    use_relevance: bool = True
    rg_k1: float = 4.0  # pred_overlap weight
    rg_k2: float = 2.0  # term_reuse weight
    rg_k3: float = 1.0  # term_depth penalty
    rg_top_k: int = 3
    rg_min_candidates: int = 2

    fairness_period: int = 32  # round-robin every K-th retaining choice


def config_for(label: str) -> ImprovedConfig:

    """Build a config for one of C1, C2, C3, C4."""

    label = label.upper()
    if label == "C1":
        return ImprovedConfig(use_memo=False, use_relevance=False)
    if label == "C2":
        return ImprovedConfig(use_memo=True, use_relevance=False)
    if label == "C3":
        return ImprovedConfig(use_memo=False, use_relevance=True)
    if label == "C4":
        return ImprovedConfig(use_memo=True, use_relevance=True)
    raise ValueError(f"unknown ablation label {label!r}; expected one of C1, C2, C3, C4")
