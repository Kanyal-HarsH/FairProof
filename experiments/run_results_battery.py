"""Battery: full + stress + compare + analyse. test generates a JSONL."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.run_experiments import (
    CONFIGS, _load_dataset, _open_run_file, _run_cells,
)


# Hard cases for the stress run. Expect some to still hit NODE_LIMIT.

HARD_LABELS = (
    # Pelletier FOL_QHEAVY core
    "pel_18", "pel_19", "pel_20", "pel_21", "pel_22", "pel_23",
    "pel_24", "pel_26", "pel_27", "pel_28", "pel_29", "pel_30",
    "pel_31", "pel_32", "pel_33", "pel_34", "pel_35", "pel_36",
    "pel_37", "pel_38", "pel_44", "pel_46",

    # Pigeonhole family

    "php_2", "php_3", "php_4", "php_5", "php_6",

    # Drinker variants

    "drinker_canonical", "drinker_dual",
    "drinker_var1", "drinker_var2", "drinker_var3",

    # Parametric: valid + invalid quantifier alternations

    "param_01", "param_02", "param_03", "param_04", "param_05",
    "param_06", "param_07", "param_08", "param_09", "param_10",
)

# Curated comparison set: a small panel where the report discussion can
# point at specific numbers. Mix of easy, medium, and hard.

COMPARE_LABELS = (
    "pel_1", "pel_2", "pel_18", "pel_24", "pel_38",
    "drinker_canonical", "drinker_dual", "drinker_var2",
    "php_2", "php_3",
    "param_02", "param_04", "param_09",
    "hou_ex01", "hou_ex07", "hou_ex08",
    "inv_01", "inv_02",
)


def _filter_by_labels(records, labels) -> list:
    label_set = set(labels)
    return [r for r in records if r["label"] in label_set]


def _phase_banner(name: str, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    print(f"\n[{name}] phase complete  elapsed={elapsed:.1f}s\n", flush=True)


def cmd_full(args) -> int:
    records = _load_dataset()
    if not records:
        print("[error] data/dataset.jsonl is empty; run build_corpus.py all", file=sys.stderr)
        return 2
    out = _open_run_file("full")
    print(f"[full] {len(records) * len(CONFIGS) * args.reps} cells -> {out.name}", flush=True)
    t0 = time.perf_counter()
    n_done, n_proved = _run_cells(
        records, CONFIGS, reps=args.reps,
        node_budget=args.node_budget, max_depth=args.max_depth,
        out_path=out, timeout_s=args.timeout_s,
        workers=args.workers,
    )
    _phase_banner(f"full: {n_proved}/{n_done} PROVED", t0)
    return 0


def cmd_stress(args) -> int:
    records = _load_dataset()
    selected = _filter_by_labels(records, HARD_LABELS)
    if not selected:
        print(f"[error] no hard records found", file=sys.stderr)
        return 2
    out = _open_run_file("stress")
    print(f"[stress] {len(selected) * len(CONFIGS) * args.reps} cells "
          f"({len(selected)} hard formulas) -> {out.name}", flush=True)
    t0 = time.perf_counter()
    n_done, n_proved = _run_cells(
        selected, CONFIGS, reps=args.reps,
        node_budget=args.stress_node_budget, max_depth=args.stress_max_depth,
        out_path=out, timeout_s=args.stress_timeout_s,
        workers=args.workers,
    )
    _phase_banner(f"stress: {n_proved}/{n_done} PROVED", t0)
    return 0


def cmd_compare(args) -> int:
    records = _load_dataset()
    selected = _filter_by_labels(records, COMPARE_LABELS)
    if not selected:
        print(f"[error] no comparison records found", file=sys.stderr)
        return 2
    out = _open_run_file("compare")
    print(f"[compare] {len(selected) * len(CONFIGS)} cells "
          f"({len(selected)} curated formulas, 1 rep) -> {out.name}", flush=True)
    t0 = time.perf_counter()
    n_done, n_proved = _run_cells(
        selected, CONFIGS, reps=1,
        node_budget=args.compare_node_budget, max_depth=args.compare_max_depth,
        out_path=out, timeout_s=args.compare_timeout_s,
        workers=args.workers,
    )
    _phase_banner(f"compare: {n_proved}/{n_done} PROVED", t0)
    return 0


def cmd_analyse(args) -> int:
    from experiments import analyse
    print("[analyse] producing results/tables, results/figures, results/summary.txt", flush=True)
    t0 = time.perf_counter()
    rc = analyse.cmd_all()
    _phase_banner("analyse", t0)
    return rc


def cmd_all(args) -> int:
    started = time.perf_counter()

    print("=" * 60)
    print("PHASE 1 of 4: full report-grade run")
    print("=" * 60)
    rc = cmd_full(args)
    if rc != 0:
        return rc

    print("=" * 60)
    print("PHASE 2 of 4: stress run on hard formulas")
    print("=" * 60)
    rc = cmd_stress(args)
    if rc != 0:
        return rc

    print("=" * 60)
    print("PHASE 3 of 4: side-by-side comparison run")
    print("=" * 60)
    rc = cmd_compare(args)
    if rc != 0:
        return rc

    print("=" * 60)
    print("PHASE 4 of 4: analyse and produce report artefacts")
    print("=" * 60)
    rc = cmd_analyse(args)
    if rc != 0:
        return rc

    total = time.perf_counter() - started
    print()
    print("=" * 60)
    print(f"BATTERY DONE in {total:.1f}s ({total / 60:.1f} min)")
    print("=" * 60)
    print()
    print("Outputs:")
    print(f"  results/raw/run_full_*.jsonl")
    print(f"  results/raw/run_stress_*.jsonl")
    print(f"  results/raw/run_compare_*.jsonl")
    print(f"  results/tables/table{{1..5}}_*.csv")
    print(f"  results/figures/{{cactus,scatter,ecdf,boxplot}}.svg")
    print(f"  results/summary.txt")
    print()
    print("Open results/summary.txt for the headline H1..H6 results.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full report-grade result generation battery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared budget knobs
    def _add_budget(sp, kind: str = "full"):
        prefix = "" if kind == "full" else f"{kind}_"
        sp.add_argument(f"--{prefix.replace('_', '-')}reps".lstrip("-"),
                        dest=f"{prefix}reps", type=int,
                        default=(3 if kind == "full" else 1 if kind == "compare" else 3))

    default_workers = max(1, (os.cpu_count() or 2))

    sp_full = sub.add_parser("full", help="91 x 5 x N reps at report budget")
    sp_full.add_argument("--reps", type=int, default=3)
    sp_full.add_argument("--node-budget", type=int, default=200_000)
    sp_full.add_argument("--max-depth", type=int, default=400)
    sp_full.add_argument("--timeout-s", type=float, default=30.0,
                         help="wall-clock cutoff per cell (seconds); 0 to disable")
    sp_full.add_argument("--workers", type=int, default=default_workers,
                         help="parallel worker processes (1 = sequential)")
    sp_full.set_defaults(func=cmd_full)

    sp_stress = sub.add_parser("stress", help="hard subset at max budgets")
    sp_stress.add_argument("--reps", type=int, default=3)
    sp_stress.add_argument("--stress-node-budget", type=int, default=1_000_000)
    sp_stress.add_argument("--stress-max-depth", type=int, default=800)
    sp_stress.add_argument("--stress-timeout-s", type=float, default=120.0,
                           help="wall-clock cutoff per cell (seconds); 0 to disable")
    sp_stress.add_argument("--workers", type=int, default=default_workers,
                           help="parallel worker processes (1 = sequential)")
    sp_stress.set_defaults(func=cmd_stress)

    sp_compare = sub.add_parser("compare", help="curated formulas, all 5 configs, 1 rep")
    sp_compare.add_argument("--compare-node-budget", type=int, default=300_000)
    sp_compare.add_argument("--compare-max-depth", type=int, default=500)
    sp_compare.add_argument("--compare-timeout-s", type=float, default=60.0,
                            help="wall-clock cutoff per cell (seconds); 0 to disable")
    sp_compare.add_argument("--workers", type=int, default=default_workers,
                            help="parallel worker processes (1 = sequential)")
    sp_compare.set_defaults(func=cmd_compare)

    sp_analyse = sub.add_parser("analyse", help="run analyse.py all on collected JSONLs")
    sp_analyse.set_defaults(func=cmd_analyse)

    sp_all = sub.add_parser("all", help="full + stress + compare + analyse")
    sp_all.add_argument("--reps", type=int, default=3,
                        help="reps for the full and stress phases")
    sp_all.add_argument("--node-budget", type=int, default=200_000,
                        help="node budget for the full phase")
    sp_all.add_argument("--max-depth", type=int, default=400,
                        help="max depth for the full phase")
    sp_all.add_argument("--timeout-s", type=float, default=30.0,
                        help="wall-clock cutoff per cell for the full phase (seconds); 0 to disable")
    sp_all.add_argument("--stress-node-budget", type=int, default=1_000_000)
    sp_all.add_argument("--stress-max-depth", type=int, default=800)
    sp_all.add_argument("--stress-timeout-s", type=float, default=120.0,
                        help="wall-clock cutoff per cell for the stress phase (seconds); 0 to disable")
    sp_all.add_argument("--compare-node-budget", type=int, default=300_000)
    sp_all.add_argument("--compare-max-depth", type=int, default=500)
    sp_all.add_argument("--compare-timeout-s", type=float, default=60.0,
                        help="wall-clock cutoff per cell for the compare phase (seconds); 0 to disable")
    sp_all.add_argument("--workers", type=int, default=default_workers,
                        help="parallel worker processes for every phase (1 = sequential)")
    sp_all.set_defaults(func=cmd_all)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    for attr in ("timeout_s", "stress_timeout_s", "compare_timeout_s"):
        if hasattr(args, attr):
            v = getattr(args, attr)
            if v is not None and v <= 0:
                setattr(args, attr, None)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
