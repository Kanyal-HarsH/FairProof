"""Experiment harness. Subcommands: pilot, full, one. Writes one JSONL per cell.

    python experiments/run_experiments.py pilot
    python experiments/run_experiments.py full --reps 5 --workers 16
    python experiments/run_experiments.py one --label pel_24 --config C4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._metrics import run_cell, JSONL_FIELDS

DATASET_PATH = _REPO_ROOT / "data" / "dataset.jsonl"
RESULTS_RAW = _REPO_ROOT / "results" / "raw"

CONFIGS = ("C0", "C1", "C2", "C3", "C4")
PILOT_TAG = "HOU"


def _load_dataset() -> list:
    if not DATASET_PATH.exists():
        return []
    return [
        json.loads(ln)
        for ln in DATASET_PATH.read_text(encoding="ascii").splitlines()
        if ln
    ]


def _filter_records(records, label=None, dataset=None, tag=None) -> list:
    out = list(records)
    if label:
        out = [r for r in out if r["label"] == label]
    if dataset:
        out = [r for r in out if r["dataset"].upper() == dataset.upper()]
    if tag:
        out = [r for r in out if tag.upper() in (t.upper() for t in r.get("tags", []))]
    return out


def _open_run_file(name_hint: str) -> Path:
    RESULTS_RAW.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_RAW / f"run_{name_hint}_{stamp}.jsonl"


def _write_record(fh, record: dict) -> None:
    ordered = {k: record.get(k) for k in JSONL_FIELDS}
    fh.write(json.dumps(ordered, ensure_ascii=True))
    fh.write("\n")


def _run_cell_kwargs(args):
    # top-level so ProcessPoolExecutor can pickle it
    return run_cell(**args)


def _run_cells(records, configs, reps: int, node_budget: int, max_depth: int,
               out_path: Path, timeout_s: Optional[float] = 30.0,
               workers: int = 1) -> tuple:
    cells = [
        (run_index, r, cfg_name)
        for run_index in range(1, reps + 1)
        for r in records
        for cfg_name in configs
    ]
    n_total = len(cells)
    n_done = 0
    n_proved = 0
    timeout_msg = f"timeout {timeout_s:.0f}s/cell" if timeout_s else "no timeout"
    worker_msg = f"{workers} workers" if workers > 1 else "sequential"
    print(f"[harness] {n_total} cells -> {out_path.name} ({timeout_msg}, {worker_msg})",
          flush=True)

    if workers <= 1:
        with open(out_path, "w", encoding="ascii", newline="\n") as fh:
            for run_index, r, cfg_name in cells:
                cell_t0 = time.perf_counter()
                print(
                    f"[{n_done + 1:>5d}/{n_total}] {cfg_name} {r['label']:24s} ...",
                    end="", flush=True,
                )
                rec = run_cell(
                    formula_str=r["formula"], label=r["label"],
                    config_name=cfg_name, run_index=run_index,
                    node_budget=node_budget, max_depth=max_depth,
                    timeout_s=timeout_s,
                )
                _write_record(fh, rec)
                n_done += 1
                if rec["status"] == "PROVED":
                    n_proved += 1
                cell_dt = time.perf_counter() - cell_t0
                print(
                    f" {rec['status']:11s} nodes={rec['node_count']:>6d} dt={cell_dt:.3f}s",
                    flush=True,
                )
        return n_done, n_proved

    with open(out_path, "w", encoding="ascii", newline="\n") as fh, \
         ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for run_index, r, cfg_name in cells:
            kwargs = dict(
                formula_str=r["formula"], label=r["label"],
                config_name=cfg_name, run_index=run_index,
                node_budget=node_budget, max_depth=max_depth,
                timeout_s=timeout_s,
            )
            fut = pool.submit(_run_cell_kwargs, kwargs)
            futures[fut] = (r["label"], cfg_name, run_index)
        for fut in as_completed(futures):
            label, cfg_name, run_index = futures[fut]
            rec = fut.result()
            _write_record(fh, rec)
            fh.flush()
            n_done += 1
            if rec["status"] == "PROVED":
                n_proved += 1
            print(
                f"[{n_done:>5d}/{n_total}] {cfg_name} {label:24s} ... "
                f"{rec['status']:11s} nodes={rec['node_count']:>6d} "
                f"dt={rec['wall_time_s']:.3f}s",
                flush=True,
            )
    return n_done, n_proved


def cmd_pilot(args) -> int:
    records = _filter_records(_load_dataset(), tag=PILOT_TAG)
    if not records:
        print(f"[error] no records tagged {PILOT_TAG}", file=sys.stderr)
        return 2
    out_path = _open_run_file("pilot")
    n_done, n_proved = _run_cells(
        records, CONFIGS, reps=1,
        node_budget=args.node_budget, max_depth=args.max_depth,
        out_path=out_path, timeout_s=args.timeout_s,
        workers=args.workers,
    )
    print(f"[pilot] done. {n_proved}/{n_done} cells PROVED. file: {out_path}")
    return 0


def cmd_full(args) -> int:
    records = _load_dataset()
    if not records:
        print("[error] data/dataset.jsonl is empty; "
              "run experiments/build_corpus.py all", file=sys.stderr)
        return 2
    out_path = _open_run_file("full")
    n_done, n_proved = _run_cells(
        records, CONFIGS, reps=args.reps,
        node_budget=args.node_budget, max_depth=args.max_depth,
        out_path=out_path, timeout_s=args.timeout_s,
        workers=args.workers,
    )
    print(f"[full] done. {n_proved}/{n_done} cells PROVED. file: {out_path}")
    return 0


def cmd_one(args) -> int:
    records = _load_dataset()
    selected = _filter_records(records, label=args.label)
    if not selected:
        print(f"[error] no record with label {args.label!r}", file=sys.stderr)
        return 2
    out_path = _open_run_file(f"one_{args.label}_{args.config}")
    cfgs = (args.config,)
    n_done, n_proved = _run_cells(
        selected, cfgs, reps=1,
        node_budget=args.node_budget, max_depth=args.max_depth,
        out_path=out_path, timeout_s=args.timeout_s,
        workers=args.workers,
    )
    print(f"[one] done. {n_proved}/{n_done} cells PROVED. file: {out_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the experiment harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    default_workers = max(1, (os.cpu_count() or 2))

    sp = sub.add_parser("pilot", help="one rep, HOU textbook examples only (smoke test)")
    sp.add_argument("--node-budget", type=int, default=20_000)
    sp.add_argument("--max-depth", type=int, default=100)
    sp.add_argument("--timeout-s", type=float, default=30.0,
                    help="wall-clock cutoff per cell (seconds); 0 to disable")
    sp.add_argument("--workers", type=int, default=default_workers,
                    help="parallel worker processes (1 = sequential)")
    sp.set_defaults(func=cmd_pilot)

    sf = sub.add_parser("full", help="all formulas, all configs, 3 reps")
    sf.add_argument("--reps", type=int, default=3)
    sf.add_argument("--node-budget", type=int, default=50_000)
    sf.add_argument("--max-depth", type=int, default=200)
    sf.add_argument("--timeout-s", type=float, default=30.0,
                    help="wall-clock cutoff per cell (seconds); 0 to disable")
    sf.add_argument("--workers", type=int, default=default_workers,
                    help="parallel worker processes (1 = sequential)")
    sf.set_defaults(func=cmd_full)

    so = sub.add_parser("one", help="one formula, one config, one rep")
    so.add_argument("--label", required=True)
    so.add_argument("--config", required=True, choices=CONFIGS)
    so.add_argument("--node-budget", type=int, default=50_000)
    so.add_argument("--max-depth", type=int, default=200)
    so.add_argument("--timeout-s", type=float, default=30.0,
                    help="wall-clock cutoff per cell (seconds); 0 to disable")
    so.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes (1 = sequential)")
    so.set_defaults(func=cmd_one)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(args, "timeout_s") and args.timeout_s is not None and args.timeout_s <= 0:
        args.timeout_s = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
