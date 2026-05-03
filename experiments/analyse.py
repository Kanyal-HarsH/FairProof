"""Generate tables, figures and the H1..H6 summary from results/raw/*.jsonl.
Subcommands: tables | figures | stats | all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments._statistics import (
    wilcoxon_log_ratio, mcnemar_exact, median_ratio, sign_test_one_sided,
)

DATASET_PATH = _REPO_ROOT / "data" / "dataset.jsonl"
RESULTS_RAW = _REPO_ROOT / "results" / "raw"
RESULTS_TABLES = _REPO_ROOT / "results" / "tables"
RESULTS_FIGURES = _REPO_ROOT / "results" / "figures"
RESULTS_SUMMARY = _REPO_ROOT / "results" / "summary.txt"

CONFIGS = ("C0", "C1", "C2", "C3", "C4")


def _load_records() -> pd.DataFrame:
    records = []
    if RESULTS_RAW.exists():
        for path in sorted(RESULTS_RAW.glob("*.jsonl")):
            for line in path.read_text(encoding="ascii").splitlines():
                if line.strip():
                    records.append(json.loads(line))
    if not records:
        print("[error] no records under results/raw/. "
              "Run experiments/run_experiments.py pilot or full first.",
              file=sys.stderr)
        sys.exit(2)
    df = pd.DataFrame.from_records(records)
    df = df.drop_duplicates(subset=["label", "config", "run_index"], keep="last")
    return df


def _load_dataset_meta() -> pd.DataFrame:
    rows = [json.loads(ln) for ln in DATASET_PATH.read_text(encoding="ascii").splitlines() if ln]
    df = pd.DataFrame.from_records(rows)
    return df[["label", "dataset", "status_label", "tags", "source"]]


def _median_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "wall_time_s", "node_count", "peak_memory_mb", "proof_depth",
        "cache_hit_rate", "distinct_instantiations",
        "eigenvariables_introduced", "max_term_depth",
        "iteration_at_solution", "fraction_closed_before_branch",
    ]
    agg = {col: "median" for col in numeric_cols if col in df.columns}
    grouped = df.groupby(["label", "config"], as_index=False).agg(
        {**agg, "status": lambda s: s.mode().iat[0] if len(s.mode()) else "ERROR"},
    )
    return grouped


# tables

def _ensure_dirs():
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)


DIFFICULTY_TIERS = ("easy", "medium", "hard", "challenging")
EASY_MEDIUM_THRESHOLD_S = 0.01


def _classify_one(c0_status, c0_wall_s, other_proved: bool) -> str | None:
    # easy: C0 PROVED < 10ms. medium: C0 PROVED in [10ms, 30s]. hard: C0 fails
    # but some Ci PROVES. challenging: no config PROVES.
    if c0_status is None or (isinstance(c0_status, float) and pd.isna(c0_status)):
        return "hard" if other_proved else "challenging"
    if c0_status == "ERROR":
        return None
    if c0_status == "PROVED":
        try:
            t = float(c0_wall_s)
        except (TypeError, ValueError):
            return None
        return "easy" if t < EASY_MEDIUM_THRESHOLD_S else "medium"
    return "hard" if other_proved else "challenging"


def _classify_difficulty_per_formula(median: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    # COUNTERSAT formulas are tagged non_theorem and excluded from the tier count.
    c0 = median[median["config"] == "C0"][["label", "status", "wall_time_s"]].rename(
        columns={"status": "c0_status", "wall_time_s": "c0_wall_s"}
    )
    others = median[median["config"].isin(("C1", "C2", "C3", "C4"))]
    other_proved = (
        others.assign(_p=lambda d: d["status"] == "PROVED")
        .groupby("label")["_p"].any()
        .rename("other_proved")
    )
    df = (
        meta[["label", "dataset", "status_label"]]
        .merge(c0, on="label", how="left")
        .merge(other_proved, on="label", how="left")
    )
    df["other_proved"] = df["other_proved"].fillna(False)
    rows = []
    for _, r in df.iterrows():
        if r["status_label"] == "COUNTERSAT":
            tier = "non_theorem"
        else:
            tier = _classify_one(r.get("c0_status"), r.get("c0_wall_s"),
                                 bool(r["other_proved"]))
            if tier is None:
                tier = "unknown"
        rows.append({
            "label": r["label"],
            "dataset": r["dataset"],
            "status_label": r["status_label"],
            "difficulty": tier,
        })
    return pd.DataFrame(rows)


def table1_dataset_summary(meta: pd.DataFrame, median: pd.DataFrame) -> pd.DataFrame:
    counts = meta.groupby("dataset").size().reset_index(name="count")
    statuses = meta.groupby("dataset")["status_label"].agg(
        lambda s: ",".join(sorted(set(s)))
    ).reset_index().rename(columns={"status_label": "validity_status"})
    sources = meta.groupby("dataset")["source"].agg(
        lambda s: s.iloc[0]
    ).reset_index()
    out = counts.merge(statuses, on="dataset").merge(sources, on="dataset")

    per_formula = _classify_difficulty_per_formula(median, meta)
    tier_counts = (
        per_formula.groupby(["dataset", "difficulty"]).size().unstack(fill_value=0)
    )
    for tier in DIFFICULTY_TIERS:
        col = f"n_{tier}"
        out[col] = out["dataset"].map(
            tier_counts[tier] if tier in tier_counts.columns else {}
        ).fillna(0).astype(int)
    out["n_non_theorem"] = out["dataset"].map(
        tier_counts["non_theorem"] if "non_theorem" in tier_counts.columns else {}
    ).fillna(0).astype(int)

    out = out[["dataset", "source", "count", "validity_status",
               "n_easy", "n_medium", "n_hard", "n_challenging", "n_non_theorem"]]
    return out.sort_values("dataset").reset_index(drop=True)


def table2_baseline_vs_full(median: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = median.merge(meta[["label", "dataset"]], on="label")
    rows = []
    for dataset, sub in df.groupby("dataset"):
        wide = sub.pivot(index="label", columns="config",
                         values=["status", "wall_time_s"])
        if "status" not in wide.columns.get_level_values(0):
            continue
        statuses = wide["status"]
        times = wide["wall_time_s"]
        c0 = "C0" if "C0" in statuses.columns else None
        c4 = "C4" if "C4" in statuses.columns else None
        if c0 is None or c4 is None:
            continue
        solved_c0 = (statuses[c0] == "PROVED").sum()
        solved_c4 = (statuses[c4] == "PROVED").sum()
        both = (statuses[c0] == "PROVED") & (statuses[c4] == "PROVED")
        x = times.loc[both, c0].tolist()
        y = times.loc[both, c4].tolist()
        med_c0 = float(np.median(x)) if x else float("nan")
        med_c4 = float(np.median(y)) if y else float("nan")
        ratio = median_ratio(x, y)
        wil = wilcoxon_log_ratio(x, y)
        only_c0 = ((statuses[c0] == "PROVED") & (statuses[c4] != "PROVED")).sum()
        only_c4 = ((statuses[c0] != "PROVED") & (statuses[c4] == "PROVED")).sum()
        mc = mcnemar_exact(int(only_c0), int(only_c4))
        rows.append({
            "dataset": dataset,
            "|D|": len(sub["label"].unique()),
            "solved_C0": int(solved_c0),
            "solved_C4": int(solved_c4),
            "median_wall_time_C0_solved_by_both": med_c0,
            "median_wall_time_C4_solved_by_both": med_c4,
            "median_runtime_ratio": ratio,
            "wilcoxon_W": wil["W"],
            "wilcoxon_p": wil["p"],
            "mcnemar_p": mc["p"],
        })
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def table3_ablation(median: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = median.merge(meta[["label", "dataset"]], on="label")
    rows = []
    for dataset, sub in df.groupby("dataset"):
        row = {"dataset": dataset, "|D|": len(sub["label"].unique())}
        for cfg in CONFIGS:
            cells = sub[sub["config"] == cfg]
            row[f"solved_{cfg}"] = int((cells["status"] == "PROVED").sum())
            solved_times = cells.loc[cells["status"] == "PROVED", "wall_time_s"]
            row[f"median_wall_time_{cfg}"] = (
                float(solved_times.median()) if len(solved_times) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def table4_hard_only(median: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = median.merge(meta[["label", "dataset"]], on="label")
    wide = df.pivot(index=["label", "dataset"], columns="config",
                    values=["status", "wall_time_s", "node_count"]).reset_index()
    rows = []
    for _, row in wide.iterrows():
        try:
            status_c0 = row[("status", "C0")]
            time_c0 = row[("wall_time_s", "C0")]
            status_c4 = row[("status", "C4")]
            time_c4 = row[("wall_time_s", "C4")]
            nodes_c0 = row[("node_count", "C0")]
            nodes_c4 = row[("node_count", "C4")]
        except KeyError:
            continue
        is_hard = (status_c0 != "PROVED") or (
            isinstance(time_c0, (int, float)) and time_c0 > 5.0
        )
        if not is_hard:
            continue
        if status_c0 == "PROVED" and status_c4 == "PROVED" and time_c0 > 0:
            gain = f"ratio={time_c4 / time_c0:.3f}"
        elif status_c4 == "PROVED" and status_c0 != "PROVED":
            gain = "C4 only"
        elif status_c0 == "PROVED" and status_c4 != "PROVED":
            gain = "C0 only"
        else:
            # Both failed; report which got further by node count for the
            # report's hard-only discussion (lower nodes = earlier abort,
            # higher nodes = explored more before timeout/budget).
            try:
                if isinstance(nodes_c0, (int, float)) and isinstance(nodes_c4, (int, float)):
                    if nodes_c4 > nodes_c0 * 1.1:
                        gain = "neither (C4 explored more)"
                    elif nodes_c0 > nodes_c4 * 1.1:
                        gain = "neither (C0 explored more)"
                    else:
                        gain = "neither (similar)"
                else:
                    gain = "neither"
            except (TypeError, ValueError):
                gain = "neither"
        rows.append({
            "label": row[("label", "")],
            "dataset": row[("dataset", "")],
            "status_C0": status_c0,
            "wall_time_C0": time_c0,
            "status_C4": status_c4,
            "wall_time_C4": time_c4,
            "node_count_C0": nodes_c0,
            "node_count_C4": nodes_c4,
            "gain": gain,
        })
    return pd.DataFrame(rows)


def table5_memoisation_stats(median: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = median.merge(meta[["label", "dataset"]], on="label")
    rows = []
    for cfg in ("C2", "C4"):
        for dataset, sub in df[df["config"] == cfg].groupby("dataset"):
            rows.append({
                "config": cfg,
                "dataset": dataset,
                "mean_cache_hit_rate": float(sub["cache_hit_rate"].mean(skipna=True))
                    if sub["cache_hit_rate"].notna().any() else float("nan"),
                "median_distinct_instantiations": float(sub["distinct_instantiations"].median()),
                "max_node_count": int(sub["node_count"].max()),
                "peak_memory_mb": float(sub["peak_memory_mb"].max()),
            })
    return pd.DataFrame(rows).sort_values(["config", "dataset"]).reset_index(drop=True)


def table6_status_matrix(median: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = median.merge(meta[["label", "dataset"]], on="label")
    statuses = df.pivot(index=["label", "dataset"], columns="config", values="status").reset_index()
    times = df.pivot(index=["label", "dataset"], columns="config", values="wall_time_s").reset_index()
    nodes = df.pivot(index=["label", "dataset"], columns="config", values="node_count").reset_index()
    out = statuses[["label", "dataset"]].copy()
    for cfg in CONFIGS:
        if cfg in statuses.columns:
            out[f"status_{cfg}"] = statuses[cfg]
        if cfg in times.columns:
            out[f"wall_time_{cfg}"] = times[cfg]
        if cfg in nodes.columns:
            out[f"node_count_{cfg}"] = nodes[cfg]
    return out.sort_values(["dataset", "label"]).reset_index(drop=True)


def cmd_tables(args=None) -> int:
    _ensure_dirs()
    df = _load_records()
    median = _median_per_cell(df)
    meta = _load_dataset_meta()
    table1_dataset_summary(meta, median).to_csv(
        RESULTS_TABLES / "table1_dataset_summary.csv", index=False)
    table2_baseline_vs_full(median, meta).to_csv(
        RESULTS_TABLES / "table2_baseline_vs_full.csv", index=False)
    table3_ablation(median, meta).to_csv(
        RESULTS_TABLES / "table3_ablation.csv", index=False)
    table4_hard_only(median, meta).to_csv(
        RESULTS_TABLES / "table4_hard_only.csv", index=False)
    table5_memoisation_stats(median, meta).to_csv(
        RESULTS_TABLES / "table5_memoisation_stats.csv", index=False)
    table6_status_matrix(median, meta).to_csv(
        RESULTS_TABLES / "table6_status_matrix.csv", index=False)
    print(f"[tables] wrote 6 CSVs to {RESULTS_TABLES.relative_to(_REPO_ROOT)}")

    per_formula = _classify_difficulty_per_formula(median, meta)
    counts = per_formula["difficulty"].value_counts().to_dict()
    n_theorems = sum(counts.get(t, 0) for t in DIFFICULTY_TIERS)
    parts = []
    for tier in DIFFICULTY_TIERS:
        c = counts.get(tier, 0)
        pct = (100.0 * c / n_theorems) if n_theorems else 0.0
        parts.append(f"{tier}={c} ({pct:.0f}%)")
    extras = []
    if counts.get("non_theorem", 0):
        extras.append(f"non_theorem={counts['non_theorem']}")
    if counts.get("unknown", 0):
        extras.append(f"unknown={counts['unknown']}")
    suffix = ("  [excluded: " + ", ".join(extras) + "]") if extras else ""
    print(f"[tables] corpus difficulty (theorems only, n={n_theorems}): "
          + ", ".join(parts) + suffix)
    return 0


def _save(fig, name: str) -> None:
    path = RESULTS_FIGURES / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


_CFG_COLOURS = {
    "C0": "#1f77b4",
    "C1": "#ff7f0e",
    "C2": "#2ca02c",
    "C3": "#d62728",
    "C4": "#9467bd",
}


def _legend_outside(ax, title: str = "config") -> None:
    ax.legend(
        title=title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        borderaxespad=0.0,
    )


def figure_cactus(median: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for cfg in CONFIGS:
        sub = median[(median["config"] == cfg) & (median["status"] == "PROVED")]
        times = sorted(t for t in sub["wall_time_s"].tolist() if t > 0)
        if not times:
            continue
        ax.plot(times, range(1, len(times) + 1),
                label=cfg, drawstyle="steps-post",
                color=_CFG_COLOURS[cfg], linewidth=1.6)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("formulas solved (cumulative)")
    ax.set_title("Cactus plot: cumulative solved vs wall time")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    _legend_outside(ax)
    fig.tight_layout()
    _save(fig, "cactus")


_DATASET_COLOURS = {
    "ALTERNATION":  "#1f77b4",  # blue
    "DRINKER":      "#ff7f0e",  # orange
    "HOU":          "#2ca02c",
    "ILTP":         "#d62728",
    "INVALID_OPEN": "#7f7f7f",
    "PELLETIER":    "#9467bd",
    "PHP":          "#8c564b",
}

TIMEOUT_S = 30.0


def figure_scatter(median: pd.DataFrame, meta: pd.DataFrame) -> None:
    df = median.merge(meta[["label", "dataset"]], on="label", how="left")
    pivot_st = df.pivot(index=["label", "dataset"], columns="config", values="status").reset_index()
    pivot_wt = df.pivot(index=["label", "dataset"], columns="config", values="wall_time_s").reset_index()
    if "C0" not in pivot_st.columns or "C4" not in pivot_st.columns:
        return

    both, c4_only, c0_only = [], [], []
    for i, row in pivot_st.iterrows():
        ds = row["dataset"]
        c0_st, c4_st = row.get("C0"), row.get("C4")
        c0_wt = pivot_wt.iloc[i].get("C0")
        c4_wt = pivot_wt.iloc[i].get("C4")
        if c0_st == "PROVED" and c4_st == "PROVED":
            if c0_wt and c4_wt and c0_wt > 0 and c4_wt > 0:
                both.append((row["label"], ds, c0_wt, c4_wt))
        elif c0_st == "PROVED" and c4_st != "PROVED":
            if c0_wt and c0_wt > 0:
                c0_only.append((row["label"], ds, c0_wt))
        elif c4_st == "PROVED" and c0_st != "PROVED":
            if c4_wt and c4_wt > 0:
                c4_only.append((row["label"], ds, c4_wt))

    lo, hi = 1e-4, TIMEOUT_S * 2.0

    fig, ax = plt.subplots(figsize=(8, 7))

    seen_ds = set()
    for label, ds, x, y in both:
        c = _DATASET_COLOURS.get(ds, "black")
        ax.scatter(max(x, lo), max(y, lo), s=42,
                   color=c, edgecolor="black", linewidth=0.4,
                   label=ds if ds not in seen_ds else None,
                   alpha=0.8, zorder=3)
        seen_ds.add(ds)

    # right-censored: C0 PROVED, C4 hit timeout. drawn at (C0_wt, TIMEOUT).
    for label, ds, x in c0_only:
        c = _DATASET_COLOURS.get(ds, "black")
        ax.scatter(max(x, lo), TIMEOUT_S, s=70, marker="^",
                   color=c, edgecolor="black", linewidth=0.6,
                   label=ds if ds not in seen_ds else None,
                   alpha=0.95, zorder=4)
        seen_ds.add(ds)

    # right-censored the other way.
    for label, ds, y in c4_only:
        c = _DATASET_COLOURS.get(ds, "black")
        ax.scatter(TIMEOUT_S, max(y, lo), s=70, marker=">",
                   color=c, edgecolor="black", linewidth=0.6,
                   label=ds if ds not in seen_ds else None,
                   alpha=0.95, zorder=4)
        seen_ds.add(ds)

    xs = np.array([lo, hi])
    ax.plot(xs, xs, color="gray", linestyle="-", linewidth=1.0, alpha=0.7,
            label="y = x (parity)", zorder=1)
    ax.plot(xs, xs * 0.5, color="gray", linestyle="--", linewidth=0.8,
            alpha=0.5, label="y = x/2  (C4 2x faster)", zorder=1)
    ax.plot(xs, xs * 2.0, color="gray", linestyle="--", linewidth=0.8,
            alpha=0.5, label="y = 2x   (C4 2x slower)", zorder=1)
    ax.axhline(TIMEOUT_S, color="red", linestyle=":", linewidth=1.0,
               alpha=0.6, zorder=1)
    ax.axvline(TIMEOUT_S, color="red", linestyle=":", linewidth=1.0,
               alpha=0.6, zorder=1)
    ax.text(TIMEOUT_S * 1.05, lo * 2, f"{int(TIMEOUT_S)}s timeout",
            color="red", alpha=0.7, fontsize=8, rotation=90,
            verticalalignment="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("wall time C0 (baseline) [s]")
    ax.set_ylabel("wall time C4 (HFB+M+RG) [s]")
    ax.set_title(
        "Per-formula wall time: C0 vs C4\n"
        "(triangles = right-censored: pointed-at axis exceeded the 30s timeout)"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, which="both")
    _legend_outside(ax, title="dataset / reference")
    fig.tight_layout()
    _save(fig, "scatter")


def figure_ecdf(median: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for cfg in CONFIGS:
        sub = median[(median["config"] == cfg) & (median["status"] == "PROVED")]
        times = sorted(t for t in sub["wall_time_s"].tolist() if t > 0)
        if not times:
            continue
        y = np.arange(1, len(times) + 1) / len(times)
        ax.plot(times, y, label=cfg, drawstyle="steps-post",
                color=_CFG_COLOURS[cfg], linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("fraction of solved formulas (ECDF)")
    ax.set_title("Empirical CDF of wall time across configs")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    _legend_outside(ax)
    fig.tight_layout()
    _save(fig, "ecdf")


def figure_boxplot(median: pd.DataFrame, meta: pd.DataFrame) -> None:
    df = median.merge(meta[["label", "dataset"]], on="label")
    df = df[(df["status"] == "PROVED") & (df["wall_time_s"] > 0)]
    if df.empty:
        return
    datasets = [d for d in sorted(df["dataset"].unique())
                if not df[df["dataset"] == d].empty]
    n = len(datasets)
    if n == 0:
        return
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows),
                             squeeze=False, sharey=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // cols][idx % cols]
        per_cfg = []
        labels = []
        colours = []
        for cfg in CONFIGS:
            vals = df[(df["dataset"] == dataset) & (df["config"] == cfg)]["wall_time_s"].values
            if len(vals) > 0:
                per_cfg.append(vals)
                labels.append(cfg)
                colours.append(_CFG_COLOURS[cfg])
        if not per_cfg:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(per_cfg, positions=range(len(per_cfg)),
                        widths=0.6, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.4)
        ax.set_xticks(range(len(per_cfg)))
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_title(dataset, fontsize=10)
        ax.set_ylabel("wall time (s)")
        ax.grid(True, alpha=0.3, axis="y", which="both")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle("Per-category runtime distribution by config (PROVED only)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, "boxplot")


def cmd_figures(args=None) -> int:
    _ensure_dirs()
    df = _load_records()
    median = _median_per_cell(df)
    meta = _load_dataset_meta()
    figure_cactus(median)
    figure_scatter(median, meta)
    figure_ecdf(median)
    figure_boxplot(median, meta)
    print(f"[figures] wrote 4 SVGs to {RESULTS_FIGURES.relative_to(_REPO_ROOT)}")
    return 0


def _paired_values(median, meta, dataset_filter, cfg_x, cfg_y, value_col):
    df = median.merge(meta[["label", "dataset", "tags"]], on="label")
    if isinstance(dataset_filter, str):
        df = df[df["dataset"] == dataset_filter]
    elif callable(dataset_filter):
        df = df[df.apply(dataset_filter, axis=1)]
    pivot = df.pivot(index="label", columns="config", values=value_col)
    statuses = df.pivot(index="label", columns="config", values="status")
    if cfg_x not in pivot.columns or cfg_y not in pivot.columns:
        return [], []
    both_proved = (statuses[cfg_x] == "PROVED") & (statuses[cfg_y] == "PROVED")
    pairs = pivot.loc[both_proved, [cfg_x, cfg_y]].dropna()
    return pairs[cfg_x].tolist(), pairs[cfg_y].tolist()


def _solved_counts(median, meta, dataset_filter, cfg_x, cfg_y):
    df = median.merge(meta[["label", "dataset", "tags"]], on="label")
    if isinstance(dataset_filter, str):
        df = df[df["dataset"] == dataset_filter]
    elif callable(dataset_filter):
        df = df[df.apply(dataset_filter, axis=1)]
    statuses = df.pivot(index="label", columns="config", values="status")
    if cfg_x not in statuses.columns or cfg_y not in statuses.columns:
        return 0, 0
    only_x = ((statuses[cfg_x] == "PROVED") & (statuses[cfg_y] != "PROVED")).sum()
    only_y = ((statuses[cfg_x] != "PROVED") & (statuses[cfg_y] == "PROVED")).sum()
    return int(only_x), int(only_y)


def _format_hyp(label, claim, result, threshold) -> str:
    return (
        f"{label}: {claim}\n"
        f"  result    : {result}\n"
        f"  threshold : {threshold}\n"
    )


def cmd_stats(args=None) -> int:
    _ensure_dirs()
    df = _load_records()
    median = _median_per_cell(df)
    meta = _load_dataset_meta()
    lines = []
    lines.append("Hypothesis tests for HFB+M+RG\n")
    lines.append("=" * 60 + "\n\n")

    df_all = median.merge(meta[["label", "dataset"]], on="label")
    cats = sorted(df_all["dataset"].unique())

    # H1: whole corpus
    only_c0, only_c4 = _solved_counts(median, meta, lambda r: True, "C0", "C4")
    h1 = sign_test_one_sided(b=only_c4, c=only_c0)
    h1_pass = h1["p"] is not None and h1["p"] < 0.05
    lines.append(_format_hyp(
        "H1", "C4 strictly improves solved count over C0 (whole corpus)",
        f"discordant pairs: C4 wins={only_c4}, C0 wins={only_c0}, "
        f"n_disc={h1['n']}, exact one-sided p={h1['p']:.4f}",
        "exact one-sided p < 0.05  =>  " + ("PASS" if h1_pass else "FAIL"),
    ))

    # H2: per-category
    per_cat_counts = []
    all_geq = True
    for cat in cats:
        sub = df_all[df_all["dataset"] == cat]
        statuses = sub.pivot(index="label", columns="config", values="status")
        c0 = int((statuses.get("C0") == "PROVED").sum()) if "C0" in statuses else 0
        c4 = int((statuses.get("C4") == "PROVED").sum()) if "C4" in statuses else 0
        per_cat_counts.append(f"{cat}: C0={c0} C4={c4}{' (regression)' if c4 < c0 else ''}")
        if c4 < c0:
            all_geq = False
    lines.append(_format_hyp(
        "H2", "C4 does not regress in any category",
        "; ".join(per_cat_counts),
        "s_C4(c) >= s_C0(c) for all c  =>  " + ("PASS" if all_geq else "FAIL"),
    ))

    # H3: RG isolation, C3 vs C1 on ALTERNATION + HOU
    targeted = ("ALTERNATION", "HOU")
    deltas = {}
    h3_b_total = 0
    h3_c_total = 0
    for cat in targeted:
        sub = df_all[df_all["dataset"] == cat]
        statuses = sub.pivot(index="label", columns="config", values="status")
        if "C1" not in statuses or "C3" not in statuses:
            deltas[cat] = ("missing", 0, 0)
            continue
        c1 = int((statuses["C1"] == "PROVED").sum())
        c3 = int((statuses["C3"] == "PROVED").sum())
        only_c1 = int(((statuses["C1"] == "PROVED") & (statuses["C3"] != "PROVED")).sum())
        only_c3 = int(((statuses["C1"] != "PROVED") & (statuses["C3"] == "PROVED")).sum())
        deltas[cat] = (c3 - c1, only_c1, only_c3)
        h3_b_total += only_c3
        h3_c_total += only_c1
    n_cats_with_gain = sum(1 for cat in targeted
                           if isinstance(deltas[cat][0], int) and deltas[cat][0] >= 1)
    mc_h3 = sign_test_one_sided(b=h3_b_total, c=h3_c_total)
    h3_pass = n_cats_with_gain >= 2
    lines.append(_format_hyp(
        "H3", "RG closes formulas B alone cannot (C3 vs C1 on ALTERNATION + HOU)",
        f"per-category delta(C3-C1): " + "; ".join(
            f"{cat}: delta={deltas[cat][0]}" for cat in targeted)
        + f"; aggregate McNemar one-sided p={mc_h3['p']:.4f} (n_disc={mc_h3['n']}, descriptive)",
        f"strict gain in >= 2 of {len(targeted)} categories  =>  "
        + ("PASS" if h3_pass else "FAIL"),
    ))

    # H4: memo overhead on HOU, C2 vs C1
    x, y = _paired_values(median, meta, "HOU", "C1", "C2", "wall_time_s")
    r4 = median_ratio(x, y)
    wil4 = wilcoxon_log_ratio(x, y)
    h4_pass = (r4 is not None) and (r4 <= 1.5)
    lines.append(_format_hyp(
        "H4", "M overhead bounded on HOU (median wall-time C2/C1 <= 1.5)",
        f"median_ratio={r4}, n={len(x)}, Wilcoxon p={wil4['p']}, note={wil4['note']}",
        "median ratio <= 1.5  =>  " + ("PASS" if h4_pass else "FAIL"),
    ))

    # H5: per-category McNemar
    per_cat_mc = []
    any_significant = False
    for cat in cats:
        sub = df_all[df_all["dataset"] == cat]
        statuses = sub.pivot(index="label", columns="config", values="status")
        if "C0" not in statuses or "C4" not in statuses:
            continue
        only_c0 = int(((statuses["C0"] == "PROVED") & (statuses["C4"] != "PROVED")).sum())
        only_c4 = int(((statuses["C0"] != "PROVED") & (statuses["C4"] == "PROVED")).sum())
        mc = mcnemar_exact(only_c0, only_c4)
        if mc["p"] is not None and mc["p"] < 0.05:
            any_significant = True
        per_cat_mc.append(
            f"{cat}: only_C0={only_c0} only_C4={only_c4} p={mc['p']:.3f}"
        )
    h5_pass = any_significant
    lines.append(_format_hyp(
        "H5", "Per-category statistical significance reached (McNemar exact, two-sided)",
        "; ".join(per_cat_mc),
        "p < 0.05 in >= 1 category  =>  "
        + ("PASS" if h5_pass else "FAIL  (future work: scale corpus to >= 200 formulas)"),
    ))

    # H6: runtime parity per category
    per_cat_ratio = []
    cats_below_1 = 0
    cats_eligible = 0
    for cat in cats:
        sub = df_all[df_all["dataset"] == cat]
        statuses = sub.pivot(index="label", columns="config", values="status")
        times = sub.pivot(index="label", columns="config", values="wall_time_s")
        if "C0" not in statuses or "C4" not in statuses:
            continue
        both = (statuses["C0"] == "PROVED") & (statuses["C4"] == "PROVED")
        x = times.loc[both, "C0"].tolist()
        y = times.loc[both, "C4"].tolist()
        r = median_ratio(x, y)
        if r is None:
            continue
        cats_eligible += 1
        wil = wilcoxon_log_ratio(x, y)
        sig = (wil["p"] is not None and wil["p"] < 0.05)
        if r < 1.0 and sig:
            cats_below_1 += 1
        per_cat_ratio.append(
            f"{cat}: r={r:.2f} (n={wil['n']}, Wilcoxon p={wil['p']})"
        )
    threshold = max(4, (cats_eligible + 1) // 2)
    h6_pass = cats_below_1 >= threshold
    lines.append(_format_hyp(
        "H6", "C4 reaches per-node runtime parity with C0 on solved-by-both",
        "; ".join(per_cat_ratio)
        + f"; cats_with_r<1_and_sig={cats_below_1}/{cats_eligible}",
        f"r(c) < 1 with Wilcoxon p < 0.05 in >= {threshold} of {cats_eligible} "
        f"non-empty categories  =>  "
        + ("PASS" if h6_pass else "FAIL  (future work: per-node Python-overhead reduction)"),
    ))

    RESULTS_SUMMARY.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"[stats] wrote {RESULTS_SUMMARY.relative_to(_REPO_ROOT)}")
    return 0


def cmd_all(args=None) -> int:
    rc = cmd_tables()
    if rc != 0:
        return rc
    rc = cmd_figures()
    if rc != 0:
        return rc
    return cmd_stats()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse the experiment harness output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("tables", cmd_tables), ("figures", cmd_figures),
                     ("stats", cmd_stats), ("all", cmd_all)]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
