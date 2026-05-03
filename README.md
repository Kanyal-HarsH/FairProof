# FairProof (HFB+M+RG)

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Tests](https://img.shields.io/badge/tests-222%20passing-brightgreen)
![Corpus](https://img.shields.io/badge/corpus-94%20theorems-blue)
![Reps](https://img.shields.io/badge/reps-5-blue)
![Hypotheses](https://img.shields.io/badge/H1...H6-5%20pass%20%2F%201%20fail-yellowgreen)
![License](https://img.shields.io/badge/license-academic-lightgrey)

A first-order logic theorem prover. Two engines share the same rule
machinery: the **baseline** (Algorithm 2) and an improved variant
called **FairProof (HFB+M+RG)**.

The improved engine adds three independent components. **HFB** runs
iterative deepening over a node budget with a round-robin counter
that prevents term starvation. **M** caches sequents by fingerprint,
with cycle detection on `IN_PROGRESS` entries; failed lookups are
never stored. **R** ranks candidate terms by predicate overlap, term
reuse, and a depth penalty. The four ablation cells C1 through C4
all share the same search driver.

## Table of contents

1. [Quick start](#quick-start)
2. [Configurations](#configurations)
3. [Corpus](#corpus)
4. [Results](#results)
5. [Reproducing the canonical battery](#reproducing-the-canonical-battery)
6. [Reading the output](#reading-the-output)
7. [Repository layout](#repository-layout)
8. [Reference](#reference)

## Quick start

```
pip install -r requirements.txt
pytest -q
```

The test suite returns `222 passed` in roughly two seconds.

To prove a single formula:

```
python experiments/run_experiments.py one --label hou_ex19 --config C4 --node-budget 200000 --max-depth 200 --timeout-s 30.0
```

## Configurations

| Config | HFB | M | RG | Description |
|---|---|---|---|---|
| C0 | no  | no  | no  | baseline (Algorithm 2) |
| C1 | yes | no  | no  | iterative deepening only |
| C2 | yes | yes | no  | C1 plus memoisation |
| C3 | yes | no  | yes | C1 plus relevance |
| C4 | yes | yes | yes | full FairProof (HFB+M+RG) |

C1 through C4 are produced by `prover.config.config_for(label)`.

## Corpus

110 records in `data/dataset.jsonl`: 94 theorems and 16 non-theorems.

| Category | Theorems | Non-theorems | Source |
|---|---:|---:|---|
| ALTERNATION | 6 | 4 | parametric quantifier-alternation generator |
| DRINKER | 5 | 0 | Smullyan 1968 plus four variants |
| HOU | 16 | 0 | textbook examples from Hou 2021 |
| ILTP | 5 | 0 | classical-valid subset of ILTP 1.1.2 |
| INVALID_OPEN | 0 | 12 | hand-constructed invalid forms |
| PELLETIER | 57 | 0 | TPTP SYN files mapped to Pelletier 1..46 plus screened additions |
| PHP | 5 | 0 | Cook-Reckhow 1979 propositional pigeonhole |

Per-formula difficulty:

| Tier | Definition | Count |
|---|---|---:|
| easy | C0 PROVED in under 10 ms | 35 |
| medium | C0 PROVED in 10 ms to 30 s | 15 |
| hard | C0 fails, but at least one of C1 to C4 PROVED | 12 |
| challenging | no configuration PROVED within budget | 32 |

The hard tier is the band where the contribution lives: the baseline
cannot close those formulas within budget, but FairProof (HFB+M+RG)
can.

## Results

Solved counts per configuration, from `results/tables/table3_ablation.csv`:

| Category | C0 | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|---:|
| ALTERNATION | 5 | 4 | 4 | 6 | 6 |
| DRINKER | 5 | 5 | 5 | 5 | 5 |
| HOU | 14 | 15 | 15 | 16 | 16 |
| ILTP | 0 | 0 | 0 | 0 | 0 |
| PELLETIER | 25 | 33 | 33 | 33 | 33 |
| PHP | 1 | 1 | 2 | 1 | 2 |
| **Total** | **50** | **58** | **59** | **61** | **62** |

C4 closes 12 more theorems than C0, a 24 percent gain at one-sided
sign-test p = 0.0002.

Hypothesis verdicts from `results/summary.txt`:

| ID | Claim | Test | Verdict |
|---|---|---|---|
| H1 | C4 strictly improves solved count over C0 | one-sided sign test | **PASS** (p = 0.0002) |
| H2 | C4 does not regress on any category | per-category count | **PASS** |
| H3 | RG closes formulas B alone cannot, on ALTERNATION + HOU | C3 vs C1 count | **PASS** |
| H4 | Memo overhead bounded on HOU | Wilcoxon on log ratio | **PASS** (1.06x) |
| H5 | Per-category statistical significance reached | McNemar exact | **PASS** (PELLETIER p = 0.008) |
| H6 | Per-node runtime parity | Wilcoxon on log ratio | FAIL |

H6 reflects per-node Python overhead in the cache and the relevance
score; it is documented as engineering future work.

## Reproducing the canonical battery

```
PYTHONHASHSEED=0 python experiments/run_experiments.py full --reps 5 --node-budget 200000 --max-depth 200 --timeout-s 30.0 --workers 16
python experiments/analyse.py all
```

The first command runs 110 records by five configurations by five
reps on sixteen worker processes. On a recent laptop this finishes
in about twenty-five minutes. The second command writes six CSV
tables, four SVG figures, and `results/summary.txt`.

`PYTHONHASHSEED=0` must be exported before the harness starts. The
baseline picks principals via a deterministic lex tiebreak, but
worker subprocesses inherit fresh hash seeds by default; fixing the
seed makes reruns byte-stable on the same machine.

## Reading the output

```
results/
  summary.txt                     H1..H6 verdicts with test statistics
  tables/
    table1_dataset_summary.csv    per-formula difficulty tier counts
    table2_baseline_vs_full.csv   C0 vs C4 with Wilcoxon and McNemar
    table3_ablation.csv           solved counts per (category, config)
    table4_hard_only.csv          per-formula breakdown of the hard tier
    table5_memoisation_stats.csv  cache hit rate per (config, category)
    table6_status_matrix.csv      full label-by-config status matrix
  figures/
    cactus.svg                    cumulative solved against wall time
    scatter.svg                   per-formula C0 vs C4 with right censoring
    ecdf.svg                      ECDF of wall time across configs
    boxplot.svg                   per-category runtime distribution
```

## Repository layout

```
logic/        AST, parser, pretty printer, substitution, unification
prover/       baseline + FairProof (HFB+M+RG) drivers, memo, relevance, rules
experiments/  harness, analyser, dataset builders, statistics
tests/        222 tests across every module
data/         dataset.jsonl plus per-source manifests
results/      tables and figures from the canonical battery
```

## Reference

Hou, Z. *Fundamentals of Logic and Computation: With Practical
Automated Reasoning and Verification*. Springer Texts in Computer
Science, 2021. ISBN 978-3-030-87881-8. Algorithm 2 is on page 67.
