"""Merge per-category bundles into data/dataset.jsonl. Idempotent."""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATA_DIR = _REPO_ROOT / "data"


GENERATED_CATEGORIES = (
    ("php", "PHP", "PROP_BRANCH"),
    ("drinker", "DRINKER", "FOL_QHEAVY"),
    ("hou", "HOU", "HOU"),
    ("parametric", "ALTERNATION", "FOL_QHEAVY"),
    ("invalid", "INVALID_OPEN", "INVALID"),
)

CONVERTED_SOURCES = (
    ("pelletier", "PELLETIER", "PELLETIER"),
    ("iltp", "ILTP", "ILTP"),
)


def _read_tsv(path: Path) -> tuple:
    if not path.exists():
        return ([], [])
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines:
        return ([], [])
    header = lines[0].split("\t")
    rows = [dict(zip(header, ln.split("\t"))) for ln in lines[1:] if ln]
    return (header, rows)


def _read_fol(path: Path) -> list:
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="ascii").splitlines() if ln.strip()]


def _normalise_tags(raw: str, default: str) -> list:
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    if default and default not in tags:
        tags.append(default)
    return tags


def _normalise_status(raw: str) -> str:
    s = raw.strip().upper()
    if s == "THEOREM":
        return "THEOREM"
    if s in {"COUNTERSAT", "COUNTERSATISFIABLE", "COUNTERSATISFIABLE_FOF"}:
        return "COUNTERSAT"
    return s or "UNKNOWN"


def _category_records(name: str, dataset_label: str, default_tag: str) -> list:
    fol_path = DATA_DIR / name / f"{name}.fol"
    manifest_path = DATA_DIR / name / "manifest.tsv"
    formulas = _read_fol(fol_path)
    _, rows = _read_tsv(manifest_path)
    if len(formulas) != len(rows):
        raise ValueError(
            f"{name}: formula count {len(formulas)} != manifest rows {len(rows)}"
        )
    out = []
    for formula, row in zip(formulas, rows):
        out.append({
            "label": row.get("label", "").strip(),
            "formula": formula,
            "dataset": dataset_label,
            "tags": _normalise_tags(row.get("tags", ""), default_tag),
            "status_label": _normalise_status(row.get("status", "")),
            "source": row.get("source", "").strip(),
        })
    return out


def _converted_records(source: str, dataset_label: str, default_tag: str) -> list:
    fol_path = DATA_DIR / "converted" / f"{source}.fol"
    tsv_path = DATA_DIR / "converted" / f"{source}.tsv"
    formulas = _read_fol(fol_path)
    _, rows = _read_tsv(tsv_path)
    if len(formulas) != len(rows):
        raise ValueError(
            f"{source}: formula count {len(formulas)} != manifest rows {len(rows)}"
        )
    out = []
    for formula, row in zip(formulas, rows):
        out.append({
            "label": row.get("label", "").strip(),
            "formula": formula,
            "dataset": dataset_label,
            "tags": _normalise_tags(row.get("tags", ""), default_tag),
            "status_label": _normalise_status(row.get("status", "")),
            "source": row.get("source", "").strip(),
        })
    return out


def write_text_ascii(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(content)


def main(do_check: bool) -> int:
    records = []
    for name, dataset_label, default_tag in GENERATED_CATEGORIES:
        records.extend(_category_records(name, dataset_label, default_tag))
    for source, dataset_label, default_tag in CONVERTED_SOURCES:
        records.extend(_converted_records(source, dataset_label, default_tag))

    by_dataset = {}
    for r in records:
        by_dataset[r["dataset"]] = by_dataset.get(r["dataset"], 0) + 1
    print(f"[total]  {len(records)} records", flush=True)
    for k in sorted(by_dataset):
        print(f"[{k:11s}] {by_dataset[k]}", flush=True)

    if do_check:
        return 0

    jsonl_lines = [json.dumps(r, ensure_ascii=True) for r in records]
    write_text_ascii(DATA_DIR / "dataset.jsonl", "\n".join(jsonl_lines) + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the combined dataset.")
    p.add_argument("--check", action="store_true",
                   help="report counts without writing files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args.check))
