"""Stage the eight per-category data folders. Idempotent.

python experiments/download_data.py [--dry-run]
"""

import argparse
import hashlib
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "_cache"

TPTP_VERSION = "9.2.1"
TPTP_URL = f"https://tptp.org/TPTP/Distribution/TPTP-v{TPTP_VERSION}.tgz"
TPTP_URL_FALLBACK = f"http://tptp.org/TPTP/Distribution/TPTP-v{TPTP_VERSION}.tgz"
TPTP_ARCHIVE_NAME = f"TPTP-v{TPTP_VERSION}.tgz"

ILTP_VERSION = "1.1.2"
ILTP_URL = f"https://www.iltp.de/download/ILTP-v{ILTP_VERSION}-firstorder.tar.gz"
ILTP_URL_FALLBACK = f"http://www.iltp.de/download/ILTP-v{ILTP_VERSION}-firstorder.tar.gz"
ILTP_ARCHIVE_NAME = f"ILTP-v{ILTP_VERSION}-firstorder.tar.gz"

ARITHMETIC_TOKENS = (
    "$less", "$lesseq", "$greater", "$greatereq",
    "$sum", "$difference", "$product", "$quotient",
    "$real", "$rat", "$int", "$distinct",
)

PELLETIER_RANGE = set(range(1, 47))

CATEGORY_TARGETS = {
    "pelletier": 40,
    "tptp_syn_release": 30,
    "iltp": 5,
    "php": 5,
    "drinker": 5,
    "hou": 8,
    "parametric": 10,
    "invalid": 12,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_ascii(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(content)


def write_tsv(path: Path, header: list, rows: list) -> None:
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in header))
    write_text_ascii(path, "\n".join(lines) + "\n")


# networking

def _download(url: str, target_tmp: Path) -> None:
    chunk_size = 1 << 20
    progress_step = 64 << 20
    next_log = progress_step
    written = 0
    with urllib.request.urlopen(url, timeout=180) as resp, open(target_tmp, "wb") as out:
        total = resp.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        if total:
            log(f"[net]   size {total // 1_048_576} MB")
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            if written >= next_log:
                if total:
                    log(f"[net]   {written // 1_048_576} / {total // 1_048_576} MB")
                else:
                    log(f"[net]   {written // 1_048_576} MB")
                next_log += progress_step


def download_to_cache(primary_url: str, fallback_url: str, archive_name: str,
                      dry_run: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / archive_name
    if target.exists() and target.stat().st_size > 0:
        log(f"[cache] {archive_name} present ({target.stat().st_size // 1_048_576} MB)")
        return target
    if dry_run:
        log(f"[dry]   would fetch {primary_url}")
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    last_err = None
    for url in (primary_url, fallback_url):
        if url is None or url == "":
            continue
        log(f"[net]   fetching {url}")
        try:
            _download(url, tmp)
            tmp.replace(target)
            log(f"[ok]    {archive_name} ({target.stat().st_size // 1_048_576} MB)")
            return target
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            if tmp.exists():
                tmp.unlink()
            log(f"[retry] {url} failed: {e}")
    raise RuntimeError(f"failed to download {archive_name}: {last_err}")


# TPTP processing

SYN_FILE_RE = re.compile(r"(?:^|/)SYN(\d+)\+1\.p$")
HEADER_NAMES_RE = re.compile(
    r"^%\s*Names\s*:\s*(.*?)(?=^%\s*[A-Z]\S*\s*:|\Z)",
    re.MULTILINE | re.DOTALL,
)
HEADER_STATUS_RE = re.compile(r"^%\s*Status\s*:\s*(\S+)", re.MULTILINE)
HEADER_SPC_RE = re.compile(r"^%\s*SPC\s*:\s*(\S+)", re.MULTILINE)
HEADER_RATING_RE = re.compile(r"^%\s*Rating\s*:\s*([\d.]+)", re.MULTILINE)
PELLETIER_NUM_RE = re.compile(r"Pelletier\s+(\d+)\b", re.IGNORECASE)


def extract_syn_files(archive: Path, out_dir: Path, dry_run: bool = False) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    if dry_run:
        log(f"[dry]   would scan {archive.name} for SYN*+1.p files")
        return extracted
    log(f"[scan]  reading {archive.name}")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if not SYN_FILE_RE.search(member.name):
                continue
            target = out_dir / Path(member.name).name
            extracted.append(target)
            if target.exists() and target.stat().st_size > 0:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    log(f"[scan]  {len(extracted)} SYN+1 files staged in {out_dir.relative_to(REPO_ROOT)}")
    return extracted


def parse_tptp_header(path: Path) -> dict:
    text = path.read_text(encoding="latin-1", errors="replace")
    head = text[:8000]
    out = {}
    m = HEADER_STATUS_RE.search(head)
    out["status"] = m.group(1).strip() if m else ""
    m = HEADER_SPC_RE.search(head)
    out["spc"] = m.group(1).strip() if m else ""
    m = HEADER_RATING_RE.search(head)
    try:
        out["rating"] = float(m.group(1)) if m else float("nan")
    except ValueError:
        out["rating"] = float("nan")
    m = HEADER_NAMES_RE.search(head)
    out["names"] = (m.group(1) if m else "").strip()
    out["body"] = text
    return out


def has_arithmetic(body: str) -> bool:
    return any(tok in body for tok in ARITHMETIC_TOKENS)


def has_equality(body: str) -> bool:
    code_lines = [ln for ln in body.splitlines() if not ln.lstrip().startswith("%")]
    code = "\n".join(code_lines)
    if re.search(r"\bequal\s*\(", code):
        return True
    if re.search(r"(?<![<>!=])=(?![=>])", code):
        return True
    if "!=" in code:
        return True
    return False


def select_pelletier(syn_dir: Path) -> dict:
    keep = {}
    for path in sorted(syn_dir.glob("SYN*+1.p")):
        info = parse_tptp_header(path)
        if info["status"] != "Theorem":
            continue
        m = PELLETIER_NUM_RE.search(info["names"])
        if not m:
            continue
        pel = int(m.group(1))
        if pel not in PELLETIER_RANGE:
            continue
        if has_arithmetic(info["body"]) or has_equality(info["body"]):
            continue
        if any(d["pelletier"] == pel for d in keep.values()):
            continue
        keep[path] = {
            "pelletier": pel,
            "status": info["status"],
            "spc": info["spc"],
            "rating": info["rating"],
        }
    return keep


def select_tptp_syn_release(syn_dir: Path, exclude: set, target: int) -> dict:
    candidates = []
    for path in sorted(syn_dir.glob("SYN*+1.p")):
        if path in exclude:
            continue
        info = parse_tptp_header(path)
        if info["spc"] != "FOF_THM_RFO_NEQ":
            continue
        rating = info["rating"]
        if rating != rating or not (0.10 <= rating <= 0.50):
            continue
        if has_arithmetic(info["body"]) or has_equality(info["body"]):
            continue
        candidates.append((path, info))
    candidates.sort(key=lambda kv: (-kv[1]["rating"], kv[0].name))
    keep = {}
    for path, info in candidates[:target]:
        keep[path] = {
            "pelletier": -1,
            "status": info["status"],
            "spc": info["spc"],
            "rating": info["rating"],
        }
    return keep


def remove_syn_files_outside(out_dir: Path, kept: set) -> int:
    n = 0
    for path in out_dir.glob("SYN*+1.p"):
        if path not in kept:
            path.unlink()
            n += 1
    return n


def write_pelletier_names_tsv(out_dir: Path, pelletier: dict, release: dict) -> None:
    rows = []
    for path, d in sorted(pelletier.items()):
        rows.append({
            "syn_file": path.name,
            "origin": "PELLETIER",
            "pelletier": d["pelletier"],
            "status": d["status"],
            "spc": d["spc"],
            "rating": f"{d['rating']:.2f}" if d["rating"] == d["rating"] else "",
        })
    for path, d in sorted(release.items()):
        rows.append({
            "syn_file": path.name,
            "origin": "TPTP_SYN_RELEASE",
            "pelletier": "-",
            "status": d["status"],
            "spc": d["spc"],
            "rating": f"{d['rating']:.2f}" if d["rating"] == d["rating"] else "",
        })
    write_tsv(
        out_dir / "Names.tsv",
        ["syn_file", "origin", "pelletier", "status", "spc", "rating"],
        rows,
    )


# ILTP processing

ILTP_INTUIT_RE = re.compile(
    r"^%\s*Status\s*\(\s*intuit\.?\s*\)\s*:\s*(\S+)",
    re.IGNORECASE | re.MULTILINE,
)
ILTP_CLASSICAL_RE = re.compile(
    r"^%\s*Status\s*:\s*(\S+)",
    re.MULTILINE,
)
ILTP_INCLUDE_RE = re.compile(r"^\s*include\s*\(", re.MULTILINE)


def select_iltp(archive: Path, out_dir: Path, target: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {}
    if not archive.exists():
        log("[warn]  ILTP archive missing; skipping")
        return keep
    try:
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                if len(keep) >= target:
                    break
                if not member.isfile() or not member.name.endswith(".p"):
                    continue
                src = tar.extractfile(member)
                if src is None:
                    continue
                full = src.read().decode("latin-1", errors="replace")
                head = full[:8000]
                m_cls = ILTP_CLASSICAL_RE.search(head)
                if not m_cls or m_cls.group(1).strip().lower() != "theorem":
                    continue
                if ILTP_INCLUDE_RE.search(full):
                    continue
                if has_arithmetic(full) or has_equality(full):
                    continue
                target_path = out_dir / Path(member.name).name
                target_path.write_bytes(full.encode("latin-1", errors="replace"))
                m_int = ILTP_INTUIT_RE.search(head)
                keep[target_path] = {
                    "classical": m_cls.group(1).strip(),
                    "intuit": m_int.group(1).strip() if m_int else "",
                }
    except (tarfile.TarError, OSError) as e:
        log(f"[warn]  ILTP extraction failed: {e}")
    return keep


def write_iltp_selected_tsv(out_dir: Path, kept: dict) -> None:
    rows = []
    for path, d in sorted(kept.items()):
        rows.append({
            "iltp_file": path.name,
            "classical_status": d["classical"],
            "intuit_status": d.get("intuit", ""),
        })
    write_tsv(
        out_dir / "selected.tsv",
        ["iltp_file", "classical_status", "intuit_status"],
        rows,
    )


# generated formulas

def php_formula(n: int) -> str:
    pigeons = list(range(1, n + 2))
    holes = list(range(1, n + 1))

    def atom(i: int, j: int) -> str:
        return f"p_{i}_{j}"

    pigeon_clauses = []
    for i in pigeons:
        terms = [atom(i, j) for j in holes]
        pigeon_clauses.append("(" + " \\/ ".join(terms) + ")")
    antecedent = "(" + " /\\ ".join(pigeon_clauses) + ")"

    collisions = []
    for j in holes:
        for idx, i in enumerate(pigeons):
            for k in pigeons[idx + 1:]:
                collisions.append(f"({atom(i, j)} /\\ {atom(k, j)})")
    succedent = "(" + " \\/ ".join(collisions) + ")"
    return f"({antecedent} -> {succedent})"


PHP_RANGE = (2, 3, 4, 5, 6)


DRINKER_ENTRIES = (
    ("drinker_canonical.fol", "drinker_canonical",
     "(exists x. (D(x) -> (forall y. D(y))))"),
    ("drinker_dual.fol", "drinker_dual",
     "(exists x. ((forall y. D(y)) -> D(x)))"),
    ("drinker_var1.fol", "drinker_var1",
     "(exists x. (D(x) -> (forall y. (D(y) \\/ E(y)))))"),
    ("drinker_var2.fol", "drinker_var2",
     "(exists x. ((D(x) /\\ E(x)) -> (forall y. (D(y) /\\ E(y)))))"),
    ("drinker_var3.fol", "drinker_var3",
     "(exists x. (D(x) -> (forall y. (E(y) -> D(y)))))"),
)


HOU_ENTRIES = (
    ("hou_ex01.fol", "hou_ex01",
     "((P -> Q) -> ((~Q) -> (~P)))"),
    ("hou_ex02.fol", "hou_ex02",
     "((forall x. P(x)) -> P(a))"),
    ("hou_ex03.fol", "hou_ex03",
     "(P(a) -> (exists x. P(x)))"),
    ("hou_ex04.fol", "hou_ex04",
     "(((P -> Q) -> P) -> P)"),
    ("hou_ex05.fol", "hou_ex05",
     "(~(P /\\ (~P)))"),
    ("hou_ex06.fol", "hou_ex06",
     "((P /\\ Q) -> (Q /\\ P))"),
    ("hou_ex07.fol", "hou_ex07",
     "(((forall x. (P(x) -> Q(x))) /\\ (forall x. P(x))) -> (forall x. Q(x)))"),
    ("hou_ex08.fol", "hou_ex08",
     "((forall x. (forall y. R(x, y))) -> (forall y. (forall x. R(x, y))))"),
)


PARAMETRIC_ENTRIES = (
    ("F_01.fol", "param_01", "THEOREM",
     "((forall x. (exists y. P(x, y))) -> (forall x. (exists y. P(x, y))))"),
    ("F_02.fol", "param_02", "COUNTERSAT",
     "((forall x. (exists y. P(x, y))) -> (exists y. (forall x. P(x, y))))"),
    ("F_03.fol", "param_03", "THEOREM",
     "((exists x. (forall y. P(x, y))) -> (forall y. (exists x. P(x, y))))"),
    ("F_04.fol", "param_04", "COUNTERSAT",
     "((forall x. (forall y. (exists z. R(x, y, z)))) -> (exists z. (forall x. (forall y. R(x, y, z)))))"),
    ("F_05.fol", "param_05", "THEOREM",
     "((forall x. (forall y. P(x, y))) -> (forall y. (forall x. P(x, y))))"),
    ("F_06.fol", "param_06", "THEOREM",
     "((exists x. (exists y. P(x, y))) -> (exists y. (exists x. P(x, y))))"),
    ("F_07.fol", "param_07", "THEOREM",
     "((forall x. P(x)) -> (exists x. P(x)))"),
    ("F_08.fol", "param_08", "THEOREM",
     "((forall x. (P(x) -> Q(x))) -> ((forall x. P(x)) -> (forall x. Q(x))))"),
    ("F_09.fol", "param_09", "COUNTERSAT",
     "(((forall x. P(x)) -> (forall x. Q(x))) -> (forall x. (P(x) -> Q(x))))"),
    ("F_10.fol", "param_10", "COUNTERSAT",
     "((forall x. (forall y. (forall z. ((R(x, y) /\\ R(y, z)) -> R(x, z))))) -> (forall x. R(x, x)))"),
)


INVALID_ENTRIES = (
    ("inv_01.fol", "inv_01",
     "((forall x. (P(x) \\/ Q(x))) -> ((forall x. P(x)) \\/ (forall x. Q(x))))"),
    ("inv_02.fol", "inv_02",
     "((forall x. (exists y. R(x, y))) -> (exists y. (forall x. R(x, y))))"),
    ("inv_03.fol", "inv_03",
     "((exists x. P(x)) -> (forall x. P(x)))"),
    ("inv_04.fol", "inv_04",
     "(((forall x. P(x)) -> Q) -> (forall x. (P(x) -> Q)))"),
    ("inv_05.fol", "inv_05",
     "((exists x. P(x)) -> (exists x. (P(x) /\\ Q(x))))"),
    ("inv_06.fol", "inv_06",
     "(P(a) -> (forall x. P(x)))"),
    ("inv_07.fol", "inv_07",
     "((forall x. (P(x) -> Q(x))) -> (forall x. P(x)))"),
    ("inv_08.fol", "inv_08",
     "((P -> Q) -> (Q -> P))"),
    ("inv_09.fol", "inv_09",
     "((P \\/ Q) -> (P /\\ Q))"),
    ("inv_10.fol", "inv_10",
     "((forall x. (P(x) <-> Q(x))) -> (P(a) <-> Q(b)))"),
    ("inv_11.fol", "inv_11",
     "(((forall x. (P(x) -> Q(x))) /\\ Q(a)) -> P(a))"),
    ("inv_12.fol", "inv_12",
     "((forall x. (exists y. (R(x, y) /\\ R(y, x)))) -> (exists x. R(x, x)))"),
)


def write_category_bundle(out_dir: Path, category_name: str,
                          formulas: list, rows: list) -> int:
    """Write all formulas of a category into one .fol (one line per formula)
    plus a parallel manifest.tsv whose row order matches the .fol line order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_fol = f"{category_name}.fol"
    for stale in out_dir.glob("*.fol"):
        if stale.name != keep_fol:
            stale.unlink()
    write_text_ascii(out_dir / keep_fol, "\n".join(formulas) + "\n")
    write_tsv(
        out_dir / "manifest.tsv",
        ["line", "label", "status", "tags", "source"],
        rows,
    )
    return len(formulas)


def write_php(out_dir: Path) -> int:
    formulas = []
    rows = []
    for line, n in enumerate(PHP_RANGE, start=1):
        formulas.append(php_formula(n))
        rows.append({
            "line": line,
            "label": f"php_{n}",
            "status": "THEOREM",
            "tags": "PROP_BRANCH,PHP",
            "source": "Cook-Reckhow 1979",
        })
    return write_category_bundle(out_dir, "php", formulas, rows)


def write_drinker(out_dir: Path) -> int:
    formulas = []
    rows = []
    for line, (_, label, formula) in enumerate(DRINKER_ENTRIES, start=1):
        formulas.append(formula)
        rows.append({
            "line": line,
            "label": label,
            "status": "THEOREM",
            "tags": "DRINKER,FOL_QHEAVY",
            "source": "Smullyan 1968 and variants",
        })
    return write_category_bundle(out_dir, "drinker", formulas, rows)


def write_hou(out_dir: Path) -> int:
    formulas = []
    rows = []
    for line, (_, label, formula) in enumerate(HOU_ENTRIES, start=1):
        formulas.append(formula)
        rows.append({
            "line": line,
            "label": label,
            "status": "THEOREM",
            "tags": "HOU",
            "source": "Hou 2021 textbook (placeholder; refine after reading)",
        })
    return write_category_bundle(out_dir, "hou", formulas, rows)


def write_parametric(out_dir: Path) -> int:
    formulas = []
    rows = []
    for line, (_, label, status, formula) in enumerate(PARAMETRIC_ENTRIES, start=1):
        formulas.append(formula)
        rows.append({
            "line": line,
            "label": label,
            "status": status,
            "tags": "ALTERNATION,FOL_QHEAVY",
            "source": "parametric quantifier-alternation generator",
        })
    return write_category_bundle(out_dir, "parametric", formulas, rows)


def write_invalid(out_dir: Path) -> int:
    formulas = []
    rows = []
    for line, (_, label, formula) in enumerate(INVALID_ENTRIES, start=1):
        formulas.append(formula)
        rows.append({
            "line": line,
            "label": label,
            "status": "COUNTERSAT",
            "tags": "INVALID,INVALID_OPEN",
            "source": "hand-crafted",
        })
    return write_category_bundle(out_dir, "invalid", formulas, rows)


# top-level

def write_sources_tsv(rows: list) -> None:
    write_tsv(
        DATA_DIR / "SOURCES.tsv",
        ["source", "version", "url", "archive", "sha256", "fetched_at"],
        rows,
    )


def main(dry_run: bool = False) -> int:
    started = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sources = []

    pelletier_dir = DATA_DIR / "pelletier"
    iltp_dir = DATA_DIR / "iltp"

    log(f"[plan]  TPTP v{TPTP_VERSION}, ILTP v{ILTP_VERSION}, dry_run={dry_run}")
    log(f"[plan]  cache at {CACHE_DIR.relative_to(REPO_ROOT)}")

    # Stage 1: TPTP archive -> SYN files.
    try:
        tptp_archive = download_to_cache(TPTP_URL, TPTP_URL_FALLBACK,
                                         TPTP_ARCHIVE_NAME, dry_run=dry_run)
    except RuntimeError as e:
        log(f"[fatal] {e}")
        return 2

    syn_paths = []
    if tptp_archive.exists():
        syn_paths = extract_syn_files(tptp_archive, pelletier_dir, dry_run=dry_run)
        if not dry_run:
            sources.append({
                "source": "TPTP",
                "version": TPTP_VERSION,
                "url": TPTP_URL,
                "archive": TPTP_ARCHIVE_NAME,
                "sha256": sha256_of(tptp_archive),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

    pelletier_kept = {}
    release_kept = {}
    if syn_paths and not dry_run:
        pelletier_kept = select_pelletier(pelletier_dir)
        log(f"[pel]   {len(pelletier_kept)} Pelletier-tagged SYN files retained "
            f"(target {CATEGORY_TARGETS['pelletier']})")
        release_kept = select_tptp_syn_release(
            pelletier_dir, set(pelletier_kept.keys()),
            CATEGORY_TARGETS["tptp_syn_release"],
        )
        log(f"[syn]   {len(release_kept)} TPTP SYN release files retained "
            f"(target {CATEGORY_TARGETS['tptp_syn_release']})")
        kept_set = set(pelletier_kept.keys()) | set(release_kept.keys())
        removed = remove_syn_files_outside(pelletier_dir, kept_set)
        if removed:
            log(f"[trim]  removed {removed} SYN files not selected by filters")
        write_pelletier_names_tsv(pelletier_dir, pelletier_kept, release_kept)

    # Stage 2: ILTP archive -> classical-valid problems.
    iltp_kept = {}
    iltp_archive = CACHE_DIR / ILTP_ARCHIVE_NAME
    try:
        iltp_archive = download_to_cache(ILTP_URL, ILTP_URL_FALLBACK,
                                         ILTP_ARCHIVE_NAME, dry_run=dry_run)
    except RuntimeError as e:
        log(f"[warn]  ILTP download failed: {e}")
    if iltp_archive.exists() and not dry_run:
        iltp_kept = select_iltp(iltp_archive, iltp_dir, CATEGORY_TARGETS["iltp"])
        log(f"[iltp]  {len(iltp_kept)} classical-valid ILTP files retained "
            f"(target up to {CATEGORY_TARGETS['iltp']})")
        write_iltp_selected_tsv(iltp_dir, iltp_kept)
        sources.append({
            "source": "ILTP",
            "version": ILTP_VERSION,
            "url": ILTP_URL,
            "archive": ILTP_ARCHIVE_NAME,
            "sha256": sha256_of(iltp_archive),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    # Stage 3: generated and hand-crafted.
    if not dry_run:
        log(f"[php]   {write_php(DATA_DIR / 'php')} files written "
            f"(target {CATEGORY_TARGETS['php']})")
        log(f"[drink] {write_drinker(DATA_DIR / 'drinker')} files written "
            f"(target {CATEGORY_TARGETS['drinker']})")
        log(f"[hou]   {write_hou(DATA_DIR / 'hou')} files written "
            f"(target {CATEGORY_TARGETS['hou']})")
        log(f"[param] {write_parametric(DATA_DIR / 'parametric')} files written "
            f"(target {CATEGORY_TARGETS['parametric']})")
        log(f"[inv]   {write_invalid(DATA_DIR / 'invalid')} files written "
            f"(target {CATEGORY_TARGETS['invalid']})")
        write_sources_tsv(sources)

    elapsed = time.time() - started
    log(f"[done]  {elapsed:.1f}s")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage the eight datasets into data/ subdirs.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="describe actions without touching disk")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(dry_run=args.dry_run))
