"""TPTP-FOF header regexes and content filters."""

import re

HEADER_NAMES_RE = re.compile(
    r"^%\s*Names\s*:\s*(.*?)(?=^%\s*[A-Z]\S*\s*:|\Z)",
    re.MULTILINE | re.DOTALL,
)
HEADER_STATUS_RE = re.compile(r"^%\s*Status\s*:\s*(\S+)", re.MULTILINE)
HEADER_SPC_RE = re.compile(r"^%\s*SPC\s*:\s*(\S+)", re.MULTILINE)
HEADER_RATING_RE = re.compile(r"^%\s*Rating\s*:\s*([\d.]+)", re.MULTILINE)
PELLETIER_NUM_RE = re.compile(r"Pelletier\s+(\d+)\b", re.IGNORECASE)
ILTP_INTUIT_RE = re.compile(
    r"^%\s*Status\s*\(\s*intuit\.?\s*\)\s*:\s*(\S+)",
    re.IGNORECASE | re.MULTILINE,
)
ILTP_INCLUDE_RE = re.compile(r"^\s*include\s*\(", re.MULTILINE)


ARITHMETIC_TOKENS = (
    "$less", "$lesseq", "$greater", "$greatereq",
    "$sum", "$difference", "$product", "$quotient",
    "$real", "$rat", "$int", "$distinct",
)


_EQ_FUN_RE = re.compile(r"\bequal\s*\(")
_EQ_OP_RE = re.compile(r"(?<![<>!=])=(?![=>])")


def parse_header(text: str) -> dict:
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
    return out


def has_arithmetic(body: str) -> bool:
    return any(tok in body for tok in ARITHMETIC_TOKENS)


def has_equality(body: str) -> bool:
    code_lines = [ln for ln in body.splitlines() if not ln.lstrip().startswith("%")]
    code = "\n".join(code_lines)
    if _EQ_FUN_RE.search(code):
        return True
    if _EQ_OP_RE.search(code):
        return True
    if "!=" in code:
        return True
    return False
