"""TPTP FOF -> course syntax. Rejects equality and arithmetic.

CLI: python experiments/tptp_fof_to_course.py [--source pelletier|iltp|all] [--check]
"""

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lark import Lark
from lark.exceptions import LarkError

from logic.parser import parse as course_parse, ParseError
from logic.pretty import pretty
from logic.syntax import (
    Term, Formula,
    atom, neg, conj, disj, imp, iff, forall, exists, TOP, BOT,
)
from experiments._tptp_header import (
    parse_header, has_arithmetic, has_equality, ILTP_INCLUDE_RE,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


TPTP_FOF_GRAMMAR = r"""
?start: fof_formula
?fof_formula: fof_iff
?fof_iff: fof_imp
        | fof_imp "<=>" fof_imp        -> iff_op
?fof_imp: fof_or
       | fof_or "=>" fof_imp           -> imp_op
?fof_or: fof_and
      | fof_or "|" fof_and             -> or_op
?fof_and: fof_unary
       | fof_and "&" fof_unary         -> and_op
?fof_unary: "~" fof_unary              -> neg_op
         | "!" "[" var_list "]" ":" fof_unary -> forall_op
         | "?" "[" var_list "]" ":" fof_unary -> exists_op
         | fof_atom
?fof_atom: TRUE                        -> top_op
         | FALSE                       -> bot_op
         | NAME "(" fof_term ("," fof_term)* ")"  -> pred_app
         | NAME                                    -> pred_atom_bare
         | "(" fof_formula ")"
?fof_term: NAME "(" fof_term ("," fof_term)* ")"  -> func_app
        | VAR                                       -> var_ref
        | NAME                                      -> const_ref
var_list: VAR ("," VAR)*

TRUE: "$true"
FALSE: "$false"
VAR: /[A-Z][A-Za-z0-9_]*/
NAME: /[a-z][A-Za-z0-9_]*/

%import common.WS
%ignore WS
COMMENT: /%[^\n]*/
%ignore COMMENT
"""


_FOF_PARSER = Lark(TPTP_FOF_GRAMMAR, parser="lalr", maybe_placeholders=False)


def _tree_to_term(node) -> Term:
    if not hasattr(node, "data"):
        return node
    data = node.data
    children = node.children
    if data == "func_app":
        name = str(children[0])
        return Term("func", name, tuple(_tree_to_term(c) for c in children[1:]))
    if data == "var_ref":
        return Term("var", str(children[0]), ())
    if data == "const_ref":
        return Term("const", str(children[0]), ())
    raise ValueError(f"unknown TPTP term node: {data}")


def _tree_to_formula(node) -> Formula:
    if not hasattr(node, "data"):
        raise ValueError(f"expected Tree, got token: {node!r}")
    data = node.data
    children = node.children
    if data == "iff_op":
        return iff(_tree_to_formula(children[0]), _tree_to_formula(children[1]))
    if data == "imp_op":
        return imp(_tree_to_formula(children[0]), _tree_to_formula(children[1]))
    if data == "or_op":
        return disj(_tree_to_formula(children[0]), _tree_to_formula(children[1]))
    if data == "and_op":
        return conj(_tree_to_formula(children[0]), _tree_to_formula(children[1]))
    if data == "neg_op":
        return neg(_tree_to_formula(children[0]))
    if data == "forall_op":
        var_list = children[0]
        body = _tree_to_formula(children[1])
        for tok in reversed(var_list.children):
            body = forall(str(tok), body)
        return body
    if data == "exists_op":
        var_list = children[0]
        body = _tree_to_formula(children[1])
        for tok in reversed(var_list.children):
            body = exists(str(tok), body)
        return body
    if data == "top_op":
        return TOP
    if data == "bot_op":
        return BOT
    if data == "pred_app":
        pred = str(children[0])
        terms = tuple(_tree_to_term(c) for c in children[1:])
        return atom(pred, terms)
    if data == "pred_atom_bare":
        return atom(str(children[0]), ())
    raise ValueError(f"unknown TPTP formula node: {data}")


class TPTPParseError(ValueError):
    pass


def parse_tptp_formula(s: str) -> Formula:
    try:
        tree = _FOF_PARSER.parse(s)
    except LarkError as e:
        raise TPTPParseError(str(e)) from e
    return _tree_to_formula(tree)


_FOF_KW_RE = re.compile(r"\bfof\s*\(")


def _strip_line_comments(text: str) -> str:
    out_lines = []
    for ln in text.splitlines():
        idx = ln.find("%")
        if idx >= 0:
            ln = ln[:idx]
        out_lines.append(ln)
    return "\n".join(out_lines)


def _split_top_level_commas(s: str) -> list:
    out = []
    depth = 0
    bracket = 0
    cur = []
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "[":
            bracket += 1
        elif c == "]":
            bracket -= 1
        if c == "," and depth == 0 and bracket == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if cur:
        out.append("".join(cur))
    return out


def extract_fof_blocks(text: str) -> list:
    """Return list of (name, role, formula_str) tuples from TPTP text."""
    cleaned = _strip_line_comments(text)
    out = []
    pos = 0
    while True:
        m = _FOF_KW_RE.search(cleaned, pos)
        if not m:
            break
        i = m.end()
        depth = 1
        body_start = i
        while i < len(cleaned) and depth > 0:
            c = cleaned[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            break
        body = cleaned[body_start:i]
        pos = i + 1
        parts = _split_top_level_commas(body)
        if len(parts) < 3:
            continue
        name = parts[0].strip()
        role = parts[1].strip()
        formula = parts[2].strip()
        out.append((name, role, formula))
    return out


class ConvertReason(ValueError):
    pass


def convert_text(text: str) -> Formula:
    """Convert raw TPTP text to a single course-syntax Formula.

    Multi-axiom assembly: if there is at least one conjecture and one or more
    axioms, emit `(ax1 /\\ ax2 /\\ ...) -> conjecture`. With axioms only, emit
    their conjunction. With a conjecture only, emit it directly.
    """
    if has_arithmetic(text):
        raise ConvertReason("arithmetic atom present")
    if has_equality(text):
        raise ConvertReason("equality present")
    if ILTP_INCLUDE_RE.search(text):
        raise ConvertReason("include() directive present")
    blocks = extract_fof_blocks(text)
    if not blocks:
        raise ConvertReason("no fof() blocks found")
    axioms = []
    conjecture = None
    for _name, role, formula_str in blocks:
        f = parse_tptp_formula(formula_str)
        role_l = role.lower()
        if role_l == "conjecture":
            if conjecture is not None:
                raise ConvertReason("multiple conjectures not supported")
            conjecture = f
        else:
            axioms.append(f)
    if conjecture is None and not axioms:
        raise ConvertReason("no usable formulas")
    if conjecture is None:
        result = axioms[0]
        for a in axioms[1:]:
            result = conj(result, a)
        return result
    if not axioms:
        return conjecture
    ax = axioms[0]
    for a in axioms[1:]:
        ax = conj(ax, a)
    return imp(ax, conjecture)


def round_trip(course_str: str) -> bool:
    f = course_parse(course_str)
    f2 = course_parse(pretty(f))
    return f == f2


def write_text_ascii(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(content)


def write_tsv(path: Path, header: list, rows: list) -> None:
    lines = ["\t".join(header)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in header))
    write_text_ascii(path, "\n".join(lines) + "\n")


def _label_for_pelletier(stem: str, header: dict) -> str:
    from experiments._tptp_header import PELLETIER_NUM_RE
    m = PELLETIER_NUM_RE.search(header.get("names", ""))
    if m:
        return f"pel_{m.group(1)}"
    return f"syn_release_{stem}"


def convert_directory(src_dir: Path, source_name: str,
                      out_dir: Path, failed_dir: Path,
                      do_write: bool) -> tuple[int, int, int]:
    """Returns (converted, rejected, round_trip_failed)."""
    paths = sorted(src_dir.glob("*.p"))
    converted = 0
    rejected = 0
    rt_failed = 0
    formulas = []
    rows = []
    for path in paths:
        text = path.read_text(encoding="latin-1", errors="replace")
        header = parse_header(text)
        stem = path.stem
        try:
            f = convert_text(text)
        except (ConvertReason, TPTPParseError, ValueError) as e:
            rejected += 1
            if do_write:
                failed_dir.mkdir(parents=True, exist_ok=True)
                (failed_dir / f"{stem}.reason").write_text(
                    f"convert: {type(e).__name__}: {e}\n",
                    encoding="ascii",
                )
            continue
        course_str = pretty(f)
        try:
            f2 = course_parse(course_str)
        except ParseError as e:
            rt_failed += 1
            if do_write:
                failed_dir.mkdir(parents=True, exist_ok=True)
                (failed_dir / f"{stem}.reason").write_text(
                    f"reparse: {e}\n", encoding="ascii",
                )
            continue
        if f != f2:
            rt_failed += 1
            if do_write:
                failed_dir.mkdir(parents=True, exist_ok=True)
                (failed_dir / f"{stem}.reason").write_text(
                    f"round_trip mismatch:\n  course: {course_str}\n  reparsed pretty: {pretty(f2)}\n",
                    encoding="ascii",
                )
            continue
        converted += 1
        if source_name == "pelletier":
            label = _label_for_pelletier(stem, header)
            origin = "PELLETIER" if label.startswith("pel_") else "TPTP_SYN_RELEASE"
            tags = "PELLETIER,TPTP_SYN" if origin == "PELLETIER" else "TPTP_SYN"
            src_field = f"TPTP {stem}; {header.get('names', '').strip()}"
        else:
            label = f"iltp_{stem}"
            origin = "ILTP"
            tags = "ILTP"
            src_field = f"ILTP {stem}"
        status = header.get("status", "Theorem").upper()
        if status == "THEOREM":
            status_label = "THEOREM"
        elif status == "COUNTERSATISFIABLE":
            status_label = "COUNTERSAT"
        else:
            status_label = status
        line = converted
        formulas.append(course_str)
        rows.append({
            "line": line,
            "label": label,
            "status": status_label,
            "tags": tags,
            "source": src_field,
        })
    if do_write and formulas:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_text_ascii(out_dir / f"{source_name}.fol", "\n".join(formulas) + "\n")
        write_tsv(
            out_dir / f"{source_name}.tsv",
            ["line", "label", "status", "tags", "source"],
            rows,
        )
    return converted, rejected, rt_failed


def main(source: str, do_check: bool) -> int:
    out_dir = DATA_DIR / "converted"
    failed_dir = out_dir / "failed"
    sources = []
    if source in ("pelletier", "all"):
        sources.append(("pelletier", DATA_DIR / "pelletier"))
    if source in ("iltp", "all"):
        sources.append(("iltp", DATA_DIR / "iltp"))
    total_failed = 0
    for name, src_dir in sources:
        if not src_dir.exists():
            print(f"[skip] {name}: {src_dir} not found", flush=True)
            continue
        converted, rejected, rt_failed = convert_directory(
            src_dir, name, out_dir, failed_dir, do_write=not do_check,
        )
        print(
            f"[{name:9s}] converted={converted} rejected={rejected} round_trip_failed={rt_failed}",
            flush=True,
        )
        total_failed += rt_failed
    return 0 if total_failed == 0 else 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert TPTP-FOF .p files to course syntax.")
    p.add_argument("--source", choices=["pelletier", "iltp", "all"], default="all")
    p.add_argument("--check", action="store_true",
                   help="round-trip without writing files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args.source, args.check))
