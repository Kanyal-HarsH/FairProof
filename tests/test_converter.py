import pytest
from pathlib import Path

from experiments.tptp_fof_to_course import (
    parse_tptp_formula, convert_text, ConvertReason, extract_fof_blocks, round_trip,
)
from logic.parser import parse as course_parse
from logic.pretty import pretty
from logic.syntax import (
    var, const, atom, neg, conj, disj, imp, iff, forall, exists, TOP, BOT,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_parse_tptp_simple_atom():
    f = parse_tptp_formula("p(X)")
    assert f == atom("p", (var("X"),))


def test_parse_tptp_propositional_imp():
    f = parse_tptp_formula("p => q")
    assert f == imp(atom("p"), atom("q"))


def test_parse_tptp_universal():
    f = parse_tptp_formula("![X]: p(X)")
    assert f == forall("X", atom("p", (var("X"),)))


def test_parse_tptp_existential_with_imp():
    f = parse_tptp_formula("?[X]: (p(X) => q(X))")
    assert f == exists("X", imp(atom("p", (var("X"),)), atom("q", (var("X"),))))


def test_parse_tptp_multi_var_quantifier_expands():
    f = parse_tptp_formula("![X, Y]: p(X, Y)")
    assert f == forall("X", forall("Y", atom("p", (var("X"), var("Y")))))


def test_parse_tptp_negation_and_conj():
    f = parse_tptp_formula("~p & q")
    assert f == conj(neg(atom("p")), atom("q"))


def test_parse_tptp_constants():
    assert parse_tptp_formula("$true") == TOP
    assert parse_tptp_formula("$false") == BOT


def test_extract_fof_blocks_strips_comments():
    text = """
% file header
fof(a1, axiom, p(a)).
% inline comment
fof(c, conjecture, ![X]: p(X)).
"""
    blocks = extract_fof_blocks(text)
    assert len(blocks) == 2
    assert blocks[0] == ("a1", "axiom", "p(a)")
    assert blocks[1] == ("c", "conjecture", "![X]: p(X)")


def test_convert_text_single_conjecture():
    text = "fof(c, conjecture, ![X]: (p(X) | ~p(X)))."
    f = convert_text(text)
    expected = forall("X", disj(atom("p", (var("X"),)), neg(atom("p", (var("X"),)))))
    assert f == expected


def test_convert_text_axioms_to_implication():
    text = """
fof(a1, axiom, p).
fof(a2, axiom, q).
fof(c, conjecture, p & q).
"""
    f = convert_text(text)
    expected = imp(conj(atom("p"), atom("q")), conj(atom("p"), atom("q")))
    assert f == expected


def test_convert_text_axioms_only_conjoins():
    text = """
fof(a1, axiom, p).
fof(a2, axiom, q).
"""
    f = convert_text(text)
    assert f == conj(atom("p"), atom("q"))


def test_convert_rejects_arithmetic():
    text = "fof(a, axiom, $less(X, Y))."
    with pytest.raises(ConvertReason):
        convert_text(text)


def test_convert_rejects_equality():
    text = "fof(a, axiom, X = Y)."
    with pytest.raises(ConvertReason):
        convert_text(text)


def test_convert_rejects_inequality():
    text = "fof(a, axiom, X != Y)."
    with pytest.raises(ConvertReason):
        convert_text(text)


def test_convert_rejects_include():
    text = """
include('Axioms/foo.ax').
fof(a, axiom, p).
"""
    with pytest.raises(ConvertReason):
        convert_text(text)


def test_round_trip_preserves_formula():
    text = "fof(c, conjecture, ![X]: ?[Y]: r(X, Y))."
    f = convert_text(text)
    s = pretty(f)
    assert round_trip(s)
    assert course_parse(s) == f


def test_imp_right_associative_in_tptp():
    f = parse_tptp_formula("p => q => r")
    assert f == imp(atom("p"), imp(atom("q"), atom("r")))


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "data" / "pelletier").glob("*.p")))
def test_pelletier_files_round_trip(path):
    text = path.read_text(encoding="latin-1", errors="replace")
    f = convert_text(text)
    s = pretty(f)
    f2 = course_parse(s)
    assert f == f2, f"round-trip failed for {path.name}"


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "data" / "iltp").glob("*.p")))
def test_iltp_files_round_trip(path):
    text = path.read_text(encoding="latin-1", errors="replace")
    f = convert_text(text)
    s = pretty(f)
    f2 = course_parse(s)
    assert f == f2, f"round-trip failed for {path.name}"
