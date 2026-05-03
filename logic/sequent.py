"""Sequent for LK'. Sides are frozensets; the whole thing is hashable."""

from dataclasses import dataclass

from logic.syntax import Formula, TOP, BOT
from logic.pretty import pretty


@dataclass(frozen=True, slots=True)
class Sequent:
    antecedent: frozenset
    succedent: frozenset


def make_sequent(left=(), right=()) -> Sequent:
    return Sequent(frozenset(left), frozenset(right))


def add_left(seq: Sequent, f: Formula) -> Sequent:
    return Sequent(seq.antecedent | {f}, seq.succedent)


def add_right(seq: Sequent, f: Formula) -> Sequent:
    return Sequent(seq.antecedent, seq.succedent | {f})


def remove_left(seq: Sequent, f: Formula) -> Sequent:
    return Sequent(seq.antecedent - {f}, seq.succedent)


def remove_right(seq: Sequent, f: Formula) -> Sequent:
    return Sequent(seq.antecedent, seq.succedent - {f})


def is_axiom(seq: Sequent) -> bool:
    if TOP in seq.succedent:
        return True
    if BOT in seq.antecedent:
        return True
    for f in seq.antecedent:
        if f.kind == "atom" and f in seq.succedent:
            return True
    return False


def pretty_sequent(seq: Sequent) -> str:
    left = ", ".join(sorted(pretty(f) for f in seq.antecedent))
    right = ", ".join(sorted(pretty(f) for f in seq.succedent))
    return f"{left} |- {right}"
