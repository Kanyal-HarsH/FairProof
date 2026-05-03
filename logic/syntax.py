"""FOL AST. Term and Formula are frozen and hashable."""

from dataclasses import dataclass

VALID_TERM_KINDS = frozenset({"var", "const", "func"})
VALID_FORMULA_KINDS = frozenset({
    "atom", "neg", "and", "or", "imp", "iff",
    "forall", "exists", "top", "bot",
})


@dataclass(frozen=True, slots=True)
class Term:
    kind: str
    name: str
    args: tuple

    def __post_init__(self):
        if not isinstance(self.args, tuple):
            raise TypeError("Term.args must be a tuple")
        if self.kind in {"var", "const"} and len(self.args) != 0:
            raise ValueError(f"{self.kind} term must have no args")


@dataclass(frozen=True, slots=True)
class Formula:
    kind: str
    payload: tuple

    def __post_init__(self):
        if self.kind not in VALID_FORMULA_KINDS:
            raise ValueError(f"unknown formula kind: {self.kind}")
        if not isinstance(self.payload, tuple):
            raise TypeError("Formula.payload must be a tuple")


def var(name: str) -> Term:
    return Term("var", name, ())


def const(name: str) -> Term:
    return Term("const", name, ())


def func(name: str, args) -> Term:
    return Term("func", name, tuple(args))


def atom(pred: str, args=()) -> Formula:
    return Formula("atom", (pred, tuple(args)))


def neg(f: Formula) -> Formula:
    return Formula("neg", (f,))


def conj(a: Formula, b: Formula) -> Formula:
    return Formula("and", (a, b))


def disj(a: Formula, b: Formula) -> Formula:
    return Formula("or", (a, b))


def imp(a: Formula, b: Formula) -> Formula:
    return Formula("imp", (a, b))


def iff(a: Formula, b: Formula) -> Formula:
    return Formula("iff", (a, b))


def forall(x: str, body: Formula) -> Formula:
    return Formula("forall", (x, body))


def exists(x: str, body: Formula) -> Formula:
    return Formula("exists", (x, body))


TOP = Formula("top", ())
BOT = Formula("bot", ())


def term_vars(t: Term) -> frozenset:
    if t.kind == "var":
        return frozenset({t.name})
    if t.kind == "const":
        return frozenset()
    out = set()
    for a in t.args:
        out |= term_vars(a)
    return frozenset(out)


def free_vars(f: Formula) -> frozenset:
    if f.kind == "atom":
        _, args = f.payload
        out = set()
        for a in args:
            out |= term_vars(a)
        return frozenset(out)
    if f.kind == "neg":
        (g,) = f.payload
        return free_vars(g)
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        return free_vars(a) | free_vars(b)
    if f.kind in {"forall", "exists"}:
        x, body = f.payload
        return free_vars(body) - {x}
    return frozenset()


def subterms(t: Term):
    yield t
    for a in t.args:
        yield from subterms(a)


def subformulas(f: Formula):
    yield f
    if f.kind == "neg":
        yield from subformulas(f.payload[0])
    elif f.kind in {"and", "or", "imp", "iff"}:
        yield from subformulas(f.payload[0])
        yield from subformulas(f.payload[1])
    elif f.kind in {"forall", "exists"}:
        yield from subformulas(f.payload[1])


def is_atom(f: Formula) -> bool:
    return f.kind == "atom"
