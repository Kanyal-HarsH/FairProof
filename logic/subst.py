"""Capture-avoiding substitution. Alpha-renames on conflict."""

from logic.syntax import Term, Formula, term_vars, free_vars


def subst_term(t: Term, x: str, replacement: Term) -> Term:
    if t.kind == "var" and t.name == x:
        return replacement
    if t.kind == "func":
        return Term("func", t.name, tuple(subst_term(a, x, replacement) for a in t.args))
    return t


def _rename_term(t: Term, old: str, new: str) -> Term:
    if t.kind == "var" and t.name == old:
        return Term("var", new, ())
    if t.kind == "func":
        return Term("func", t.name, tuple(_rename_term(a, old, new) for a in t.args))
    return t


def alpha_rename(f: Formula, old: str, new: str) -> Formula:
    if f.kind == "atom":
        pred, args = f.payload
        return Formula("atom", (pred, tuple(_rename_term(a, old, new) for a in args)))
    if f.kind == "neg":
        return Formula("neg", (alpha_rename(f.payload[0], old, new),))
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        return Formula(f.kind, (alpha_rename(a, old, new), alpha_rename(b, old, new)))
    if f.kind in {"forall", "exists"}:
        x, body = f.payload
        if x == old:
            return f
        return Formula(f.kind, (x, alpha_rename(body, old, new)))
    return f


def _fresh(used: frozenset, hint: str) -> str:
    if hint not in used:
        return hint
    i = 1
    while f"{hint}_{i}" in used:
        i += 1
    return f"{hint}_{i}"


def subst_formula(f: Formula, x: str, replacement: Term) -> Formula:
    if f.kind == "atom":
        pred, args = f.payload
        return Formula("atom", (pred, tuple(subst_term(a, x, replacement) for a in args)))
    if f.kind == "neg":
        return Formula("neg", (subst_formula(f.payload[0], x, replacement),))
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        return Formula(f.kind, (
            subst_formula(a, x, replacement),
            subst_formula(b, x, replacement),
        ))
    if f.kind in {"forall", "exists"}:
        bound, body = f.payload
        if bound == x:
            return f
        if x not in free_vars(body):
            return f
        repl_vars = term_vars(replacement)
        if bound in repl_vars:
            used = free_vars(body) | repl_vars | {x, bound}
            new_bound = _fresh(used, bound)
            renamed = alpha_rename(body, bound, new_bound)
            return Formula(f.kind, (new_bound, subst_formula(renamed, x, replacement)))
        return Formula(f.kind, (bound, subst_formula(body, x, replacement)))
    return f
