"""

LK' rules. apply_X returns premise sequents.
forall L, exists R keep the principal. forall R, exists L drop it
and use a fresh eigenvariable.

"""

from dataclasses import dataclass, field

from logic.syntax import Term, Formula, free_vars
from logic.sequent import (
    Sequent, add_left, add_right, remove_left, remove_right, is_axiom,
)
from logic.subst import subst_formula


class RuleError(ValueError):
    pass


@dataclass
class RuleContext:
    fresh_counter: int = 0
    used_names: frozenset = field(default_factory=frozenset)
    eigenvar_count: int = 0
    max_term_depth: int = 0
    axiom_closures: int = 0
    branching_applications: int = 0

    def fresh_const(self, hint: str = "c") -> Term:
        self.fresh_counter += 1
        name = f"{hint}_{self.fresh_counter}"
        while name in self.used_names:
            self.fresh_counter += 1
            name = f"{hint}_{self.fresh_counter}"
        self.used_names = self.used_names | {name}
        return Term("const", name, ())


def _term_depth(t: Term) -> int:
    if t.kind in {"var", "const"}:
        return 0
    return 1 + max((_term_depth(a) for a in t.args), default=0)


def _record_term_depth(ctx, t: Term) -> None:
    if ctx is None:
        return
    d = _term_depth(t)
    if d > ctx.max_term_depth:
        ctx.max_term_depth = d


def _consts_in_term(t: Term) -> frozenset:
    if t.kind == "const":
        return frozenset({t.name})
    if t.kind == "func":
        out = set()
        for a in t.args:
            out |= _consts_in_term(a)
        return frozenset(out)
    return frozenset()


def _consts_in_formula(f: Formula) -> frozenset:
    if f.kind == "atom":
        _, args = f.payload
        out = set()
        for a in args:
            out |= _consts_in_term(a)
        return frozenset(out)
    if f.kind == "neg":
        return _consts_in_formula(f.payload[0])
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        return _consts_in_formula(a) | _consts_in_formula(b)
    if f.kind in {"forall", "exists"}:
        return _consts_in_formula(f.payload[1])
    return frozenset()


def _names_in_sequent(seq: Sequent) -> frozenset:
    out = set()
    for f in seq.antecedent | seq.succedent:
        out |= _consts_in_formula(f)
        out |= free_vars(f)
    return frozenset(out)


def apply_id(seq: Sequent, principal: Formula, ctx: RuleContext):
    if not is_axiom(seq):
        raise RuleError("not an axiom")
    return ()


def apply_top_R(seq: Sequent, principal: Formula, ctx: RuleContext):
    if principal.kind != "top" or principal not in seq.succedent:
        raise RuleError("top R requires top in succedent")
    return ()


def apply_bot_L(seq: Sequent, principal: Formula, ctx: RuleContext):
    if principal.kind != "bot" or principal not in seq.antecedent:
        raise RuleError("bot L requires bot in antecedent")
    return ()


def apply_neg_L(seq, principal, ctx):

    """neg L. Move A from antecedent to succedent."""

    if principal.kind != "neg" or principal not in seq.antecedent:
        raise RuleError("~L requires ~A in antecedent")
    (a,) = principal.payload
    return (add_right(remove_left(seq, principal), a),)


def apply_neg_R(seq, principal, ctx):

    """neg R. Move A from succedent to antecedent."""

    if principal.kind != "neg" or principal not in seq.succedent:
        raise RuleError("~R requires ~A in succedent")
    (a,) = principal.payload
    return (add_left(remove_right(seq, principal), a),)


def apply_and_L(seq, principal, ctx):

    """and L. Replace A /\\ B with A and B on the left."""

    if principal.kind != "and" or principal not in seq.antecedent:
        raise RuleError("/\\ L requires A /\\ B in antecedent")
    a, b = principal.payload
    return (add_left(add_left(remove_left(seq, principal), a), b),)


def apply_and_R(seq, principal, ctx):

    """and R. Two premises: one with A on the right, one with B on the right."""

    if principal.kind != "and" or principal not in seq.succedent:
        raise RuleError("/\\ R requires A /\\ B in succedent")
    a, b = principal.payload
    base = remove_right(seq, principal)
    return (add_right(base, a), add_right(base, b))


def apply_or_L(seq, principal, ctx):

    """or L. Two premises: one with A on the left, one with B on the left."""

    if principal.kind != "or" or principal not in seq.antecedent:
        raise RuleError("\\/ L requires A \\/ B in antecedent")
    a, b = principal.payload
    base = remove_left(seq, principal)
    return (add_left(base, a), add_left(base, b))


def apply_or_R(seq, principal, ctx):

    """or R. Replace A \\/ B with A and B on the right."""

    if principal.kind != "or" or principal not in seq.succedent:
        raise RuleError("\\/ R requires A \\/ B in succedent")
    a, b = principal.payload
    return (add_right(add_right(remove_right(seq, principal), a), b),)


def apply_imp_L(seq, principal, ctx):

    """imp L. Two premises: A on the right, B on the left."""

    if principal.kind != "imp" or principal not in seq.antecedent:
        raise RuleError("-> L requires A -> B in antecedent")
    a, b = principal.payload
    base = remove_left(seq, principal)
    return (add_right(base, a), add_left(base, b))


def apply_imp_R(seq, principal, ctx):

    """imp R. Add A on the left and B on the right."""

    if principal.kind != "imp" or principal not in seq.succedent:
        raise RuleError("-> R requires A -> B in succedent")
    a, b = principal.payload
    return (add_right(add_left(remove_right(seq, principal), a), b),)


def apply_forall_L(seq, principal, ctx, term):

    """forall L. Add A[t/x]; keep the principal in the premise."""

    if principal.kind != "forall" or principal not in seq.antecedent:
        raise RuleError("forall L requires forall x. A in antecedent")
    x, body = principal.payload
    _record_term_depth(ctx, term)
    instantiated = subst_formula(body, x, term)
    return (add_left(seq, instantiated),)


def apply_forall_R(seq, principal, ctx):

    """forall R. Drop the principal; instantiate with a fresh eigenvariable."""

    if principal.kind != "forall" or principal not in seq.succedent:
        raise RuleError("forall R requires forall x. A in succedent")
    x, body = principal.payload
    used = _names_in_sequent(seq)
    ctx.used_names = ctx.used_names | used
    a = ctx.fresh_const("a")
    if a.name in used:
        raise RuleError(f"freshness violated: {a.name} appears in conclusion")
    ctx.eigenvar_count += 1
    instantiated = subst_formula(body, x, a)
    return (add_right(remove_right(seq, principal), instantiated),)


def apply_exists_L(seq, principal, ctx):

    """exists L. Drop the principal; instantiate with a fresh eigenvariable."""

    if principal.kind != "exists" or principal not in seq.antecedent:
        raise RuleError("exists L requires exists x. A in antecedent")
    x, body = principal.payload
    used = _names_in_sequent(seq)
    ctx.used_names = ctx.used_names | used
    a = ctx.fresh_const("a")
    if a.name in used:
        raise RuleError(f"freshness violated: {a.name} appears in conclusion")
    ctx.eigenvar_count += 1
    instantiated = subst_formula(body, x, a)
    return (add_left(remove_left(seq, principal), instantiated),)


def apply_exists_R(seq, principal, ctx, term):

    """exists R. Add A[t/x]; keep the principal in the premise."""

    if principal.kind != "exists" or principal not in seq.succedent:
        raise RuleError("exists R requires exists x. A in succedent")
    x, body = principal.payload
    _record_term_depth(ctx, term)
    instantiated = subst_formula(body, x, term)
    return (add_right(seq, instantiated),)


RULE_NAMES = (
    "id", "top_R", "bot_L",
    "neg_L", "neg_R",
    "and_L", "and_R", "or_L", "or_R", "imp_L", "imp_R",
    "forall_L", "forall_R", "exists_L", "exists_R",
)
