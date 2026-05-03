"""First-order unification with occurs check. Returns dict or None."""

from logic.syntax import Term, Formula


def _walk(t: Term, sigma: dict) -> Term:
    while t.kind == "var" and t.name in sigma:
        t = sigma[t.name]
    return t


def _occurs(x: str, t: Term, sigma: dict) -> bool:
    t = _walk(t, sigma)
    if t.kind == "var":
        return t.name == x
    if t.kind == "func":
        return any(_occurs(x, a, sigma) for a in t.args)
    return False


def unify_terms(t1: Term, t2: Term, sigma=None):
    sigma = {} if sigma is None else dict(sigma)
    t1 = _walk(t1, sigma)
    t2 = _walk(t2, sigma)
    if t1 == t2:
        return sigma
    if t1.kind == "var":
        if _occurs(t1.name, t2, sigma):
            return None
        sigma[t1.name] = t2
        return sigma
    if t2.kind == "var":
        if _occurs(t2.name, t1, sigma):
            return None
        sigma[t2.name] = t1
        return sigma
    if t1.kind == "func" and t2.kind == "func":
        if t1.name != t2.name or len(t1.args) != len(t2.args):
            return None
        for a, b in zip(t1.args, t2.args):
            sigma = unify_terms(a, b, sigma)
            if sigma is None:
                return None
        return sigma
    if t1.kind == "const" and t2.kind == "const":
        return sigma if t1.name == t2.name else None
    return None


def unify_atoms(a1: Formula, a2: Formula):
    if a1.kind != "atom" or a2.kind != "atom":
        return None
    p1, args1 = a1.payload
    p2, args2 = a2.payload
    if p1 != p2 or len(args1) != len(args2):
        return None
    sigma = {}
    for x, y in zip(args1, args2):
        sigma = unify_terms(x, y, sigma)
        if sigma is None:
            return None
    return sigma


def apply_subst_term(sigma: dict, t: Term) -> Term:
    if t.kind == "var" and t.name in sigma:
        return apply_subst_term(sigma, sigma[t.name])
    if t.kind == "func":
        return Term("func", t.name, tuple(apply_subst_term(sigma, a) for a in t.args))
    return t


def apply_subst_formula(sigma: dict, f: Formula) -> Formula:
    if f.kind == "atom":
        pred, args = f.payload
        return Formula("atom", (pred, tuple(apply_subst_term(sigma, a) for a in args)))
    if f.kind == "neg":
        return Formula("neg", (apply_subst_formula(sigma, f.payload[0]),))
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        return Formula(f.kind, (apply_subst_formula(sigma, a), apply_subst_formula(sigma, b)))
    if f.kind in {"forall", "exists"}:
        x, body = f.payload
        return Formula(f.kind, (x, apply_subst_formula(sigma, body)))
    return f
