"""
Relevance score: k1*pred_overlap + k2*term_reuse - k3*term_depth.
pred_overlap is a syntactic surrogate of E-matching (de Moura & Bjorner 2007).

"""

from logic.syntax import Term, Formula
from logic.subst import subst_formula


def term_depth(t: Term) -> int:
    if t.kind in {"var", "const"}:
        return 0
    return 1 + max((term_depth(a) for a in t.args), default=0)


def collect_atoms(f: Formula) -> list:
    out = []
    _collect(f, out)
    return out


def _collect(f: Formula, out: list) -> None:
    if f.kind == "atom":
        out.append(f)
        return
    if f.kind == "neg":
        _collect(f.payload[0], out)
        return
    if f.kind in {"and", "or", "imp", "iff"}:
        a, b = f.payload
        _collect(a, out)
        _collect(b, out)
        return
    if f.kind in {"forall", "exists"}:
        _collect(f.payload[1], out)


def _atoms_on_side(side: frozenset) -> set:
    out = set()
    for f in side:
        if f.kind == "atom":
            out.add(f)
    return out


def _term_matches(t_cand, t_opp) -> bool:
    # var matches anything (unfreed-quantifier hole); const and func match
    # structurally.
    if t_cand.kind == "var":
        return True
    if t_cand.kind == "const":
        return t_opp.kind == "const" and t_cand.name == t_opp.name
    if t_cand.kind == "func":
        if t_opp.kind != "func" or t_cand.name != t_opp.name:
            return False
        if len(t_cand.args) != len(t_opp.args):
            return False
        return all(_term_matches(c, o) for c, o in zip(t_cand.args, t_opp.args))
    return False


def _atom_matches(a_cand, a_opp) -> bool:
    if a_cand.kind != "atom" or a_opp.kind != "atom":
        return False
    p_cand, args_cand = a_cand.payload
    p_opp, args_opp = a_opp.payload
    if p_cand != p_opp or len(args_cand) != len(args_opp):
        return False
    return all(_term_matches(c, o) for c, o in zip(args_cand, args_opp))


def _terms_in_term(t: Term) -> set:
    if t.kind in {"var", "const"}:
        return {t}
    out = {t}
    for a in t.args:
        out |= _terms_in_term(a)
    return out


def _terms_in_sequent_atoms(seq) -> set:
    out = set()
    for f in seq.antecedent | seq.succedent:
        if f.kind == "atom":
            _, args = f.payload
            for a in args:
                out |= _terms_in_term(a)
    return out


def pred_overlap(term: Term, principal: Formula, seq, side: str) -> int:
    if principal.kind not in {"forall", "exists"}:
        return 0
    x, body = principal.payload
    instantiated = subst_formula(body, x, term)
    atoms = collect_atoms(instantiated)
    opposite = seq.succedent if side == "antecedent" else seq.antecedent
    opposite_atoms = _atoms_on_side(opposite)
    return sum(1 for a in atoms
               if any(_atom_matches(a, oa) for oa in opposite_atoms))


def term_reuse(term: Term, seq) -> int:
    return 1 if term in _terms_in_sequent_atoms(seq) else 0


def relevance_score(term: Term, principal: Formula, seq, side: str,
                    k1: float = 4.0, k2: float = 2.0, k3: float = 1.0) -> float:
    return (
        k1 * pred_overlap(term, principal, seq, side)
        + k2 * term_reuse(term, seq)
        - k3 * term_depth(term)
    )


def rank_terms(terms, principal: Formula, seq, side: str,
               k1: float = 4.0, k2: float = 2.0, k3: float = 1.0) -> list:

    """Returns terms by descending relevance score with lex tiebreak."""

    if principal.kind not in {"forall", "exists"}:
        return sorted(terms, key=_term_key)
    x, body = principal.payload
    opposite = seq.succedent if side == "antecedent" else seq.antecedent
    opposite_atoms = _atoms_on_side(opposite)
    seq_terms = _terms_in_sequent_atoms(seq)

    scored = []
    for t in terms:
        instantiated = subst_formula(body, x, t)
        atoms = collect_atoms(instantiated)
        po = sum(1 for a in atoms
                 if any(_atom_matches(a, oa) for oa in opposite_atoms))
        tr = 1 if t in seq_terms else 0
        td = term_depth(t)
        score = k1 * po + k2 * tr - k3 * td
        scored.append((score, _term_key(t), t))
    scored.sort(key=lambda triple: (-triple[0], triple[1]))
    return [t for (_score, _key, t) in scored]


def _term_key(t: Term) -> str:
    if t.kind in {"var", "const"}:
        return t.name
    return f"{t.name}(" + ",".join(_term_key(a) for a in t.args) + ")"
