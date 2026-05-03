"""AST -> ASCII course syntax. Every binary operand wrapped in parens."""

from logic.syntax import Term, Formula


def pretty_term(t: Term) -> str:
    if t.kind in {"var", "const"}:
        return t.name
    args = ", ".join(pretty_term(a) for a in t.args)
    return f"{t.name}({args})"


def pretty(f: Formula) -> str:
    if f.kind == "top":
        return "top"
    if f.kind == "bot":
        return "bot"
    if f.kind == "atom":
        pred, args = f.payload
        if not args:
            return pred
        return f"{pred}({', '.join(pretty_term(a) for a in args)})"
    if f.kind == "neg":
        (g,) = f.payload
        return f"~({pretty(g)})"
    if f.kind == "and":
        a, b = f.payload
        return f"({pretty(a)} /\\ {pretty(b)})"
    if f.kind == "or":
        a, b = f.payload
        return f"({pretty(a)} \\/ {pretty(b)})"
    if f.kind == "imp":
        a, b = f.payload
        return f"({pretty(a)} -> {pretty(b)})"
    if f.kind == "iff":
        a, b = f.payload
        return f"({pretty(a)} <-> {pretty(b)})"
    if f.kind == "forall":
        x, body = f.payload
        return f"(forall {x}. {pretty(body)})"
    if f.kind == "exists":
        x, body = f.payload
        return f"(exists {x}. {pretty(body)})"
    raise ValueError(f"unknown formula kind: {f.kind}")
