from logic.syntax import atom, var, conj, TOP, BOT
from logic.sequent import (
    Sequent, make_sequent, add_left, add_right, remove_left, remove_right,
    is_axiom, pretty_sequent,
)


def test_make_sequent_uses_frozenset():
    s = make_sequent([atom("P")], [atom("Q")])
    assert isinstance(s.antecedent, frozenset)
    assert isinstance(s.succedent, frozenset)


def test_add_remove_returns_new_sequent():
    s = make_sequent([], [])
    s2 = add_left(s, atom("P"))
    assert s.antecedent == frozenset()
    assert s2.antecedent == frozenset({atom("P")})


def test_is_axiom_top_right():
    s = make_sequent([atom("P")], [TOP])
    assert is_axiom(s)


def test_is_axiom_bot_left():
    s = make_sequent([BOT], [atom("Q")])
    assert is_axiom(s)


def test_is_axiom_common_atom():
    p = atom("P")
    s = make_sequent([p, atom("Q")], [atom("R"), p])
    assert is_axiom(s)


def test_is_not_axiom():
    s = make_sequent([atom("P")], [atom("Q")])
    assert not is_axiom(s)


def test_axiom_requires_atomic_match_not_compound():
    f = conj(atom("P"), atom("Q"))
    s = make_sequent([f], [f])
    assert not is_axiom(s)


def test_sequent_equality_is_structural():
    s1 = make_sequent([atom("P"), atom("Q")], [atom("R")])
    s2 = make_sequent([atom("Q"), atom("P")], [atom("R")])
    assert s1 == s2


def test_pretty_sequent_deterministic():
    s = make_sequent([atom("Q"), atom("P")], [atom("R")])
    out = pretty_sequent(s)
    assert out == "P, Q |- R"
