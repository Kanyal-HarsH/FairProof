from logic.syntax import (
    var, const, atom, forall,
)
from logic.sequent import make_sequent
from prover.memoisation import MemoisationCache, fingerprint, cache_lookup


def test_fingerprint_distinguishes_history():
    """Two sequents that differ only in
    instantiation history must have distinct fingerprints. Without this,
    the cache can alias distinct quantifier-instantiation states.
    """
    fa = forall("x", atom("P", (var("x"),)))
    seq = make_sequent([fa], [atom("Q")])
    h1 = {fa: frozenset({const("a")})}
    h2 = {fa: frozenset({const("a"), const("b")})}
    assert fingerprint(seq, h1) != fingerprint(seq, h2)


def test_fingerprint_ignores_empty_history_entries():
    fa = forall("x", atom("P", (var("x"),)))
    seq = make_sequent([fa], [])
    assert fingerprint(seq, {}) == fingerprint(seq, {fa: frozenset()})


def test_fingerprint_same_sequent_same_history():
    fa = forall("x", atom("P", (var("x"),)))
    seq1 = make_sequent([fa, atom("Q")], [atom("R")])
    seq2 = make_sequent([atom("Q"), fa], [atom("R")])  # frozenset order
    assert fingerprint(seq1, {}) == fingerprint(seq2, {})


def test_cache_store_and_get():
    cache = MemoisationCache()
    fp = ("k1",)
    proof = "PROOF_OBJ"
    cache.store_closed(fp, proof)
    assert cache.get_closed(fp) == proof
    assert cache.hits == 1


def test_cache_lookup_states():
    cache = MemoisationCache()
    fp = ("k",)
    status, _ = cache_lookup(cache, fp, frozenset())
    assert status == "MISS"

    cache.mark_in_progress(fp)
    status, _ = cache_lookup(cache, fp, frozenset({fp}))
    assert status == "CYCLE"
    status, _ = cache_lookup(cache, fp, frozenset())
    assert status == "MISS"  # in_progress but not on path -> miss

    cache.unmark_in_progress(fp)
    cache.store_closed(fp, "PROOF")
    status, proof = cache_lookup(cache, fp, frozenset())
    assert status == "CLOSED_HIT"
    assert proof == "PROOF"


def test_cache_lru_eviction():
    cache = MemoisationCache(max_entries=3)
    cache.store_closed(("a",), 1)
    cache.store_closed(("b",), 2)
    cache.store_closed(("c",), 3)
    assert cache.closed_size() == 3
    cache.store_closed(("d",), 4)
    assert cache.closed_size() == 3
    assert cache.evictions == 1
    assert cache.get_closed(("a",)) is None  # evicted


def test_cache_no_failed_storage():
    """The cache only exposes CLOSED + IN_PROGRESS. There is no FAILED state
    because failure under one budget does not imply failure under a larger one
    (Astrachan & Stickel 1992; Gore & Nguyen 2013).
    """
    cache = MemoisationCache()
    assert not hasattr(cache, "store_failed")
    assert not hasattr(cache, "FAILED")
