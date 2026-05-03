"""
Memo cache. CLOSED and IN_PROGRESS only; never stores FAILED.

The history component of the fingerprint is needed for soundness: without it,
two sequents that differ only in instantiation history would alias.

"""

from collections import OrderedDict
from typing import Optional


CLOSED = "CLOSED"
IN_PROGRESS = "IN_PROGRESS"


def fingerprint(seq, history: dict) -> tuple:

    """

    Hashable triple over antecedent, succedent, and instantiation history.
    The history component is needed for soundness with retaining quantifiers.

    """
    hist = frozenset(
        (formula, frozenset(terms))
        for formula, terms in history.items()
        if terms
    )
    return (seq.antecedent, seq.succedent, hist)


class MemoisationCache:
    __slots__ = ("_closed", "_in_progress", "_max", "hits", "misses",
                 "stores", "evictions")

    def __init__(self, max_entries: int = 10_000_000):
        self._closed: "OrderedDict[tuple, object]" = OrderedDict()
        self._in_progress: set = set()
        self._max = max_entries
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.evictions = 0

    def get_closed(self, fp: tuple):
        proof = self._closed.get(fp)
        if proof is not None:
            self._closed.move_to_end(fp)
            self.hits += 1
            return proof
        self.misses += 1
        return None

    def store_closed(self, fp: tuple, proof) -> None:
        if fp in self._closed:
            self._closed.move_to_end(fp)
            self._closed[fp] = proof
            return
        self._closed[fp] = proof
        self.stores += 1
        while len(self._closed) > self._max:
            self._closed.popitem(last=False)
            self.evictions += 1

    def mark_in_progress(self, fp: tuple) -> None:
        self._in_progress.add(fp)

    def unmark_in_progress(self, fp: tuple) -> None:
        self._in_progress.discard(fp)

    def is_in_progress(self, fp: tuple) -> bool:
        return fp in self._in_progress

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def closed_size(self) -> int:
        return len(self._closed)


def cache_lookup(cache: Optional[MemoisationCache], fp: tuple, ancestor_path: frozenset):

    """Returns (status, proof). status is CLOSED_HIT, CYCLE, or MISS."""

    if cache is None:
        if fp in ancestor_path:
            return ("CYCLE", None)
        return ("MISS", None)
    proof = cache.get_closed(fp)
    if proof is not None:
        return ("CLOSED_HIT", proof)
    if fp in ancestor_path:
        return ("CYCLE", None)
    return ("MISS", None)
