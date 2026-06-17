"""
FrequencyMap — a thin wrapper around dict for counting categorical distributions.

This is a deliberate DSA exercise: implement the counting logic explicitly
rather than reaching for collections.Counter, so the mechanics are clear.
Once you understand it, Counter and value_counts() are just shortcuts.
"""

from __future__ import annotations

from typing import Hashable, Iterator, TypeVar

K = TypeVar("K", bound=Hashable)


class FrequencyMap:
    """
    A frequency counter for categorical values.

    Intentionally re-implements Counter's core functionality to practice
    dict-based data structures. Supports the full iteration protocol so
    it behaves like a dict in for loops and comprehensions.

    Example:
        fm = FrequencyMap(["cat", "dog", "cat", "cat", "bird"])
        fm["cat"]        # 3
        fm.most_common() # [("cat", 3), ("dog", 1), ("bird", 1)]
        fm.proportion("cat")  # 0.6
    """

    def __init__(self, iterable: list | None = None) -> None:
        self._counts: dict[Hashable, int] = {}
        self._total: int = 0
        if iterable:
            for item in iterable:
                self.add(item)

    def add(self, item: Hashable) -> None:
        """Increment count for item by 1."""
        self._counts[item] = self._counts.get(item, 0) + 1
        self._total += 1

    def add_many(self, iterable: list) -> None:
        """Increment counts for all items in iterable."""
        for item in iterable:
            self.add(item)

    def __getitem__(self, item: Hashable) -> int:
        return self._counts.get(item, 0)

    def __contains__(self, item: object) -> bool:
        return item in self._counts

    def __len__(self) -> int:
        return len(self._counts)

    def __iter__(self) -> Iterator:
        return iter(self._counts)

    def __repr__(self) -> str:
        top = self.most_common(5)
        return f"FrequencyMap({top!r}, total={self._total})"

    @property
    def total(self) -> int:
        """Total number of items added (not unique items)."""
        return self._total

    @property
    def unique_count(self) -> int:
        """Number of distinct values seen."""
        return len(self._counts)

    def proportion(self, item: Hashable) -> float:
        """Return fraction of all observations this item represents."""
        if self._total == 0:
            return 0.0
        return self._counts.get(item, 0) / self._total

    def most_common(self, n: int | None = None) -> list[tuple[Hashable, int]]:
        """
        Return the n most common (item, count) pairs, sorted descending.
        If n is None, return all pairs sorted by count.

        Uses a manual sort rather than heapq.nlargest so the logic is explicit.
        In week 4 we'll revisit this with a proper min-heap.
        """
        sorted_items = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n] if n is not None else sorted_items

    def to_dict(self) -> dict[Hashable, int]:
        return dict(self._counts)

    def summary(self) -> dict[str, object]:
        """Return a summary dict useful for EDA reporting."""
        if not self._counts:
            return {"total": 0, "unique": 0, "top": None, "top_proportion": 0.0}
        top_item, top_count = self.most_common(1)[0]
        return {
            "total": self._total,
            "unique": self.unique_count,
            "top": top_item,
            "top_proportion": round(top_count / self._total, 4),
        }
