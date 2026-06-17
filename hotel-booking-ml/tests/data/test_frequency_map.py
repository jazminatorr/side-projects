"""
Tests for FrequencyMap.

As an SDET you probably find tests easy to write — use that muscle.
These tests serve as executable documentation of how the class works.
"""

import pytest
from src.utils.frequency_map import FrequencyMap


class TestFrequencyMapBasics:
    def test_empty_map(self):
        fm = FrequencyMap()
        assert len(fm) == 0
        assert fm.total == 0
        assert fm["anything"] == 0

    def test_single_item(self):
        fm = FrequencyMap(["cat"])
        assert fm["cat"] == 1
        assert fm.total == 1
        assert fm.unique_count == 1

    def test_multiple_same_item(self):
        fm = FrequencyMap(["cat", "cat", "cat"])
        assert fm["cat"] == 3
        assert fm.total == 3
        assert fm.unique_count == 1

    def test_multiple_different_items(self):
        fm = FrequencyMap(["cat", "dog", "bird"])
        assert fm["cat"] == 1
        assert fm["dog"] == 1
        assert fm["bird"] == 1
        assert fm.total == 3
        assert fm.unique_count == 3

    def test_missing_key_returns_zero(self):
        fm = FrequencyMap(["cat", "dog"])
        assert fm["fish"] == 0

    def test_contains(self):
        fm = FrequencyMap(["cat", "dog"])
        assert "cat" in fm
        assert "fish" not in fm


class TestFrequencyMapMostCommon:
    def setup_method(self):
        self.fm = FrequencyMap(["cat", "dog", "cat", "cat", "bird", "dog"])

    def test_most_common_all(self):
        result = self.fm.most_common()
        assert result[0] == ("cat", 3)
        assert result[1] == ("dog", 2)
        assert result[2] == ("bird", 1)

    def test_most_common_n(self):
        result = self.fm.most_common(2)
        assert len(result) == 2
        assert result[0] == ("cat", 3)

    def test_most_common_n_larger_than_keys(self):
        result = self.fm.most_common(100)
        assert len(result) == 3

    def test_proportion(self):
        # 3 cats out of 6 total = 0.5
        assert self.fm.proportion("cat") == pytest.approx(0.5)
        assert self.fm.proportion("dog") == pytest.approx(2 / 6)
        assert self.fm.proportion("fish") == pytest.approx(0.0)

    def test_proportion_empty(self):
        fm = FrequencyMap()
        assert fm.proportion("anything") == 0.0


class TestFrequencyMapAddMany:
    def test_add_many(self):
        fm = FrequencyMap()
        fm.add_many(["a", "b", "a", "c"])
        assert fm["a"] == 2
        assert fm["b"] == 1
        assert fm.total == 4


class TestFrequencyMapSummary:
    def test_summary_non_empty(self):
        fm = FrequencyMap(["x", "x", "y"])
        s = fm.summary()
        assert s["total"] == 3
        assert s["unique"] == 2
        assert s["top"] == "x"
        assert s["top_proportion"] == pytest.approx(2 / 3, rel=1e-3)

    def test_summary_empty(self):
        fm = FrequencyMap()
        s = fm.summary()
        assert s["total"] == 0
        assert s["top"] is None


class TestFrequencyMapIteration:
    def test_iterable(self):
        fm = FrequencyMap(["a", "b", "c"])
        keys = list(fm)
        assert set(keys) == {"a", "b", "c"}

    def test_to_dict(self):
        fm = FrequencyMap(["a", "a", "b"])
        d = fm.to_dict()
        assert d == {"a": 2, "b": 1}
