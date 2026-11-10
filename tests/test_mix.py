from pathlib import Path

import pytest

from src.data import read_jsonl
from src.mix import mix

HAND20 = read_jsonl(Path("tests/fixtures/handlabelled20.jsonl"))
REAL = [{**r, "split": "train"} for r in HAND20]  # stands in for the real-train pool
SYNTHETIC = [{"id": f"syn-{i}"} for i in range(900)]


def ids(records):
    return {r["id"] for r in records if not r["id"].startswith("syn-")}


def test_no_real_addresses_leaves_the_synthetic_set_alone():
    assert mix(SYNTHETIC, REAL, k=0, seed=42) == SYNTHETIC


def test_subsets_are_nested_within_a_seed_and_differ_between_seeds():
    five, ten = ids(mix(SYNTHETIC, REAL, 5, seed=42)), ids(mix(SYNTHETIC, REAL, 10, seed=42))
    assert len(five) == 5 and len(ten) == 10 and five < ten
    assert ids(mix(SYNTHETIC, REAL, 10, seed=43)) != ten


def test_real_addresses_are_repeated_to_about_a_tenth_of_the_file():
    for k in (5, 20):
        mixed = mix(SYNTHETIC, REAL, k, seed=42)
        real_rows = [r for r in mixed if not r["id"].startswith("syn-")]
        assert len(mixed) == 900 + len(real_rows) and abs(len(real_rows) / len(mixed) - 0.10) < 0.006
        assert len(real_rows) % k == 0  # every chosen address the same number of times


def test_subsets_stay_nested_in_a_pool_of_real_size():
    pool = [{"id": f"real-{i}", "split": "train"} for i in range(500)]
    chosen = [ids(mix(SYNTHETIC, pool, k, seed=42)) for k in (25, 50, 100, 200)]
    assert [len(c) for c in chosen] == [25, 50, 100, 200] and chosen[0] < chosen[1] < chosen[2] < chosen[3]


def test_asking_for_more_than_the_pool_holds_is_an_error():
    with pytest.raises(ValueError, match="20"):
        mix(SYNTHETIC, REAL, 21, seed=42)


def test_only_train_split_addresses_may_be_mixed_in():
    with pytest.raises(ValueError, match="train split"):
        mix(SYNTHETIC, HAND20, 5, seed=42)  # the fixture rows are marked dev
