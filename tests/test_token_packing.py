"""
Tests for token packing
"""

import pytest
from nlpy.utils.preprocess import (
    _pack_complete,
    _token_packing)

@pytest.fixture
def candidates():
    examples = [
        "This is an example list".split(),
        "This is another example list".split(),
        "Too short example".split()]
    return examples

def test_pack_complete(candidates):
    completed_tokens = []
    completed_tokens, remainder = _pack_complete(
        completed_tokens, candidates, len_threshold=5)
    assert completed_tokens == candidates[0:2]
    assert remainder[0] == candidates[-1]


def test_token_packing(candidates):
    completed_tokens, curr_tokens = _token_packing(
        candidates[0],
        candidates[1],
        max_len = 7,
        min_len = 2,
        packing_tolerance=1)
    assert completed_tokens == [
        candidates[0] + ["<SEP>"] + candidates[1][:1]]
    assert curr_tokens == candidates[1][1:]

    completed_tokens, curr_tokens = _token_packing(
        candidates[0],
        candidates[1],
        max_len = 7,
        min_len = 2,
        packing_tolerance=2)
    assert completed_tokens == [
        candidates[0], candidates[1]]
    assert curr_tokens == []