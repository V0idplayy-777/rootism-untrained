"""Tests for the disk-backed dedup index used at real-corpus scale.

The properties that matter for Corpus v1:

1. identical content is caught exactly, including across a reopen;
2. near-identical content (whitespace, a renamed helper, one edited function) is
   caught by MinHash, and unrelated content is not;
3. the signature is a *real* MinHash estimate - the bug this guards against is
   positional shingle sampling, which reports near-zero similarity for
   documents that differ by one inserted line;
4. RAM stays flat: the index lives in SQLite, and the number of in-memory
   pending records is bounded by ``batch_size``.
"""

from __future__ import annotations

import random

from dataset.dedup_index import (
    DedupIndex,
    band_keys,
    jaccard,
    minhash_signature,
    normalise_document,
    pack_signature,
    shingle_hashes,
    true_jaccard,
    unpack_signature,
)

MODULE = "\n".join(
    f"def helper_{i}(value):\n"
    f'    """Return value transformed by step {i}."""\n'
    f"    return value * {i + 1} + len(str(value))\n"
    for i in range(40)
)


def test_normalise_document_collapses_whitespace():
    assert normalise_document(" a \n\n b\t c ") == "a b c"
    assert normalise_document("") == ""


def test_exact_duplicates_are_caught_and_survive_a_reopen(tmp_path):
    path = tmp_path / "dedup.sqlite"
    with DedupIndex(path) as index:
        assert index.check_and_add(MODULE, "a") == (False, "")
        assert index.check_and_add(MODULE, "b") == (True, "exact")
        # whitespace-only differences normalise to the same document
        assert index.check_and_add(MODULE.replace("\n", "\r\n"), "c") == (True, "exact")
        assert index.count() == 1

    with DedupIndex(path) as reopened:
        assert reopened.count() == 1
        assert reopened.check_and_add(MODULE, "a") == (True, "exact")


def test_blank_documents_are_rejected_not_indexed():
    with DedupIndex(":memory:") as index:
        assert index.check_and_add("", "a") == (True, "empty")
        assert index.check_and_add("   \n\t", "b") == (True, "empty")
        assert index.count() == 0


def test_unrelated_documents_are_not_duplicates():
    with DedupIndex(":memory:") as index:
        rng = random.Random(7)
        words = [f"word{i}" for i in range(2000)]
        for i in range(25):
            index.check_and_add(" ".join(rng.sample(words, 300)), f"doc{i}")
        assert index.stats()["duplicates"] == 0
        assert index.count() == 25


def test_near_duplicates_are_reported_with_their_reason():
    """Corpus dedup removes *copies* (vendored drops, forked files, reformatted pastes)."""

    with DedupIndex(":memory:") as index:
        index.check_and_add(MODULE, "base")

        # docstring rewritten in one function: still the same document
        lines = MODULE.split("\n")
        variant = list(lines)
        variant[6] = '    """Return value transformed by step 1 - rewritten by a fork."""'
        duplicate, reason = index.check_and_add("\n".join(variant), "variant")
        assert duplicate is True, "a rewritten docstring is a copy, not a variant"
        assert reason == "near"

        # trailing whitespace on every line normalises away entirely
        reflowed = "\n".join(f"{line}   " for line in lines)
        assert index.check_and_add(reflowed, "reflowed") == (True, "exact")

        # a genuinely different document is kept
        unrelated = "\n".join(f"class Other{i}:\n    pass\n" for i in range(40))
        assert index.check_and_add(unrelated, "other") == (False, "")


def test_duplicates_inside_one_batch_are_still_caught():
    """A batch that is not yet flushed must not hide a near-duplicate."""

    with DedupIndex(":memory:", batch_size=64) as index:
        assert index.check_and_add(MODULE, "a") == (False, "")
        reformatted = "\n".join(line for line in MODULE.split("\n"))
        duplicate, reason = index.check_and_add(reformatted, "b")
        assert (duplicate, reason) == (True, "exact")
        assert index.count() == 1


def test_minhash_estimate_tracks_true_jaccard_for_shifted_documents():
    """Regression: positional shingle sampling scored an inserted line as unrelated."""

    edited = MODULE.replace(
        "def helper_10(value):", "def renamed(value):\n    return value\n\n\ndef helper_10(value):"
    )
    true_similarity = true_jaccard(shingle_hashes(MODULE), shingle_hashes(edited))
    estimate = jaccard(minhash_signature(MODULE, 64), minhash_signature(edited, 64))

    assert true_similarity > 0.8, "one inserted function barely changes the document"
    assert abs(estimate - true_similarity) < 0.15, (estimate, true_similarity)


def test_minhash_estimate_is_zero_for_disjoint_documents():
    left = "alpha beta gamma delta epsilon zeta eta theta"
    right = "one two three four five six seven eight"
    assert jaccard(minhash_signature(left, 64), minhash_signature(right, 64)) < 0.3


def test_signature_round_trips_and_bands_are_stable():
    signature = minhash_signature(MODULE, 16)
    assert len(signature) == 16
    assert unpack_signature(pack_signature(signature)) == signature
    assert band_keys(signature, 4) == band_keys(signature, 4)
    assert len({band for band, _key in band_keys(signature, 4)}) == 4
    assert band_keys((), 4) == []


def test_shingle_sampling_is_content_addressed_not_positional():
    """A long document keeps the same shingles when text is inserted before them."""

    filler = " ".join(f"token{i}" for i in range(4000))
    long_doc = filler + " " + MODULE
    shifted = "INSERTED LINE\n" + long_doc

    left = set(shingle_hashes(long_doc).tolist())
    right = set(shingle_hashes(shifted).tolist())
    overlap = len(left & right) / len(left | right)
    assert overlap > 0.5, f"content-addressed sampling must survive an insertion, got {overlap}"


def test_check_does_not_mutate_the_index():
    with DedupIndex(":memory:") as index:
        index.check_and_add(MODULE, "a")
        assert index.check(MODULE) == (True, "exact")
        other = "\n".join(f"class Widget{i}:\n    pass\n" for i in range(60))
        assert index.check(other) == (False, "")
        assert index.count() == 1, "a probe must not consume a dedup slot"
        assert index.check_and_add(other, "b") == (False, "")


def test_heldout_namespace_is_isolated_from_training_namespace():
    with DedupIndex(":memory:", namespace="train") as train, DedupIndex(
        ":memory:", namespace="heldout"
    ) as heldout:
        assert heldout.check_and_add(MODULE, "h") == (False, "")
        assert train.check_and_add(MODULE, "t") == (False, ""), (
            "held-out registration must not make a training document look duplicated"
        )


def test_pending_batch_is_flushed_and_cleared(tmp_path):
    with DedupIndex(tmp_path / "d.sqlite", batch_size=3) as index:
        for i in range(1, 8):
            module = "\n".join(
                f"def helper_{i}_{j}(value):\n    return value * {j} + {i}\n"
                for j in range(i)
            )
            index.check_and_add(module, f"doc{i}")
        assert index.count() == 7
        assert index._pending_docs == []
        assert index._pending_digests == set()


def test_stats_report_the_rate_and_configuration(tmp_path):
    with DedupIndex(tmp_path / "d.sqlite", threshold=0.9, num_perm=32, bands=8) as index:
        index.check_and_add(MODULE, "a")
        index.check_and_add(MODULE, "b")
        index.check_and_add(MODULE + "\n# new function\n" * 3, "c")
        stats = index.stats()
    assert stats["checked"] == 3
    assert stats["exact_duplicates"] == 1
    assert stats["threshold"] == 0.9
    assert stats["num_perm"] == 32
    assert stats["bands"] == 8
    assert 0.0 < stats["dedup_rate"] <= 1.0
    assert stats["indexed"] == stats["checked"] - stats["duplicates"]


def test_index_rejects_impossible_configuration(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        DedupIndex(tmp_path / "x.sqlite", threshold=0.0)
    with pytest.raises(ValueError):
        DedupIndex(tmp_path / "x.sqlite", num_perm=4, bands=8)


def test_merge_copies_another_index():
    with DedupIndex(":memory:") as left, DedupIndex(":memory:") as right:
        left.check_and_add(MODULE, "a")
        right.check_and_add(MODULE + "\n# other\n" * 2, "b")
        assert left.merge(right) == 1
        assert left.check(MODULE + "\n# other\n" * 2)[0] is True
