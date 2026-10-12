"""Tests for the Hugging Face Dataset adapter."""

import pytest

from evalrepro.adapters.huggingface import huggingface_dataset_source
from evalrepro.compare import Verdict, compare_manifest_data
from evalrepro.errors import AdapterError
from evalrepro.manifest import build_manifest

# Skip the entire test module if datasets is not installed
pytest.importorskip("datasets")
from datasets import ClassLabel, Dataset, Features, Image, Sequence, Value


@pytest.fixture
def base_dataset() -> Dataset:
    """Provides a small, in-memory tabular dataset."""
    return Dataset.from_list(
        [
            {"qid": "1", "input_col": "A", "target_col": "1", "ignored": True},
            {"qid": "2", "input_col": "B", "target_col": "2", "ignored": False},
        ]
    )


def test_reproducible_across_revisions(base_dataset: Dataset) -> None:
    """Different revisions with identical rows must produce 'reproducible' verdict."""
    source1 = huggingface_dataset_source(
        base_dataset, dataset_name="test", field_map={"input": "input_col"}, revision="v1"
    )
    source2 = huggingface_dataset_source(
        base_dataset, dataset_name="test", field_map={"input": "input_col"}, revision="v2"
    )

    manifest1 = build_manifest(source1)
    manifest2 = build_manifest(source2)

    assert compare_manifest_data(manifest1, manifest2).verdict is Verdict.REPRODUCIBLE

    # Provenance correctly isolates the runtime revision differences
    assert source1.provenance["explicit_revision"] == "v1"
    assert source2.provenance["explicit_revision"] == "v2"


def test_scope_mismatch(base_dataset: Dataset) -> None:
    """Changing dataset_name or mapping triggers a scope_mismatch."""
    source_base = huggingface_dataset_source(
        base_dataset, dataset_name="test-a", field_map={"input": "input_col"}
    )
    source_name = huggingface_dataset_source(
        base_dataset, dataset_name="test-b", field_map={"input": "input_col"}
    )
    source_map = huggingface_dataset_source(
        base_dataset,
        dataset_name="test-a",
        field_map={"input": "input_col", "target": "target_col"},
    )

    assert (
        compare_manifest_data(build_manifest(source_base), build_manifest(source_name)).verdict
        is Verdict.SCOPE_MISMATCH
    )
    assert (
        compare_manifest_data(build_manifest(source_base), build_manifest(source_map)).verdict
        is Verdict.SCOPE_MISMATCH
    )


def test_semantic_drift_same_count(base_dataset: Dataset) -> None:
    """Mutating a mapped target value changes the records, producing semantic_drift."""
    source_base = huggingface_dataset_source(
        base_dataset, dataset_name="test", field_map={"input": "input_col", "target": "target_col"}
    )

    # Replace one row's value (same count, different membership)
    mutated = Dataset.from_list(
        [
            {"qid": "1", "input_col": "A", "target_col": "1", "ignored": True},
            {"qid": "2", "input_col": "B", "target_col": "MUTATED", "ignored": False},
        ]
    )
    source_mutated = huggingface_dataset_source(
        mutated, dataset_name="test", field_map={"input": "input_col", "target": "target_col"}
    )

    assert (
        compare_manifest_data(build_manifest(source_base), build_manifest(source_mutated)).verdict
        is Verdict.SEMANTIC_DRIFT
    )


def test_coverage_mismatch_and_sample_limit(base_dataset: Dataset) -> None:
    """Using sample_limit must trigger a coverage_mismatch against the full set."""
    source = huggingface_dataset_source(
        base_dataset, dataset_name="test", field_map={"input": "input_col"}
    )

    manifest_full = build_manifest(source)
    manifest_partial = build_manifest(source, sample_limit=1)

    assert (
        compare_manifest_data(manifest_full, manifest_partial).verdict is Verdict.COVERAGE_MISMATCH
    )


def test_id_preview_flag(base_dataset: Dataset) -> None:
    """Adapter id_column powers previews, which can be safely stripped"""
    source = huggingface_dataset_source(
        base_dataset, dataset_name="test", field_map={"input": "input_col"}, id_column="qid"
    )
    manifest = build_manifest(source, include_id_preview=False)

    assert manifest["samples"]["id_preview"]["first"] == []
    assert manifest["samples"]["id_preview"]["last"] == []


def test_strict_inputs_and_canonicalization(base_dataset: Dataset) -> None:
    """Adapter must reject empty maps and canonicalize map ordering."""
    with pytest.raises(AdapterError, match="must be a non-empty string"):
        huggingface_dataset_source(base_dataset, dataset_name="", field_map={"input": "input_col"})

    with pytest.raises(AdapterError, match="cannot be empty"):
        huggingface_dataset_source(base_dataset, dataset_name="test", field_map={})

    # Test reverse dictionary order has same scope
    s1 = huggingface_dataset_source(
        base_dataset, dataset_name="test", field_map={"input": "input_col", "target": "target_col"}
    )
    s2 = huggingface_dataset_source(
        base_dataset, dataset_name="test", field_map={"target": "target_col", "input": "input_col"}
    )
    assert s1.parameters["field_map"] == s2.parameters["field_map"]
    assert s1.fields == s2.fields


def test_feature_allowlist() -> None:
    """Adapter strictly allows safe types and rejects opaque media features."""
    # Explicitly allowed: Value, ClassLabel, nested Sequences and Dicts
    valid_features = Features(
        {
            "prompt": Value("string"),
            "label": ClassLabel(names=["neg", "pos"]),
            "nested": Sequence(Value("string")),
        }
    )
    ds_valid = Dataset.from_list(
        [{"prompt": "A", "label": 0, "nested": ["B"]}], features=valid_features
    )
    # Should not raise
    huggingface_dataset_source(ds_valid, dataset_name="test", field_map={"input": "prompt"})

    # Rejected: Image
    invalid_features = Features({"prompt": Value("string"), "img": Sequence(Image())})
    ds_invalid = Dataset.from_list([{"prompt": "A", "img": []}], features=invalid_features)

    with pytest.raises(AdapterError, match="Unsupported feature type at column 'img\\[\\]'"):
        huggingface_dataset_source(ds_invalid, dataset_name="test", field_map={"input": "prompt"})


def test_invalid_dataset_type_fails() -> None:
    """Adapter must reject standard Python lists or other non-Dataset objects."""
    with pytest.raises(AdapterError, match=r"Expected a non-streaming datasets\.Dataset"):
        huggingface_dataset_source(
            ["not", "a", "dataset"],  # type: ignore
            dataset_name="test",
            field_map={"input": "prompt"},
        )
