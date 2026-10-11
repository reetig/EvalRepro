"""Tests for Hugging Face Dataset adapter using synthetic in-memory fixtures."""

import pytest
from datasets import Dataset, Image

from evalrepro.errors import AdapterError
from evalrepro.adapters.huggingface import dataset_to_snapshot_source
from evalrepro.compare import compare_manifest_data
from evalrepro.manifest import build_manifest
import datasets.fingerprint
# Bypass the broken dill pickler in Python 3.13+
datasets.fingerprint.generate_fingerprint = lambda *args, **kwargs: "mock_fingerprint"

@pytest.fixture
def base_dataset():
    """Create a synthetic text dataset in memory."""
    data = {
        "item_id": ["1", "2", "3"],
        "question": ["2+2?", "3+3?", "4+4?"],
        "answers": ["4", "6", "8"],
        "unused": ["a", "b", "c"],
    }
    return Dataset.from_dict(data)


def test_baseline_reproducible_match(base_dataset):
    source1 = dataset_to_snapshot_source(
        base_dataset,
        dataset_name="synthetic-math",
        field_map={"input": "question", "target": "answers"},
        id_column="item_id",
    )
    source2 = dataset_to_snapshot_source(
        base_dataset,
        dataset_name="synthetic-math",
        field_map={"input": "question", "target": "answers"},
        id_column="item_id",
    )

    m1 = build_manifest(source1)
    m2 = build_manifest(source2)
    report = compare_manifest_data(m1, m2)

    assert report.verdict == "reproducible"
    assert source1.fields == ("input", "target")
    assert source1.id_field == "id"


def test_target_semantic_drift(base_dataset):
    modified_data = {
        "item_id": ["1", "2", "3"],
        "question": ["2+2?", "3+3?", "4+4?"],
        "answers": ["4", "6", "999"],  # Modified target
        "unused": ["a", "b", "c"],
    }
    mod_dataset = Dataset.from_dict(modified_data)

    s1 = dataset_to_snapshot_source(
        base_dataset,
        dataset_name="synthetic-math",
        field_map={"input": "question", "target": "answers"},
    )
    s2 = dataset_to_snapshot_source(
        mod_dataset,
        dataset_name="synthetic-math",
        field_map={"input": "question", "target": "answers"},
    )

    m1 = build_manifest(s1)
    m2 = build_manifest(s2)
    report = compare_manifest_data(m1, m2)

    assert report.verdict == "semantic_drift"


def test_order_drift(base_dataset):
    shuffled_data = {
        "item_id": ["3", "1", "2"],
        "question": ["4+4?", "2+2?", "3+3?"],
        "answers": ["8", "4", "6"],
        "unused": ["c", "a", "b"],
    }
    shuffled_ds = Dataset.from_dict(shuffled_data)

    s1 = dataset_to_snapshot_source(
        base_dataset,
        dataset_name="synthetic-math",
        field_map={"input": "question", "target": "answers"},
    )
    s2 = dataset_to_snapshot_source(
        shuffled_ds,
        dataset_name="synthetic-math",
        field_map={"input": "question", "target": "answers"},
    )

    m1 = build_manifest(s1)
    m2 = build_manifest(s2)
    report = compare_manifest_data(m1, m2)

    assert report.verdict == "order_drift"


def test_actual_partial_coverage_manifest(base_dataset):
    s1 = dataset_to_snapshot_source(
        base_dataset,
        dataset_name="synthetic-math",
        field_map={"input": "question", "target": "answers"},
    )

    m1 = build_manifest(s1)
    m2 = build_manifest(s1, sample_limit=2)
    report = compare_manifest_data(m1, m2)

    assert report.verdict == "coverage_mismatch"


def test_missing_mapped_column_raises_adapter_error(base_dataset):
    with pytest.raises(AdapterError, match="not found in dataset"):
        dataset_to_snapshot_source(
            base_dataset,
            dataset_name="synthetic-math",
            field_map={"input": "non_existent_col"},
        )


def test_unsupported_image_feature_raises_adapter_error():
    # Synthetic dataset with an Image feature schema
    ds = Dataset.from_dict({"img": ["fake_path"]}).cast_column("img", Image())
    with pytest.raises(AdapterError, match="Unsupported feature type"):
        dataset_to_snapshot_source(
            ds,
            dataset_name="synthetic-img",
            field_map={"input": "img"},
        )
