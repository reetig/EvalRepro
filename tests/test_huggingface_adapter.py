"""Tests for the Hugging Face Dataset adapter."""

import pytest

# Skip the entire test module if datasets is not installed
datasets = pytest.importorskip("datasets")
from datasets import Dataset, Image, Features, Value

from evalrepro.adapters.huggingface import huggingface_dataset_source
from evalrepro.errors import AdapterError

@pytest.fixture
def synthetic_dataset() -> Dataset:
    """Provides a small, in-memory tabular dataset."""
    return Dataset.from_list([
        {"question_id": "q1", "prompt": "Hi", "truth": "Salut", "ignore_this": 123},
        {"question_id": "q2", "prompt": "Bye", "truth": "Au revoir", "ignore_this": 456},
    ])


def test_reproducible_across_revisions(synthetic_dataset: Dataset) -> None:
    """Different revisions with identical rows must produce the same semantic scope and records."""
    field_map = {"input": "prompt", "target": "truth"}
    
    baseline = huggingface_dataset_source(
        synthetic_dataset,
        dataset_name="translation-v1",
        field_map=field_map,
        id_column="question_id",
        revision="main"
    )
    
    candidate = huggingface_dataset_source(
        synthetic_dataset,
        dataset_name="translation-v1",
        field_map=field_map,
        id_column="question_id",
        revision="dev-branch-xyz"
    )
    
    # Scope and records match identically (Reproducible)
    assert baseline.identity == candidate.identity
    assert baseline.parameters == candidate.parameters
    assert baseline.records == candidate.records
    
    # Provenance correctly isolates the runtime revision differences
    assert baseline.provenance["explicit_revision"] == "main"
    assert candidate.provenance["explicit_revision"] == "dev-branch-xyz"


def test_scope_mismatch_on_identity_change(synthetic_dataset: Dataset) -> None:
    """Changing dataset_name or config triggers a scope_mismatch."""
    base = huggingface_dataset_source(
        synthetic_dataset, dataset_name="data-A", field_map={"input": "prompt"}
    )
    diff_name = huggingface_dataset_source(
        synthetic_dataset, dataset_name="data-B", field_map={"input": "prompt"}
    )
    diff_config = huggingface_dataset_source(
        synthetic_dataset, dataset_name="data-A", config_name="subset", field_map={"input": "prompt"}
    )
    
    assert base.identity != diff_name.identity
    assert base.identity != diff_config.identity


def test_semantic_drift_on_mutated_target(synthetic_dataset: Dataset) -> None:
    """Mutating a mapped target value changes the records (semantic_drift)."""
    field_map = {"input": "prompt", "target": "truth"}
    
    baseline = huggingface_dataset_source(
        synthetic_dataset, dataset_name="test", field_map=field_map
    )
    
    # Mutate a target value
    mutated_list = [
        {"question_id": "q1", "prompt": "Hi", "truth": "Salut", "ignore_this": 123},
        {"question_id": "q2", "prompt": "Bye", "truth": "MUTATED", "ignore_this": 456},
    ]
    mutated_dataset = Dataset.from_list(mutated_list)
    
    candidate = huggingface_dataset_source(
        mutated_dataset, dataset_name="test", field_map=field_map
    )
    
    assert baseline.identity == candidate.identity
    assert baseline.records != candidate.records


def test_missing_mapped_column_fails_closed(synthetic_dataset: Dataset) -> None:
    """Adapter must raise an error rather than silently defaulting to None."""
    with pytest.raises(AdapterError, match="Mapped source column 'missing_col'"):
        huggingface_dataset_source(
            synthetic_dataset,
            dataset_name="test",
            field_map={"input": "missing_col"}
        )


def test_unsupported_media_feature_fails_closed() -> None:
    """Adapter must reject Image or Audio features recursively to avoid dropping semantics."""
    # Create a synthetic dataset with an Image column
    features = Features({
        "prompt": Value("string"),
        "image_col": Image()
    })
    
    # Dummy data path (doesn't need to load an actual image, just define the schema)
    media_dataset = Dataset.from_list([{"prompt": "describe", "image_col": None}], features=features)
    
    with pytest.raises(AdapterError, match="Unsupported media feature at column 'image_col'"):
        huggingface_dataset_source(
            media_dataset,
            dataset_name="test",
            field_map={"input": "prompt"}
        )