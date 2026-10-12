"""Tests for the Hugging Face Dataset adapter."""

import pytest

from evalrepro.adapters.huggingface import huggingface_dataset_source
from evalrepro.errors import AdapterError

# Skip the entire test module if datasets is not installed
pytest.importorskip("datasets")
from datasets import Dataset, Features, Image, Sequence, Value


@pytest.fixture
def synthetic_dataset() -> Dataset:
    """Provides a small, in-memory tabular dataset."""
    return Dataset.from_list(
        [
            {"question_id": "q1", "prompt": "Hi", "truth": "Salut", "ignore_this": 123},
            {"question_id": "q2", "prompt": "Bye", "truth": "Au revoir", "ignore_this": 456},
        ]
    )


def test_reproducible_across_revisions(synthetic_dataset: Dataset) -> None:
    """Different revisions with identical rows must produce the same semantic scope and records."""
    field_map = {"input": "prompt", "target": "truth"}

    baseline = huggingface_dataset_source(
        synthetic_dataset,
        dataset_name="translation-v1",
        field_map=field_map,
        id_column="question_id",
        revision="main",
    )

    candidate = huggingface_dataset_source(
        synthetic_dataset,
        dataset_name="translation-v1",
        field_map=field_map,
        id_column="question_id",
        revision="dev-branch-xyz",
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
        synthetic_dataset,
        dataset_name="data-A",
        config_name="subset",
        field_map={"input": "prompt"},
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
            synthetic_dataset, dataset_name="test", field_map={"input": "missing_col"}
        )


def test_unsupported_media_feature_fails_closed() -> None:
    """Adapter must reject Image or Audio features recursively to avoid dropping semantics."""
    # Create a synthetic dataset with an Image column
    features = Features({"prompt": Value("string"), "image_col": Image()})

    # Dummy data path (doesn't need to load an actual image, just define the schema)
    media_dataset = Dataset.from_list(
        [{"prompt": "describe", "image_col": None}], features=features
    )

    with pytest.raises(AdapterError, match="Unsupported media feature at column 'image_col'"):
        huggingface_dataset_source(
            media_dataset, dataset_name="test", field_map={"input": "prompt"}
        )


def test_invalid_target_field_fails(synthetic_dataset: Dataset) -> None:
    """Adapter must reject field maps that use unsupported target keys."""
    with pytest.raises(AdapterError, match="Invalid target field 'bad_target'"):
        huggingface_dataset_source(
            synthetic_dataset, dataset_name="test", field_map={"bad_target": "prompt"}
        )


def test_invalid_id_column_fails(synthetic_dataset: Dataset) -> None:
    """Adapter must reject an id_column that doesn't exist in the dataset."""
    with pytest.raises(AdapterError, match="ID column 'bad_id' not found"):
        huggingface_dataset_source(
            synthetic_dataset,
            dataset_name="test",
            field_map={"input": "prompt"},
            id_column="bad_id",
        )


def test_invalid_dataset_type_fails() -> None:
    """Adapter must reject standard Python lists or other non-Dataset objects."""
    with pytest.raises(AdapterError, match=r"Expected a non-streaming datasets\.Dataset"):
        huggingface_dataset_source(
            ["not", "a", "dataset"],  # type: ignore
            dataset_name="test",
            field_map={"input": "prompt"},
        )


def test_unsupported_media_in_sequence_fails_closed() -> None:
    """Adapter must reject media even when nested in a Sequence."""
    features = Features({"prompt": Value("string"), "image_list": Sequence(Image())})

    media_dataset = Dataset.from_list([{"prompt": "describe", "image_list": []}], features=features)

    with pytest.raises(
        AdapterError, match="Unsupported media feature at column 'image_list\\[\\]'"
    ):
        huggingface_dataset_source(
            media_dataset, dataset_name="test", field_map={"input": "prompt"}
        )


def test_provenance_extracts_dataset_info(synthetic_dataset: Dataset) -> None:
    """Adapter must extract version and size from dataset.info if available."""
    # Manually inject info metadata into our synthetic dataset
    synthetic_dataset.info.version = "2.0.5"
    synthetic_dataset.info.dataset_size = 42000

    result = huggingface_dataset_source(
        synthetic_dataset, dataset_name="test", field_map={"input": "prompt"}
    )

    assert result.provenance["dataset_info_version"] == "2.0.5"
    assert result.provenance["dataset_size_bytes"] == 42000
