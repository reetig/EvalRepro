"""Hugging Face Dataset adapter."""

from __future__ import annotations

from typing import Any

from evalrepro.adapters.base import SnapshotSource
from evalrepro.errors import AdapterError
from evalrepro.runtime import runtime_versions

ADAPTER_CONTRACT_VERSION = 1


def _validate_features(features: Any, path: str = "") -> None:
    """Recursively inspect features to fail closed on unsupported media or opaque types."""
    import datasets

    if isinstance(features, (datasets.Image, datasets.Audio)):
        raise AdapterError(
            f"Unsupported media feature at column '{path}': {type(features).__name__}. "
            "Media features are not supported in version 1."
        )

    if isinstance(features, dict):
        for key, value in features.items():
            _validate_features(value, f"{path}.{key}" if path else key)
    elif isinstance(features, datasets.Sequence):
        _validate_features(features.feature, f"{path}[]")
    # Base types like datasets.Value and datasets.ClassLabel are allowed.


def huggingface_dataset_source(
    dataset: Any,
    *,
    dataset_name: str,
    field_map: dict[str, str],
    config_name: str | None = None,
    split_name: str | None = None,
    id_column: str | None = None,
    revision: str | None = None,
) -> SnapshotSource:
    """
    Load an in-memory Hugging Face Dataset into a SnapshotSource.

    Args:
        dataset: A loaded datasets.Dataset object.
        dataset_name: The stable semantic identity of the dataset.
        field_map: Mapping of EvalRepro fields (e.g., 'input', 'target') to dataset columns.
        config_name: The dataset configuration/subset name, if applicable.
        split_name: The dataset split name, if applicable.
        id_column: The dataset column to use for compact previews.
        revision: The explicit caller-provided revision.
    """
    try:
        import datasets
    except ImportError as exc:
        raise AdapterError(
            "The 'datasets' package is required for this adapter. "
            "Install with: pip install evalrepro[huggingface]"
        ) from exc

    if not isinstance(dataset, datasets.Dataset):
        raise AdapterError(
            f"Expected a non-streaming datasets.Dataset, got {type(dataset).__name__}"
        )

    available_columns = set(dataset.column_names)

    # Strictly enforce field mapping
    valid_target_fields = {"input", "target", "choices", "metadata"}
    for target, source in field_map.items():
        if target not in valid_target_fields:
            raise AdapterError(
                f"Invalid target field '{target}'. Must be one of: {', '.join(valid_target_fields)}"
            )
        if source not in available_columns:
            raise AdapterError(
                f"Mapped source column '{source}' for target '{target}' not found in dataset."
            )

    if id_column and id_column not in available_columns:
        raise AdapterError(f"ID column '{id_column}' not found in dataset.")

    # Fail closed on unsupported features
    _validate_features(dataset.features)

    # Map records
    records = []
    for row in dataset:
        record: dict[str, Any] = {}
        if id_column:
            record["id"] = row[id_column]

        for target_field, source_column in field_map.items():
            record[target_field] = row[source_column]

        records.append(record)

    # Extract safe provenance metadata
    provenance: dict[str, Any] = {
        "source": "huggingface-dataset",
        "explicit_revision": revision,
        "huggingface_datasets_version": datasets.__version__,
    }

    if hasattr(dataset, "info") and dataset.info:
        if dataset.info.version:
            provenance["dataset_info_version"] = str(dataset.info.version)
        if dataset.info.dataset_size:
            provenance["dataset_size_bytes"] = dataset.info.dataset_size

    return SnapshotSource(
        adapter="huggingface-dataset",
        identity={
            "dataset_name": dataset_name,
            "config": config_name,
            "split": split_name,
        },
        parameters={
            "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
            "field_map_version": 1,
            "field_map": field_map,
        },
        records=records,
        declared_count=len(dataset),
        fields=tuple(field_map.keys()),
        id_field="id" if id_column else None,
        runtime=runtime_versions(),
        provenance=provenance,
    )
