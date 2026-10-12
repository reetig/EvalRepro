"""Hugging Face Dataset adapter."""

from __future__ import annotations

from typing import Any

from evalrepro.adapters.base import SnapshotSource
from evalrepro.errors import AdapterError
from evalrepro.runtime import runtime_versions

ADAPTER_CONTRACT_VERSION = 1


def _validate_features(features: Any, path: str = "") -> None:
    """
    Recursively inspect features to enforce a strict allowlist.
    Note: Unmapped unsupported features will intentionally cause rejection.
    """
    try:
        import datasets  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ImportError:
        return

    if isinstance(features, dict):
        for key, value in features.items():
            _validate_features(value, f"{path}.{key}" if path else key)
    elif isinstance(features, datasets.Sequence):
        _validate_features(features.feature, f"{path}[]")
    elif isinstance(features, (datasets.Value, datasets.ClassLabel)):
        pass
    else:
        raise AdapterError(
            f"Unsupported feature type at column '{path}': {type(features).__name__}. "
            "Only Value, ClassLabel, and nested dict/Sequence are supported."
        )


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
    """Load an in-memory Hugging Face Dataset into a SnapshotSource."""
    try:
        import datasets  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ImportError as exc:
        raise AdapterError(
            "The 'datasets' package is required for this adapter. "
            "Install with: pip install evalrepro[huggingface]"
        ) from exc

    if not isinstance(dataset, datasets.Dataset):
        raise AdapterError(f"Expected datasets.Dataset, got {type(dataset).__name__}")

    # Strict type and empty-value validation for all parameters
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise AdapterError("dataset_name must be a non-empty string.")

    for param_name, param_val in [
        ("config_name", config_name),
        ("split_name", split_name),
        ("id_column", id_column),
        ("revision", revision),
    ]:
        if param_val is not None and (not isinstance(param_val, str) or not param_val.strip()):
            raise AdapterError(f"{param_name} must be a non-empty string if provided.")

    if not isinstance(field_map, dict) or not field_map:
        raise AdapterError("field_map must be a non-empty dictionary.")

    valid_target_fields = {"input", "target", "choices", "metadata"}
    canonical_map = {}

    for target, source in field_map.items():
        if not isinstance(target, str) or not isinstance(source, str):
            raise AdapterError(
                f"field_map keys and values must be strings. Got: {target!r}: {source!r}"
            )
        if not target.strip() or not source.strip():
            raise AdapterError(f"Malformed field_map entry: {target!r}: {source!r}")
        if target not in valid_target_fields:
            raise AdapterError(
                f"Invalid target field '{target}'. Must be one of: {', '.join(valid_target_fields)}"
            )
        canonical_map[target] = source

    available_columns = set(dataset.column_names)
    for target, source in canonical_map.items():
        if source not in available_columns:
            raise AdapterError(
                f"Mapped source column '{source}' for target '{target}' not found in dataset."
            )

    if id_column and id_column not in available_columns:
        raise AdapterError(f"ID column '{id_column}' not found in dataset.")

    _validate_features(dataset.features)

    records = []
    for row in dataset:
        record: dict[str, Any] = {}
        if id_column:
            record["id"] = row[id_column]

        for target_field, source_column in canonical_map.items():
            record[target_field] = row[source_column]

        records.append(record)

    provenance: dict[str, Any] = {
        "source": "huggingface-dataset",
    }
    if revision:
        provenance["explicit_revision"] = revision

    if hasattr(dataset, "info") and dataset.info:
        if dataset.info.version:
            provenance["dataset_info_version"] = str(dataset.info.version)
        if dataset.info.dataset_size:
            provenance["dataset_size_bytes"] = dataset.info.dataset_size

    canonical_fields = tuple(sorted(canonical_map.keys()))

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
            "field_map": canonical_map,
            "id_column": id_column,
        },
        records=records,
        declared_count=len(dataset),
        fields=canonical_fields,
        id_field="id" if id_column else None,
        runtime=runtime_versions(("datasets", "huggingface_hub")),
        provenance=provenance,
    )
