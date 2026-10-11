"""Hugging Face Datasets adapter for EvalRepro."""

from collections.abc import Mapping
from typing import Any

from evalrepro.adapters.base import SnapshotSource
from evalrepro.errors import AdapterError

try:
    import datasets  # type: ignore[import-untyped]
    import huggingface_hub
    from datasets import Dataset, Features, IterableDataset
except ImportError:  # pragma: no cover
    datasets = None
    Dataset = Any
    Features = Any
    IterableDataset = Any
    huggingface_hub = None  # type: ignore[assignment]


_SUPPORTED_FEATURE_TYPES = (
    "Value",
    "ClassLabel",
    "Sequence",
)


def _check_supported_features(features: Any) -> None:
    """Recursively check that features contain only supported text/tabular types."""
    if isinstance(features, dict):
        for name, feature in features.items():
            _check_supported_features(feature)
        return

    type_name = type(features).__name__

    # Fail closed on streaming or media types (Image, Audio, Translation, etc.)
    if type_name in ("Image", "Audio", "Translation", "TranslationVariable"):
        raise AdapterError(
            f"Unsupported feature type '{type_name}' in dataset schema. "
            "Media and non-tabular features are currently unsupported."
        )

    if hasattr(features, "feature"):  # Sequence or List container
        _check_supported_features(features.feature)


def dataset_to_snapshot_source(
    dataset: Any,
    *,
    dataset_name: str,
    field_map: Mapping[str, str],
    id_column: str | None = None,
    config: str | None = None,
    split: str | None = None,
    revision: str | None = None,
) -> SnapshotSource:
    """Convert an already-loaded Hugging Face Dataset into a SnapshotSource."""
    if datasets is None:
        raise AdapterError(
            "The 'datasets' package is required for the Hugging Face adapter. "
            "Install it with `pip install evalrepro[huggingface]`."
        )

    if not dataset_name or not dataset_name.strip():
        raise AdapterError("A non-empty 'dataset_name' must be explicitly provided.")

    if isinstance(dataset, IterableDataset):
        raise AdapterError(
            "Streaming datasets (IterableDataset) are currently unsupported in this slice."
        )

    if not isinstance(dataset, Dataset):
        raise AdapterError(f"Expected a Hugging Face Dataset object, got {type(dataset).__name__}.")

    # Verify features/schema recursively
    if hasattr(dataset, "features") and dataset.features:
        _check_supported_features(dataset.features)

    # Validate mapped columns exist in dataset schema
    available_columns = set(dataset.column_names)
    for target_key, source_col in field_map.items():
        if source_col not in available_columns:
            raise AdapterError(
                f"Mapped source column '{source_col}' for key '{target_key}' not found in dataset. "
                f"Available columns: {sorted(available_columns)}"
            )

    if id_column and id_column not in available_columns:
        raise AdapterError(
            f"Specified id_column '{id_column}' not found in dataset. "
            f"Available columns: {sorted(available_columns)}"
        )

    # Canonicalize field_map order (alphabetical key sorting)
    sorted_field_map = {k: field_map[k] for k in sorted(field_map.keys())}

    # Transform records
    records = []
    for row in dataset:
        record: dict[str, Any] = {}

        if id_column is not None:
            record["id"] = row[id_column]

        for target_key, source_col in sorted_field_map.items():
            val = row[source_col]
            record[target_key] = val

        records.append(record)

    # Prepare metadata contracts
    parameters: dict[str, Any] = {
        "config": config,
        "split": split,
        "field_map": sorted_field_map,
        "field_map_version": 1,
    }

    provenance: dict[str, Any] = {}
    if hasattr(dataset, "info") and dataset.info and dataset.info.version:
        provenance["dataset_version"] = str(dataset.info.version)
    if revision is not None:
        provenance["revision"] = revision

    runtime: dict[str, str | None] = {
        "datasets": getattr(datasets, "__version__", "unknown"),
    }
    if huggingface_hub is not None:
        runtime["huggingface_hub"] = getattr(huggingface_hub, "__version__", "unknown")

    fields = tuple(sorted_field_map.keys())
    out_id_field = "id" if id_column is not None else None

    return SnapshotSource(
        adapter="huggingface",
        identity={"dataset": dataset_name},
        parameters=parameters,
        records=records,
        declared_count=len(dataset),
        fields=fields,
        id_field=out_id_field,
        runtime=runtime,
        provenance=provenance,
    )
