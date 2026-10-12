# Hugging Face Dataset Adapter

The `huggingface_dataset_source` adapter maps an already-loaded, non-streaming `datasets.Dataset` into an immutable `SnapshotSource`.

## Python API & Selected-Column Scope
The adapter requires a strict `field_map` mapping EvalRepro target fields (`input`, `target`, `choices`, `metadata`) to existing dataset columns. Unmapped columns are strictly ignored. The adapter canonicalizes this mapping so caller dictionary mutation or insertion order cannot alter the semantic scope or the resulting snapshot hash.

## Supported Features (Fail-Closed)
To prevent silently dropping or hashing opaque binary media, the adapter strictly validates the entire `Dataset.features` schema prior to extraction.
* **Supported:** `Value`, `ClassLabel`, and nested `Sequence` or `dict` containers.
* **Rejected:** `Image`, `Audio`, or any unmapped/unsupported features (raises `AdapterError` before iteration).

## Privacy & ID Previews
If an `id_column` is provided in the parameters, it powers compact previews in the manifest records. Supplying the `include_id_preview=False` flag safely strips these preview values from the export while preserving the overall semantic hash.

## Partial Coverage Limitations
If a dataset is sliced or restricted using a `sample_limit`, a comparison against the full dataset will correctly trigger a `coverage_mismatch` verdict to warn researchers that the evaluation scope differs.