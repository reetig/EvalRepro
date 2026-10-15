# Hugging Face Dataset Adapter

The `huggingface_dataset_source` adapter maps an already-loaded, non-streaming `datasets.Dataset` into an immutable `SnapshotSource`.

## Installation, Support Range & Responsibility

Install the optional dependency extra:

```bash
pip install "evalrepro[huggingface]"
```

**Validated Support Evidence:**
* **Core package:** Validated on Python 3.11, 3.12, and 3.13.
* **Hugging Face extra (`evalrepro[huggingface]`):** Validated on Python 3.11 and 3.12 with `datasets==2.21.0` (`>=2.21.0,<3.0.0`) and `huggingface_hub==0.36.2` (`>=0.36.2,<1.0.0`). Python 3.13 is not claimed for the Hugging Face extra in this initial slice.

**License & Privacy Boundary:** EvalRepro generates cryptographic hashes of evaluation records. It does not download datasets from the network, manage credentials, or enforce dataset licenses. The caller is responsible for complying with dataset terms, authenticating if required, and supplying an in-memory `datasets.Dataset`. Cryptographic hashing is for integrity verification and does **not** constitute anonymization.

## Python API & Selected-Column Scope

The adapter requires a non-empty `field_map` mapping EvalRepro contract fields (`input`, `target`, `choices`, `metadata`) to existing dataset columns. Unmapped columns are ignored during record extraction. The adapter copies and canonicalizes `field_map` (sorting field keys) so dictionary insertion order or caller mutation after construction cannot alter the semantic scope. Legitimate `None` values in mapped columns are preserved.

```python
from datasets import Dataset
from evalrepro.adapters.huggingface import huggingface_dataset_source
from evalrepro.manifest import build_manifest

ds = Dataset.from_list(
    [
        {"qid": "q1", "prompt": "Translate: hello", "answer": "bonjour", "notes": None},
        {"qid": "q2", "prompt": "Translate: world", "answer": "monde", "notes": "verified"},
    ]
)

source = huggingface_dataset_source(
    ds,
    dataset_name="synthetic-translation",
    config_name="fr-v1",
    split_name="test",
    id_column="qid",
    field_map={
        "input": "prompt",
        "target": "answer",
        "metadata": "notes",
    },
    revision="synthetic-rev-001",
)

manifest = build_manifest(source, include_id_preview=False)
```

## Supported Features (Fail-Closed)

To prevent silently dropping or hashing opaque binary objects, `_validate_features` recursively inspects the entire `Dataset.features` schema before iterating over rows:
* **Supported:** `Value`, `ClassLabel`, and recursive `dict`, `Sequence`, or `list` containers composed of these leaf types.
* **Rejected:** `Image`, `Audio`, `Translation`, `Array2D`, empty container definitions, or any other unlisted feature type (raises `AdapterError`).
* **Full-Schema Check:** Unmapped columns with unsupported features are also intentionally rejected because the entire dataset feature schema is validated prior to iteration/decoding.

## Identity, Safe Provenance & Runtime

* **Semantic Scope (`identity` & `parameters`):** `dataset_name`, `config_name`, `split_name`, the canonicalized `field_map`, and the source `id_column` define the semantic scope. Changing any of these produces `Verdict.SCOPE_MISMATCH`.
* **Caller-Owned Provenance:** `revision` is recorded under `provenance["explicit_revision"]` alongside safe dataset metadata (`dataset_info_version` and `dataset_size_bytes` when present on `dataset.info`). Local file paths, cache directories, URLs, and checksums are excluded from provenance.
* **Runtime Versions:** Installed package versions for `datasets` and `huggingface_hub` are recorded in `runtime` via `runtime_versions(("datasets", "huggingface_hub"))` and do not affect semantic scope or reproducibility verdicts.

## IDs, Previews & Whole-Record Hashes

When `id_column` is supplied, the mapped `"id"` field is included in each extracted record:
* **Whole-Record Hashing:** IDs participate directly in the whole-record hashes (`ordered_hashes` / `unordered_hashes`), not just in previews. Adding, removing, or remapping `id_column` changes the semantic scope (`SCOPE_MISMATCH`).
* **Preview Stripping:** Passing `include_id_preview=False` to `build_manifest(source, include_id_preview=False)` empties `samples.id_preview` (`{"first": [], "last": []}`) so raw IDs are omitted from the manifest output while preserving identical sample hashes.

## Coverage & Partial-Processing Boundaries

* **Supplied-Object Completeness:** An already-sliced `Dataset` passed to `huggingface_dataset_source` is treated as complete (`declared_count = len(dataset)`) relative to that supplied object.
* **Sample Limit vs Row Count:** Passing `sample_limit` to `build_manifest` processes a subset of rows while retaining the full `declared_count`, which triggers `Verdict.COVERAGE_MISMATCH` when compared against a full manifest. Adding or dropping rows in the dataset also triggers `Verdict.COVERAGE_MISMATCH`.
* **Equal-Sized Subset Limitation:** Two different equal-sized slices of an upstream dataset supplied as separate `Dataset` objects both report complete coverage of their respective objects (`declared_count == processed_count`); comparing them yields `Verdict.SEMANTIC_DRIFT` rather than `Verdict.COVERAGE_MISMATCH`.