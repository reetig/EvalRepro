# Hugging Face Dataset Adapter

The `huggingface_dataset_source` adapter maps an already-loaded, non-streaming `datasets.Dataset` into an immutable `SnapshotSource`.

## Installation & Responsibility
Install the optional dependency:
`pip install evalrepro[huggingface]`

**License Boundary:** EvalRepro generates cryptographic hashes of dataset contents. It does not download datasets or manage licenses. The caller is responsible for accepting any Hugging Face Hub conditions, authenticating, and supplying a locally loaded `Dataset`. Hashing does **not** constitute anonymization.

## Python API & Selected-Column Scope
The adapter canonicalizes the `field_map` so caller dictionary mutation or insertion order cannot alter the semantic scope. Unmapped columns are strictly ignored.

```python
from datasets import load_dataset
from evalrepro.adapters.huggingface import huggingface_dataset_source
from evalrepro.manifest import build_manifest

ds = load_dataset("harvey-lab/contracts", split="train")

source = huggingface_dataset_source(
    ds,
    dataset_name="harvey-lab/contracts",
    split_name="train",
    id_column="contract_id",
    field_map={"input": "document_text", "target": "issue_label"},
    revision="a1b2c3d4"  # Explicit caller provenance
)

# Strip previews from the final manifest
manifest = build_manifest(source, include_id_preview=False)