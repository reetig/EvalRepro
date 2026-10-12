"""Built-in evaluation-source adapters."""

from evalrepro.adapters.base import SnapshotSource
from evalrepro.adapters.harvey_lab import harvey_lab_source
from evalrepro.adapters.huggingface import huggingface_dataset_source
from evalrepro.adapters.jsonl import jsonl_source

__all__ = [
    "SnapshotSource",
    "harvey_lab_source",
    "huggingface_dataset_source",
    "jsonl_source"
]
