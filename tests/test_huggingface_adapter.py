"""Tests for the Hugging Face Dataset adapter."""

from typing import TYPE_CHECKING, Any

import pytest

from evalrepro.adapters.huggingface import huggingface_dataset_source
from evalrepro.compare import Verdict, compare_manifest_data
from evalrepro.errors import AdapterError
from evalrepro.manifest import build_manifest

pytest.importorskip("datasets")

if TYPE_CHECKING:
    from datasets import Dataset


@pytest.fixture
def base_dataset() -> "Dataset":
    from datasets import Dataset

    return Dataset.from_list(
        [
            {"qid": "1", "input_col": "A", "target_col": "1", "ignored": True, "null_col": None},
            {"qid": "2", "input_col": "B", "target_col": "2", "ignored": False, "null_col": None},
        ]
    )


def test_reproducible_across_revisions_and_map_order(base_dataset: "Dataset") -> None:
    source1 = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map={"input": "input_col", "target": "target_col"},
        revision="v1",
    )
    # Reverse the map order
    source2 = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map={"target": "target_col", "input": "input_col"},
        revision="v2",
    )

    assert (
        compare_manifest_data(build_manifest(source1), build_manifest(source2)).verdict
        is Verdict.REPRODUCIBLE
    )
    assert source1.provenance["explicit_revision"] == "v1"


def test_caller_mutation_does_not_affect_source(base_dataset: "Dataset") -> None:
    mutating_map = {"input": "input_col"}
    source = huggingface_dataset_source(base_dataset, dataset_name="test", field_map=mutating_map)
    mutating_map["target"] = "target_col"  # Mutate after creation

    manifest = build_manifest(source)
    assert "target" not in manifest["samples"]["ordered_hashes"][0]


def test_scope_mismatches(base_dataset: "Dataset") -> None:
    base = huggingface_dataset_source(
        base_dataset, dataset_name="t", field_map={"input": "input_col"}
    )
    diff_name = huggingface_dataset_source(
        base_dataset, dataset_name="x", field_map={"input": "input_col"}
    )
    diff_config = huggingface_dataset_source(
        base_dataset, dataset_name="t", config_name="c", field_map={"input": "input_col"}
    )
    diff_split = huggingface_dataset_source(
        base_dataset, dataset_name="t", split_name="s", field_map={"input": "input_col"}
    )
    diff_id = huggingface_dataset_source(
        base_dataset, dataset_name="t", id_column="qid", field_map={"input": "input_col"}
    )

    base_man = build_manifest(base)
    for diff_src in [diff_name, diff_config, diff_split, diff_id]:
        assert (
            compare_manifest_data(base_man, build_manifest(diff_src)).verdict
            is Verdict.SCOPE_MISMATCH
        )


def test_semantic_drift_and_coverage(base_dataset: "Dataset") -> None:
    from datasets import Dataset

    base_src = huggingface_dataset_source(
        base_dataset, dataset_name="test", field_map={"input": "input_col"}
    )
    base_man = build_manifest(base_src)

    # Legitimate nulls should hash cleanly, but a changed value is semantic drift
    mutated = Dataset.from_list(
        [
            {
                "qid": "1",
                "input_col": "CHANGED",
                "target_col": "1",
                "ignored": True,
                "null_col": None,
            },
            {"qid": "2", "input_col": "B", "target_col": "2", "ignored": False, "null_col": None},
        ]
    )
    mut_src = huggingface_dataset_source(
        mutated, dataset_name="test", field_map={"input": "input_col"}
    )
    assert (
        compare_manifest_data(base_man, build_manifest(mut_src)).verdict is Verdict.SEMANTIC_DRIFT
    )

    # Same count, different row
    added_dropped = Dataset.from_list(
        [
            {"qid": "1", "input_col": "A", "target_col": "1", "ignored": True, "null_col": None},
            {"qid": "3", "input_col": "C", "target_col": "3", "ignored": False, "null_col": None},
        ]
    )
    ad_src = huggingface_dataset_source(
        added_dropped, dataset_name="test", field_map={"input": "input_col"}
    )
    assert (
        compare_manifest_data(base_man, build_manifest(ad_src)).verdict is Verdict.COVERAGE_MISMATCH
    )


def test_strict_input_validation(base_dataset: "Dataset") -> None:
    with pytest.raises(AdapterError, match="must be a non-empty string"):
        huggingface_dataset_source(base_dataset, dataset_name=123, field_map={"input": "input_col"})  # type: ignore

    with pytest.raises(AdapterError, match="must be a non-empty string"):
        huggingface_dataset_source(
            base_dataset, dataset_name="test", id_column="   ", field_map={"input": "input_col"}
        )

    with pytest.raises(AdapterError, match="must be a non-empty dictionary"):
        huggingface_dataset_source(base_dataset, dataset_name="test", field_map=[])  # type: ignore

    with pytest.raises(AdapterError, match="keys and values must be strings"):
        huggingface_dataset_source(base_dataset, dataset_name="test", field_map={"input": 1})  # type: ignore


def test_feature_rejections() -> None:
    from datasets import Audio, Dataset, Features, Image, Sequence, Value

    img_features = Features({"prompt": Value("string"), "media": Image()})
    ds_img = Dataset.from_list([{"prompt": "A", "media": None}], features=img_features)

    aud_features = Features({"prompt": Value("string"), "nested": Sequence(Audio())})
    ds_aud = Dataset.from_list([{"prompt": "A", "nested": []}], features=aud_features)

    for bad_ds, match_text in [(ds_img, "Image"), (ds_aud, "Audio")]:
        with pytest.raises(AdapterError, match=match_text):
            huggingface_dataset_source(bad_ds, dataset_name="test", field_map={"input": "prompt"})


def test_iterable_dataset_rejection() -> None:
    from datasets import IterableDataset

    def gen() -> Any:
        yield {"col": "A"}

    iterable_ds = IterableDataset.from_generator(gen)
    with pytest.raises(AdapterError, match=r"Expected datasets\.Dataset"):
        huggingface_dataset_source(iterable_ds, dataset_name="test", field_map={"input": "col"})
