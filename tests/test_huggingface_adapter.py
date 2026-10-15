"""Tests for the Hugging Face Dataset adapter."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any, cast

import pytest

from evalrepro.adapters.huggingface import _validate_features, huggingface_dataset_source
from evalrepro.compare import Verdict, compare_manifest_data
from evalrepro.errors import AdapterError
from evalrepro.manifest import build_manifest

pytest.importorskip("datasets")

if TYPE_CHECKING:
    from datasets import Dataset


@pytest.fixture
def base_dataset() -> Dataset:
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "qid": "1",
                "input_col": "A",
                "target_col": "1",
                "ignored": True,
                "null_col": None,
            },
            {
                "qid": "2",
                "input_col": "B",
                "target_col": "2",
                "ignored": False,
                "null_col": "present",
            },
        ]
    )


def test_reproducible_across_revisions_runtime_and_map_order(
    base_dataset: Dataset,
) -> None:
    import datasets
    import huggingface_hub

    field_map_1 = {
        "input": "input_col",
        "target": "target_col",
        "metadata": "null_col",
    }
    source1 = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map=field_map_1,
        id_column="qid",
        revision="v1",
    )

    # Reversed insertion order in field_map
    field_map_2 = {
        "metadata": "null_col",
        "target": "target_col",
        "input": "input_col",
    }
    source2 = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map=field_map_2,
        id_column="qid",
        revision="v2",
    )

    # Verify runtime dependency versions are recorded in runtime, not provenance
    assert source1.runtime["datasets"] == datasets.__version__
    assert source1.runtime["huggingface_hub"] == huggingface_hub.__version__
    assert "huggingface_datasets_version" not in source1.provenance
    assert source1.provenance["explicit_revision"] == "v1"
    assert source2.provenance["explicit_revision"] == "v2"

    # Verify legitimate nulls are preserved in mapped records
    records = list(source1.records)
    assert records[0]["metadata"] is None
    assert records[1]["metadata"] == "present"

    manifest1 = build_manifest(source1)
    manifest2 = build_manifest(source2)
    assert compare_manifest_data(manifest1, manifest2).verdict is Verdict.REPRODUCIBLE

    # Runtime-only changes must not alter reproducibility verdict
    manifest2_runtime_changed = json.loads(json.dumps(manifest2))
    manifest2_runtime_changed["runtime"]["datasets"] = "0.0.0-synthetic"
    assert (
        compare_manifest_data(manifest1, manifest2_runtime_changed).verdict is Verdict.REPRODUCIBLE
    )


def test_caller_mutation_does_not_affect_source(base_dataset: Dataset) -> None:
    mutating_map = {"input": "input_col", "target": "target_col"}
    source = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map=mutating_map,
    )
    manifest_before = build_manifest(source)

    # Mutate caller dictionary after source construction
    mutating_map["target"] = "ignored"
    mutating_map["choices"] = "input_col"

    manifest_after = build_manifest(source)

    assert compare_manifest_data(manifest_before, manifest_after).verdict is Verdict.REPRODUCIBLE
    assert manifest_before["scope"] == manifest_after["scope"]
    assert manifest_before["samples"] == manifest_after["samples"]
    assert source.parameters["field_map"] == {
        "input": "input_col",
        "target": "target_col",
    }
    assert source.fields == ("input", "target")


def test_ignored_column_invariance(base_dataset: Dataset) -> None:
    from datasets import Dataset

    changed_ignored = Dataset.from_list(
        [
            {
                "qid": "1",
                "input_col": "A",
                "target_col": "1",
                "ignored": False,
                "null_col": None,
                "extra_unmapped": "new_col",
            },
            {
                "qid": "2",
                "input_col": "B",
                "target_col": "2",
                "ignored": True,
                "null_col": "present",
                "extra_unmapped": "other",
            },
        ]
    )
    src1 = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map={"input": "input_col", "target": "target_col"},
    )
    src2 = huggingface_dataset_source(
        changed_ignored,
        dataset_name="test",
        field_map={"input": "input_col", "target": "target_col"},
    )
    assert (
        compare_manifest_data(build_manifest(src1), build_manifest(src2)).verdict
        is Verdict.REPRODUCIBLE
    )


def test_scope_mismatches(base_dataset: Dataset) -> None:
    base = huggingface_dataset_source(
        base_dataset,
        dataset_name="t",
        config_name="cfg-a",
        split_name="train",
        id_column="qid",
        field_map={"input": "input_col"},
    )
    diff_name = huggingface_dataset_source(
        base_dataset,
        dataset_name="x",
        config_name="cfg-a",
        split_name="train",
        id_column="qid",
        field_map={"input": "input_col"},
    )
    diff_config = huggingface_dataset_source(
        base_dataset,
        dataset_name="t",
        config_name="cfg-b",
        split_name="train",
        id_column="qid",
        field_map={"input": "input_col"},
    )
    diff_split = huggingface_dataset_source(
        base_dataset,
        dataset_name="t",
        config_name="cfg-a",
        split_name="validation",
        id_column="qid",
        field_map={"input": "input_col"},
    )
    diff_id = huggingface_dataset_source(
        base_dataset,
        dataset_name="t",
        config_name="cfg-a",
        split_name="train",
        id_column="target_col",
        field_map={"input": "input_col"},
    )
    diff_no_id = huggingface_dataset_source(
        base_dataset,
        dataset_name="t",
        config_name="cfg-a",
        split_name="train",
        id_column=None,
        field_map={"input": "input_col"},
    )
    diff_added_field = huggingface_dataset_source(
        base_dataset,
        dataset_name="t",
        config_name="cfg-a",
        split_name="train",
        id_column="qid",
        field_map={"input": "input_col", "target": "target_col"},
    )
    diff_remapped_field = huggingface_dataset_source(
        base_dataset,
        dataset_name="t",
        config_name="cfg-a",
        split_name="train",
        id_column="qid",
        field_map={"input": "target_col"},
    )

    base_man = build_manifest(base)
    candidates = [
        diff_name,
        diff_config,
        diff_split,
        diff_id,
        diff_no_id,
        diff_added_field,
        diff_remapped_field,
    ]
    for diff_src in candidates:
        verdict = compare_manifest_data(base_man, build_manifest(diff_src)).verdict
        assert verdict is Verdict.SCOPE_MISMATCH


def test_semantic_drift_target_type_order_null_and_row_replacement(
    base_dataset: Dataset,
) -> None:
    from datasets import Dataset

    field_map = {
        "input": "input_col",
        "target": "target_col",
        "metadata": "null_col",
    }
    base_src = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map=field_map,
        id_column="qid",
    )
    base_man = build_manifest(base_src)

    # 1. Target value mutation
    target_mutated = Dataset.from_list(
        [
            {
                "qid": "1",
                "input_col": "A",
                "target_col": "MUTATED",
                "ignored": True,
                "null_col": None,
            },
            {
                "qid": "2",
                "input_col": "B",
                "target_col": "2",
                "ignored": False,
                "null_col": "present",
            },
        ]
    )
    # 2. Type mutation ("1" -> 1)
    type_mutated = Dataset.from_list(
        [
            {
                "qid": "1",
                "input_col": "A",
                "target_col": 1,
                "ignored": True,
                "null_col": None,
            },
            {
                "qid": "2",
                "input_col": "B",
                "target_col": 2,
                "ignored": False,
                "null_col": "present",
            },
        ]
    )
    # 3. Row order mutation
    order_mutated = Dataset.from_list(
        [
            {
                "qid": "2",
                "input_col": "B",
                "target_col": "2",
                "ignored": False,
                "null_col": "present",
            },
            {
                "qid": "1",
                "input_col": "A",
                "target_col": "1",
                "ignored": True,
                "null_col": None,
            },
        ]
    )
    # 4. Mapped null mutation (None -> "non-null")
    null_mutated = Dataset.from_list(
        [
            {
                "qid": "1",
                "input_col": "A",
                "target_col": "1",
                "ignored": True,
                "null_col": "non-null",
            },
            {
                "qid": "2",
                "input_col": "B",
                "target_col": "2",
                "ignored": False,
                "null_col": "present",
            },
        ]
    )
    # 5. Same-count row replacement
    row_replaced = Dataset.from_list(
        [
            {
                "qid": "1",
                "input_col": "A",
                "target_col": "1",
                "ignored": True,
                "null_col": None,
            },
            {
                "qid": "3",
                "input_col": "C",
                "target_col": "3",
                "ignored": False,
                "null_col": "present",
            },
        ]
    )

    for candidate_ds in [
        target_mutated,
        type_mutated,
        null_mutated,
        row_replaced,
    ]:
        cand_src = huggingface_dataset_source(
            candidate_ds,
            dataset_name="test",
            field_map=field_map,
            id_column="qid",
        )
        verdict = compare_manifest_data(base_man, build_manifest(cand_src)).verdict
        assert verdict is Verdict.SEMANTIC_DRIFT

    order_src = huggingface_dataset_source(
        order_mutated,
        dataset_name="test",
        field_map=field_map,
        id_column="qid",
    )
    assert compare_manifest_data(base_man, build_manifest(order_src)).verdict is Verdict.ORDER_DRIFT


def test_coverage_mismatch_added_dropped_and_sample_limit(
    base_dataset: Dataset,
) -> None:
    from datasets import Dataset

    base_src = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map={"input": "input_col"},
    )
    base_man = build_manifest(base_src)

    # Dropped row (1 vs 2)
    dropped_ds = Dataset.from_list(
        [
            {
                "qid": "1",
                "input_col": "A",
                "target_col": "1",
                "ignored": True,
                "null_col": None,
            }
        ]
    )
    dropped_src = huggingface_dataset_source(
        dropped_ds,
        dataset_name="test",
        field_map={"input": "input_col"},
    )
    assert (
        compare_manifest_data(base_man, build_manifest(dropped_src)).verdict
        is Verdict.COVERAGE_MISMATCH
    )

    # Added row (3 vs 2)
    added_ds = Dataset.from_list(
        [
            {"qid": "1", "input_col": "A"},
            {"qid": "2", "input_col": "B"},
            {"qid": "3", "input_col": "C"},
        ]
    )
    added_src = huggingface_dataset_source(
        added_ds,
        dataset_name="test",
        field_map={"input": "input_col"},
    )
    assert (
        compare_manifest_data(base_man, build_manifest(added_src)).verdict
        is Verdict.COVERAGE_MISMATCH
    )

    # Full vs sample_limit=1
    partial_man = build_manifest(base_src, sample_limit=1)
    assert compare_manifest_data(base_man, partial_man).verdict is Verdict.COVERAGE_MISMATCH


def test_ids_with_and_without_mapping_and_preview_exclusion(
    base_dataset: Dataset,
) -> None:
    with_id = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map={"input": "input_col"},
        id_column="qid",
    )
    visible = build_manifest(with_id, include_id_preview=True)
    hidden = build_manifest(with_id, include_id_preview=False)

    assert visible["samples"]["id_preview"]["first"] == ["1", "2"]
    assert hidden["samples"]["id_preview"] == {"first": [], "last": []}
    assert hidden["samples"]["ordered_hashes"] == visible["samples"]["ordered_hashes"]
    assert compare_manifest_data(visible, hidden).verdict is Verdict.REPRODUCIBLE

    without_id = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map={"input": "input_col"},
    )
    assert without_id.id_field is None
    assert "id" not in next(iter(without_id.records))
    no_id_manifest = build_manifest(without_id)
    assert no_id_manifest["samples"]["id_preview"] == {
        "first": [None, None],
        "last": [],
    }
    # Verify that IDs participate in whole-record hashes, not just previews
    assert no_id_manifest["samples"]["ordered_hashes"] != visible["samples"]["ordered_hashes"]


def test_safe_provenance_excludes_urls_checksums_and_paths(
    base_dataset: Dataset,
) -> None:
    base_dataset.info.version = "2.1.0"
    base_dataset.info.dataset_size = 4096
    base_dataset.info.homepage = "https://example.com/private-dataset"
    base_dataset.info.description = "/tmp/local/cache/arrow/file.arrow"

    source = huggingface_dataset_source(
        base_dataset,
        dataset_name="test",
        field_map={"input": "input_col"},
        revision="rev-123",
    )

    assert source.provenance == {
        "source": "huggingface-dataset",
        "explicit_revision": "rev-123",
        "dataset_info_version": "2.1.0",
        "dataset_size_bytes": 4096,
    }
    dumped = json.dumps(build_manifest(source))
    assert "https://example.com" not in dumped
    assert "/tmp/local/cache" not in dumped


def test_strict_input_validation(
    base_dataset: Dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(AdapterError, match="dataset_name must be a non-empty string"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name=cast(Any, 123),
            field_map={"input": "input_col"},
        )

    with pytest.raises(AdapterError, match="dataset_name must be a non-empty string"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name="   ",
            field_map={"input": "input_col"},
        )

    for opt_kwargs in [
        {"config_name": ""},
        {"split_name": "   "},
        {"id_column": ""},
        {"id_column": cast(Any, 99)},
        {"revision": "  "},
    ]:
        with pytest.raises(AdapterError, match="must be a non-empty string"):
            huggingface_dataset_source(
                base_dataset,
                dataset_name="test",
                field_map={"input": "input_col"},
                **opt_kwargs,
            )

    with pytest.raises(AdapterError, match="field_map must be a non-empty dictionary"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name="test",
            field_map=cast(Any, ["input"]),
        )

    with pytest.raises(AdapterError, match="field_map must be a non-empty dictionary"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name="test",
            field_map={},
        )

    with pytest.raises(AdapterError, match="keys and values must be strings"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name="test",
            field_map=cast(Any, {"input": 1}),
        )

    with pytest.raises(AdapterError, match="keys and values must be strings"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name="test",
            field_map=cast(Any, {1: "input_col"}),
        )

    with pytest.raises(AdapterError, match="Malformed field_map entry"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name="test",
            field_map={"input": "   "},
        )

    with pytest.raises(AdapterError, match="Invalid target field 'bad'"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name="test",
            field_map={"bad": "input_col"},
        )

    with pytest.raises(AdapterError, match="Mapped source column 'missing'"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name="test",
            field_map={"input": "missing"},
        )

    with pytest.raises(AdapterError, match="ID column 'missing_id' not found"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name="test",
            field_map={"input": "input_col"},
            id_column="missing_id",
        )

    # Simulate missing optional dependency
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(AdapterError, match="The 'datasets' package is required"):
        huggingface_dataset_source(
            base_dataset,
            dataset_name="test",
            field_map={"input": "input_col"},
        )


def test_feature_allowlist_and_rejections() -> None:
    from datasets import (
        Array2D,
        Audio,
        ClassLabel,
        Dataset,
        Features,
        Image,
        Sequence,
        Translation,
        Value,
    )

    # Allowed: Value, ClassLabel, nested dict, Sequence, and list containers
    valid_features = Features(
        {
            "prompt": Value("string"),
            "label": ClassLabel(names=["neg", "pos"]),
            "seq": Sequence(Value("int32")),
            "nested": {"inner": [Value("string")]},
        }
    )
    ds_valid = Dataset.from_list(
        [
            {
                "prompt": "hello",
                "label": 1,
                "seq": [1, 2],
                "nested": {"inner": ["x"]},
            }
        ],
        features=valid_features,
    )
    source = huggingface_dataset_source(
        ds_valid,
        dataset_name="valid",
        field_map={"input": "prompt", "target": "label", "metadata": "nested"},
    )
    assert source.declared_count == 1

    # Unmapped Image feature is still rejected (entire schema checked)
    img_features = Features({"prompt": Value("string"), "unmapped_media": Image()})
    ds_img = Dataset.from_list(
        [{"prompt": "A", "unmapped_media": None}],
        features=img_features,
    )

    # Nested Audio in Sequence
    aud_features = Features({"prompt": Value("string"), "nested": Sequence(Audio())})
    ds_aud = Dataset.from_list(
        [{"prompt": "A", "nested": []}],
        features=aud_features,
    )

    # Nested Image in list container
    list_media = Features({"prompt": Value("string"), "list_img": [Image()]})
    ds_list_img = Dataset.from_list(
        [{"prompt": "A", "list_img": []}],
        features=list_media,
    )

    # Opaque leaves: Translation and Array2D
    trans_features = Features(
        {"prompt": Value("string"), "tr": Translation(languages=["en", "de"])}
    )
    ds_trans = Dataset.from_list(
        [{"prompt": "A", "tr": {"en": "hi", "de": "hallo"}}],
        features=trans_features,
    )

    for bad_ds, match_text in [
        (ds_img, "Image"),
        (ds_aud, "Audio"),
        (ds_list_img, "Image"),
        (ds_trans, "Translation"),
    ]:
        with pytest.raises(AdapterError, match=match_text):
            huggingface_dataset_source(
                bad_ds,
                dataset_name="test",
                field_map={"input": "prompt"},
            )

    with pytest.raises(AdapterError, match="Array2D"):
        _validate_features({"arr": Array2D(shape=(2, 2), dtype="float32")})

    with pytest.raises(AdapterError, match="Empty feature mapping"):
        _validate_features({})

    with pytest.raises(AdapterError, match="Empty sequence feature"):
        _validate_features({"empty_seq": []})


def test_iterable_and_non_dataset_rejection() -> None:
    from datasets import IterableDataset

    def gen() -> Any:
        yield {"col": "A"}

    iterable_ds = IterableDataset.from_generator(gen)
    with pytest.raises(AdapterError, match=r"Expected datasets\.Dataset"):
        huggingface_dataset_source(
            iterable_ds,
            dataset_name="test",
            field_map={"input": "col"},
        )

    with pytest.raises(AdapterError, match=r"Expected datasets\.Dataset"):
        huggingface_dataset_source(
            cast(Any, [{"col": "A"}]),
            dataset_name="test",
            field_map={"input": "col"},
        )
