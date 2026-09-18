from __future__ import annotations

import json
from pathlib import Path

import pytest

from dream_exe.models import (
    MODEL_CATEGORY_BY_KIND,
    MODEL_KINDS_BY_CATEGORY,
    ModelCatalogError,
    available_models,
    compose_runtime_config,
    load_model_catalog,
    resolve_model,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CATALOG = REPOSITORY_ROOT / "examples/custom_models/models.example.json"
EXAMPLE_RUNTIME = REPOSITORY_ROOT / "examples/custom_models/runtime.example.json"


def _openai_vlm_entry() -> dict[str, object]:
    return {
        "kind": "vlm",
        "backend": "openai-compatible",
        "options": {
            "model": "provider-model",
            "base_url": "https://provider.example/v1",
            "api_key_env": "TEST_VLM_API_KEY",
        },
    }


def test_example_catalog_has_three_lifecycle_categories() -> None:
    catalog = load_model_catalog(EXAMPLE_CATALOG)

    assert catalog.layout == "categorized"
    assert dict(catalog.categories) == {
        "video_gen": ("example_local_video", "example_polling_video"),
        "exec": (
            "example_detector",
            "example_segmenter",
            "example_tracker",
            "example_depth",
            "example_pose",
        ),
        "eval": ("example_openai_vlm", "example_custom_vlm"),
    }
    assert MODEL_KINDS_BY_CATEGORY == {
        "video_gen": ("video_generation",),
        "exec": (
            "region_detector",
            "region_segmenter",
            "tracking",
            "depth",
            "pose",
        ),
        "eval": ("vlm",),
    }
    assert MODEL_CATEGORY_BY_KIND["depth"] == "exec"
    assert MODEL_CATEGORY_BY_KIND["vlm"] == "eval"


def test_category_filters_and_resolution_use_the_same_mapping() -> None:
    catalog = load_model_catalog(EXAMPLE_CATALOG)

    assert {
        row["category"] for row in available_models(catalog, category="exec")
    } == {"exec"}
    assert resolve_model(
        "example_local_video",
        catalog=catalog,
        expected_category="video_gen",
        expected_kind="video_generation",
    ).category == "video_gen"
    assert resolve_model(
        "example_openai_vlm",
        catalog=catalog,
        expected_category="eval",
        expected_kind="vlm",
    ).category == "eval"

    with pytest.raises(ModelCatalogError, match="has category 'eval'"):
        resolve_model(
            "example_openai_vlm",
            catalog=catalog,
            expected_category="exec",
        )


def test_grouped_catalog_rejects_a_kind_in_the_wrong_category(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "format": "dream-exe.models",
                "categories": {"exec": {"misfiled_vlm": _openai_vlm_entry()}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ModelCatalogError,
        match="kind 'vlm' belongs to category 'eval'",
    ):
        load_model_catalog(path)


def test_legacy_flat_catalog_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "format": "dream-exe.models",
                "models": {"legacy_vlm": _openai_vlm_entry()},
            }
        ),
        encoding="utf-8",
    )

    catalog = load_model_catalog(path)

    assert catalog.layout == "flat-legacy"
    assert catalog.categories["eval"] == ("legacy_vlm",)
    assert catalog.models["legacy_vlm"]["category"] == "eval"


def test_runtime_composition_accepts_only_exec_category(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "format": "dream-exe.runtime",
                "category": "eval",
                "device": "cuda:0",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelCatalogError, match="only category 'exec'"):
        compose_runtime_config(defaults={}, runtime_config_path=runtime)


def test_example_runtime_composes_only_exec_models() -> None:
    catalog = load_model_catalog(EXAMPLE_CATALOG)

    composition = compose_runtime_config(
        defaults={},
        catalog=catalog,
        runtime_config_path=EXAMPLE_RUNTIME,
        preflight_factories=False,
    )

    assert {item["kind"] for item in composition.selected_models} == {
        "region_detector",
        "region_segmenter",
        "tracking",
        "depth",
        "pose",
    }
