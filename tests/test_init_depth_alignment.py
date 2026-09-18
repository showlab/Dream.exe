from __future__ import annotations

import numpy as np
import pytest

from dream_exe.artifacts.io import (
    load_traj_assets_manifest,
    write_trajectory_artifacts,
)

from dream_exe.video2traj.runtime.conditioning import (
    align_runtime_init_depth_inputs,
)


def test_init_depth_alignment_matches_nearest_resize_and_preserves_inputs() -> None:
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    source[0, 0] = np.nan
    original = source.copy()

    aligned = align_runtime_init_depth_inputs(
        target_shape_hw=(2, 2),
        depth_options={"init_ref_depth": source},
        target_depth_options={"init_ref_depth": source},
        object_runtime_options={"init_depth": source},
    )

    expected = np.asarray(
        [[np.nan, 2.0], [8.0, 10.0]],
        dtype=np.float32,
    )
    for key, field in (
        ("depth_options", "init_ref_depth"),
        ("target_depth_options", "init_ref_depth"),
        ("object_runtime_options", "init_depth"),
    ):
        value = aligned[key][field]
        assert value.shape == (2, 2)
        np.testing.assert_allclose(value, expected, equal_nan=True)
    np.testing.assert_allclose(source, original, equal_nan=True)
    assert aligned["manifest"]["algorithm"] == "resize_nearest"
    assert len(aligned["manifest"]["consumers"]) == 3


def test_init_depth_alignment_is_noop_for_matching_shape() -> None:
    source = np.arange(9, dtype=np.float32).reshape(3, 3)
    aligned = align_runtime_init_depth_inputs(
        target_shape_hw=(3, 3),
        depth_options={"init_ref_depth": source},
        target_depth_options={},
        object_runtime_options={"init_depth": source},
    )

    np.testing.assert_array_equal(
        aligned["depth_options"]["init_ref_depth"],
        source,
    )
    np.testing.assert_array_equal(
        aligned["object_runtime_options"]["init_depth"],
        source,
    )
    assert aligned["manifest"]["consumers"] == []


def test_init_depth_alignment_rejects_invalid_arrays() -> None:
    try:
        align_runtime_init_depth_inputs(
            target_shape_hw=(2, 2),
            depth_options={"init_ref_depth": np.zeros((1, 2, 2), dtype=np.float32)},
            target_depth_options={},
            object_runtime_options={},
        )
    except ValueError as error:
        assert "depth_options.init_ref_depth" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("invalid depth shape was accepted")


def test_init_depth_alignment_matches_legacy_512_to_480() -> None:
    import cv2

    source = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    aligned = align_runtime_init_depth_inputs(
        target_shape_hw=(480, 480),
        depth_options={"init_ref_depth": source},
        target_depth_options={},
        object_runtime_options={},
    )
    np.testing.assert_array_equal(
        aligned["depth_options"]["init_ref_depth"],
        cv2.resize(source, (480, 480), interpolation=cv2.INTER_NEAREST),
    )


def test_alignment_provenance_is_published_and_cleared_on_republication(tmp_path) -> None:
    aligned = align_runtime_init_depth_inputs(
        target_shape_hw=(2, 2),
        depth_options={"init_ref_depth": np.ones((4, 4), dtype=np.float32)},
        target_depth_options={},
        object_runtime_options={},
    )
    output_dir = str(tmp_path / "traj")
    write_trajectory_artifacts(
        output_dir,
        {"points": []},
        manifest_payload={"conditioning_alignment": aligned["manifest"]},
    )
    section = load_traj_assets_manifest(output_dir)["trajectory"]
    assert section["conditioning_alignment"] == aligned["manifest"]
    assert section["ee_traj_json"] == "trajectory/ee_traj.json"

    write_trajectory_artifacts(output_dir, {"points": []})
    assert "conditioning_alignment" not in load_traj_assets_manifest(output_dir)["trajectory"]


def test_reserved_manifest_field_is_rejected_before_artifact_writes(tmp_path) -> None:
    output_dir = tmp_path / "traj"
    with pytest.raises(ValueError, match="publication-owned"):
        write_trajectory_artifacts(
            str(output_dir),
            {"points": []},
            manifest_payload={"ee_traj_json": "other.json"},
        )
    assert not output_dir.exists()
