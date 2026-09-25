from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dream_exe.sim.robocasa.task_components import (
    _blender_containment_diagnostics,
)


class _Object:
    name = "pear"

    def get_bbox_points(self, *, trans, rot):
        del rot
        offsets = np.asarray(
            [
                [x, y, z]
                for x in (-0.1, 0.1)
                for y in (-0.1, 0.1)
                for z in (-0.1, 0.1)
            ],
            dtype=np.float64,
        )
        return offsets + np.asarray(trans, dtype=np.float64)


class _Fixture:
    def get_int_sites(self, *, relative):
        assert relative is False
        return {
            "jar": (
                np.asarray([0.0, 0.0, 0.0]),
                np.asarray([1.0, 0.0, 0.0]),
                np.asarray([0.0, 1.0, 0.0]),
                np.asarray([0.0, 0.0, 1.0]),
            )
        }


def _environment(center):
    data = SimpleNamespace(
        body_xpos=np.asarray([center], dtype=np.float64),
        body_xquat=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    return SimpleNamespace(
        objects={"obj": _Object()},
        blender=_Fixture(),
        obj_body_id={"pear": 0},
        sim=SimpleNamespace(data=data),
    )


def test_blender_containment_diagnostic_positive_inside() -> None:
    result = _blender_containment_diagnostics(
        _environment([0.5, 0.5, 0.5]),
        threshold=0.01,
        quat_converter=lambda value: value,
    )
    best = result["blender_best_region"]
    assert best["region"] == "jar"
    assert best["predicate_signed_margin_normalized_m"] > 0.0
    assert best["interior_center_world"] == [0.5, 0.5, 0.5]


def test_blender_containment_diagnostic_identifies_outside_face() -> None:
    result = _blender_containment_diagnostics(
        _environment([1.2, 0.5, 0.5]),
        threshold=0.01,
        quat_converter=lambda value: value,
    )
    best = result["blender_best_region"]
    assert best["predicate_signed_margin_normalized_m"] < 0.0
    assert best["worst_axis"] == "u"
    assert best["worst_direction"] == "upper"
