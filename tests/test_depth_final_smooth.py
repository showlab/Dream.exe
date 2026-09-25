import numpy as np
import pytest

from dream_exe.video2traj.depth.calibration import (
    bilateral_smooth_in_roi,
    run_depth_calibration,
    run_depth_calibration_runtime,
)
from dream_exe.video2traj.depth.config import normalize_depth_base_config
from dream_exe.video2traj.depth.cache import normalize_depth_calibration_metadata


def test_smoothing_is_explicit_and_strict():
    assert not normalize_depth_base_config({})["final_smooth"]["enabled"]
    for config in ({"sigma_r": 0}, {"sigma_r": float("nan")},
                   {"enabled": "true"}, {"bilateral_mode": "bad"}, {"typo": 1}):
        with pytest.raises(ValueError):
            normalize_depth_base_config({"final_smooth": config})


def test_roi_smoothing_matches_historical_opencv_recipe():
    import cv2

    depth = np.random.default_rng(4).uniform(0.5, 1, (16, 16)).astype(np.float32)
    depth[6, 6] = np.nan
    roi = np.zeros(depth.shape, bool)
    roi[4:12, 4:12] = True
    filled = depth.copy()
    filled[6, 6] = np.median(depth[roi & np.isfinite(depth)])
    expected = filled.copy()
    filtered = cv2.bilateralFilter((filled / 0.02).clip(0, 65535), 5, 10.0, 5.0)
    expected[roi] = (filtered * 0.02)[roi]
    np.testing.assert_array_equal(bilateral_smooth_in_roi(depth, roi), expected)
    output, stats = run_depth_calibration(
        [depth], depth_space="metric", run_init_calibration=False,
        first_region_masks=[roi], run_final_smooth=True,
    )
    np.testing.assert_array_equal(output[0], expected)
    assert stats["final_smooth"]["applied"]


def test_gt_skips_entire_base_postprocess_even_if_final_smooth_is_enabled():
    depth = np.random.default_rng(7).uniform(1, 2, (1, 16, 16)).astype(np.float32)
    kwargs = dict(depths=depth, depth_source="rollout_gt_depth", depth_model="unused",
                  depth_info={"depth_space": "metric"}, init_ref_depth=np.ones((16, 16)),
                  valid_masks=None)
    output, _, stages = run_depth_calibration_runtime(depth_base_cfg={}, **kwargs)
    np.testing.assert_array_equal(output, depth)
    assert not stages["base"]["init_calibration"]["applied"]
    output, _, stages = run_depth_calibration_runtime(
        depth_base_cfg={"final_smooth": {"enabled": True, "bilateral_mode": "on"}}, **kwargs
    )
    assert stages["base"]["final_smooth"]["enabled"]
    assert not stages["base"]["final_smooth"]["applied"]
    assert "rollout_gt_depth" in stages["base"]["final_smooth"]["reason"]
    assert normalize_depth_calibration_metadata(stages) == stages
    np.testing.assert_array_equal(output, depth)
