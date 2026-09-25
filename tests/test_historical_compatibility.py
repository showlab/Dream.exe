from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from dream_exe.bench.contracts.schemas import canonical_sha256
from dream_exe.bench.runtime.compatibility import (
    HISTORICAL_COMPATIBILITY_SCHEMA,
    build_historical_task_evaluator_config,
    resolve_historical_compatibility_entry,
    validate_historical_compatibility_manifest,
    verify_historical_compatibility_manifest,
)


TASK_DEFAULTS = {
    "must_reach_min_correction_steps": 3,
    "arm_pos_gain": 2.0,
    "arm_ori_gain": 1.5,
    "warm_start_steps": 10,
    "position_dominate_correction_threshold_m": 0.02,
    "enable_close_completion_gate": True,
    "enable_open_completion_gate": True,
    "close_gate_min_hold_steps": 2,
    "close_gate_max_wait_steps": 60,
    "close_gate_qpos_delta_min": 1.0e-4,
    "close_gate_qpos_settle_tol": 1.0e-4,
    "close_gate_settle_window": 3,
    "close_gate_require_non_support_contact": False,
    "close_gate_contact_settle_steps": 2,
    "close_gate_failure_policy": "continue",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path: Path):
    root = tmp_path / "archive"
    root.mkdir()
    files = {}
    for role, content in (
        ("trajectory", b"trajectory"),
        ("objects", b"objects"),
        ("action", b"action"),
    ):
        path = root / f"{role}.json"
        path.write_bytes(content)
        files[role] = {
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": _sha(path),
        }
    case = {"uid": "case-a", "case": 1}
    environment = {"uid": "case-a", "environment": 1}
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"video")
    reference = {
        "uid": "case-a",
        "video": {"sha256": _sha(video), "size": video.stat().st_size},
        "reference": 1,
    }
    manifest = {
        "format": HISTORICAL_COMPATIBILITY_SCHEMA,
        "cohort": {"collection_id": "active101", "expected_count": 1},
        "source": {
            "name": "paper archive",
            "revision": "commit-abc",
            "description": "all applicable inputs, never outcome selected",
        },
        "input": {"kind": "reference", "reference_id": "wo_gt_depth"},
        "entries": [
            {
                "uid": "case-a",
                "status": "available",
                "reason": None,
                "video_sha256": _sha(video),
                "benchmark": {
                    "case_sha256": canonical_sha256(case),
                    "environment_sha256": canonical_sha256(environment),
                    "reference_sha256": canonical_sha256(reference),
                },
                "artifacts": files,
            }
        ],
    }

    class Repository:
        def load_case(self, uid):
            assert uid == "case-a"
            return case

        def load_environment(self, uid, *, verify_files):
            assert uid == "case-a" and verify_files is True
            return SimpleNamespace(manifest=environment)

        def load_reference(self, uid, *, verify_files):
            assert uid == "case-a" and verify_files is False
            return SimpleNamespace(manifest=reference, video=video)

    return root, manifest, Repository()


def test_historical_compatibility_verifies_closed_population(tmp_path):
    root, manifest, repository = _fixtures(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest))
    receipt = verify_historical_compatibility_manifest(
        manifest_path=manifest_path,
        artifact_root=root,
        repository=repository,
        expected_uids={"case-a"},
    )
    assert receipt["population"] == 1
    assert receipt["available"] == 1
    assert receipt["unavailable"] == 0
    assert set(receipt["entries"][0]["artifacts"]) == {
        "trajectory",
        "objects",
        "action",
    }
    selected = resolve_historical_compatibility_entry(
        manifest_path=manifest_path,
        artifact_root=root,
        repository=repository,
        uid="case-a",
    )
    assert selected["manifest_sha256"] == receipt["manifest_sha256"]
    assert selected["status"] == "available"


def test_historical_compatibility_rejects_outcome_subset(tmp_path):
    _root, manifest, _repository = _fixtures(tmp_path)
    with pytest.raises(ValueError, match="exactly cohort.expected_count"):
        changed = copy.deepcopy(manifest)
        changed["cohort"]["expected_count"] = 2
        validate_historical_compatibility_manifest(changed)


def test_historical_compatibility_requires_explicit_unavailable_rows(tmp_path):
    _root, manifest, _repository = _fixtures(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["entries"][0]["status"] = "unavailable"
    changed["entries"][0]["reason"] = "no paper-era action exists"
    changed["entries"][0]["artifacts"] = None
    normalized = validate_historical_compatibility_manifest(changed)
    assert normalized["entries"][0]["status"] == "unavailable"


def test_historical_compatibility_rejects_digest_drift(tmp_path):
    root, manifest, repository = _fixtures(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest))
    (root / "action.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        verify_historical_compatibility_manifest(
            manifest_path=manifest_path,
            artifact_root=root,
            repository=repository,
            expected_uids={"case-a"},
        )


def test_historical_compatibility_rejects_wrong_population(tmp_path):
    root, manifest, repository = _fixtures(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest))
    with pytest.raises(ValueError, match="cohort UID mismatch"):
        verify_historical_compatibility_manifest(
            manifest_path=manifest_path,
            artifact_root=root,
            repository=repository,
            expected_uids={"case-a", "case-b"},
        )


def test_historical_task_config_is_separate_from_actual_execution():
    base = {
        "execution": {
            "controller": "OSC_POSE",
            "use_ori": True,
            "pose_correction_mode": "position_dominate",
            "max_correction_steps": 1,
        },
        "runtime": {"policy_hz": 20},
    }
    evidence = {
        "controller": "OSC_POSITION",
        "use_ori": False,
        "pose_correction_mode": "coupled",
        "pos_tol": 0.005,
        "ori_tol": 0.03,
        "max_correction_steps": 3,
    }
    resolved = build_historical_task_evaluator_config(
        base_config=base,
        archived_task_evidence=evidence,
        evaluator_defaults=TASK_DEFAULTS,
    )
    assert base["execution"]["max_correction_steps"] == 1
    assert resolved["execution"]["max_correction_steps"] == 3
    assert resolved["execution"]["pose_correction_mode"] == "coupled"
    assert resolved["execution"]["close_gate_max_wait_steps"] == 60


def test_historical_task_config_requires_complete_explicit_defaults():
    with pytest.raises(ValueError, match="default fields"):
        build_historical_task_evaluator_config(
            base_config={"execution": {}},
            archived_task_evidence={
                "controller": "OSC_POSITION",
                "use_ori": False,
                "pose_correction_mode": "coupled",
                "pos_tol": 0.005,
                "ori_tol": 0.03,
                "max_correction_steps": 3,
            },
            evaluator_defaults={},
        )
