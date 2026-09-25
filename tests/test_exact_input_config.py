"""Input-specific experiment choices must not leak between videos or references."""
import copy
import json
from pathlib import Path

import pytest

from dream_exe.bench.contracts.config import compile_config
from dream_exe.bench.contracts.schemas import (
    CASE_PROTOCOL_ROUTES, PIPELINE_STAGES, canonical_sha256, validate_document,
)


def identity(reference=None, model=None, variant=None):
    return dict(kind="reference" if reference else "generated", reference_id=reference,
                model_id=model, prompt_variant=variant)


def fixture():
    inputs = [identity(reference="w_gt_depth"), identity(reference="wo_gt_depth"),
              identity(model="model", variant="standard"), identity(model="model", variant="enhanced")]
    case = dict(format="dream-exe.case-protocol", uid="case", routes={r: {} for r in CASE_PROTOCOL_ROUTES})
    case["routes"]["reference_input"] = {"action": {"gripper_actuation_mode": "parallel"}}
    case["input_configs"] = [dict(input=i, values=dict(
        video2traj={"gripper": {"method": "3d" if n == 0 else "2d"}},
        action={"settle_steps_after_close": n + 2},
        execution={"execution": {"max_correction_steps": n + 3}},
    )) for n, i in enumerate(inputs)]
    protocols = {s: dict(format="dream-exe.protocol", stage=s, values={}) for s in PIPELINE_STAGES}
    protocols["evaluation"] = json.loads((Path(__file__).resolve().parents[1] / "examples/quickstart/data/bench/protocol/evaluation.json").read_text())
    protocols["action"]["values"] = {"settle_steps_after_close": 99}
    return case, protocols, inputs


def compile_one(case, protocols, i, **kwargs):
    return compile_config(uid="case", input_identity={**i, "video_sha256": "a" * 64},
                          protocols=protocols, case_protocol=case,
                          route="reference_input" if i["kind"] == "reference" else "candidate", **kwargs)


def test_exact_inputs_are_independent_alternative_owners():
    case, protocols, inputs = fixture()
    before = copy.deepcopy(case)
    for n, i in enumerate(inputs):
        result = compile_one(case, protocols, i)
        assert result["values"]["action"]["settle_steps_after_close"] == n + 2
        assert result["values"]["execution"]["execution"]["max_correction_steps"] == n + 3
        assert result["sources"]["/action/settle_steps_after_close"].startswith("case-input:")
    gt = compile_one(case, protocols, inputs[0])
    assert gt["values"]["video2traj"]["gripper"]["method"] == "3d"
    assert gt["values"]["action"]["gripper_actuation_mode"] == "serial"
    assert case == before


def test_missing_input_fails_instead_of_silent_route_fallback():
    case, protocols, _ = fixture()
    with pytest.raises(ValueError, match="no exact input config"):
        compile_one(case, protocols, identity(model="unseen", variant="standard"))


def test_evaluation_reference_uses_the_same_exact_gt_settings():
    case, protocols, inputs = fixture()
    oracle = compile_config(uid="case", input_identity={**inputs[0], "video_sha256": "a" * 64},
                            protocols=protocols, case_protocol=case, route="evaluation_oracle")
    gt = compile_one(case, protocols, inputs[0])
    assert oracle["values"] == gt["values"]
    with pytest.raises(ValueError, match="requires the GT-depth"):
        compile_config(uid="case", input_identity={**inputs[1], "video_sha256": "a" * 64},
                       protocols=protocols, case_protocol=case, route="evaluation_oracle")


def test_duplicate_and_unknown_stage_are_rejected():
    case, _, _ = fixture()
    case["input_configs"].append(copy.deepcopy(case["input_configs"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_document(case)
    case["input_configs"].pop()
    case["input_configs"][0]["values"]["typo"] = {}
    with pytest.raises(ValueError, match="unknown fields"):
        validate_document(case)


def test_run_paths_stay_run_owned_and_bad_fields_fail():
    case, protocols, inputs = fixture()
    r = compile_one(case, protocols, inputs[0], run_values={"video2traj": {"depth": {"use_rollout_gt_depth": True}}})
    assert r["sources"]["/video2traj/depth/use_rollout_gt_depth"] == "run:video2traj"
    case["input_configs"][0]["values"]["action"]["misspelled"] = True
    with pytest.raises(ValueError, match="unknown action field"):
        compile_one(case, protocols, inputs[0])


def test_old_protocols_remain_compatible_and_changes_are_fingerprinted():
    case, protocols, inputs = fixture()
    old_hash = canonical_sha256(case)
    case["input_configs"][0]["values"]["action"]["settle_steps_after_close"] = 7
    assert canonical_sha256(case) != old_hash
    del case["input_configs"]
    result = compile_one(case, protocols, inputs[0])
    assert result["values"]["action"]["gripper_actuation_mode"] == "parallel"
    assert result["values"]["action"]["settle_steps_after_close"] == 99
