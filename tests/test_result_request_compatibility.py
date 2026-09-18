"""Historical reads retain evidence; current request writes stay strict."""
import hashlib
import json

import pytest

from dream_exe.bench.contracts.schemas import (
    PIPELINE_STAGES, PROTOCOL_STAGES, canonical_sha256, validate_document,
)
from dream_exe.bench.outputs.results import ResultRepository


def bundle(tmp_path, *, legacy=True):
    root = tmp_path / "experiments/case/model/standard"
    root.mkdir(parents=True)
    digest = "a" * 64
    identity = dict(kind="generated", model_id="model", prompt_variant="standard",
                    reference_id=None, video_sha256=digest)
    request = dict(
        format="dream-exe.result-request", case_sha256=digest, init_sha256=digest,
        generation_input_sha256=digest, reference_sha256=digest,
        packaged_defaults_sha256=dict.fromkeys(PIPELINE_STAGES, digest),
        protocol_sha256=dict.fromkeys(PROTOCOL_STAGES, digest),
        case_protocol_sha256=digest, input=identity, input_manifest_sha256=digest,
        resolved_config_sha256=None, runtime_config_sha256=None,
        initialization=dict(mode="frozen", receipt_sha256=None),
        provenance_status="reconstructed",
    )
    if legacy:
        request["run_sha256"] = digest
    else:
        request.update(execution_spec_sha256=digest, implementation_sha256=digest)
    (root / "request.json").write_text(json.dumps(request))
    payload = b"unchanged scientific artifact"
    (root / "evidence.bin").write_bytes(payload)
    result = dict(format="dream-exe.result", run_id="run", uid="case", input=identity,
                  status="partial", completed_stages=[], task_outcome="not_evaluated",
                  provenance_status="reconstructed", request_sha256=canonical_sha256(request),
                  resolved_config_sha256=None, artifacts=[dict(path="evidence.bin",
                  size=len(payload), sha256=hashlib.sha256(payload).hexdigest())])
    (root / "result.json").write_text(json.dumps(result))
    return ResultRepository(tmp_path), root, request, result


def test_legacy_read_exposes_missing_identity_without_changing_bytes(tmp_path):
    repo, root, request, result = bundle(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    assert repo.load_result(root) == result
    assert repo.load_request(root) == request
    info = repo.request_provenance(root)
    assert info["request_schema"] == "legacy"
    assert info["provenance_status"] == "reconstructed"
    assert info["implementation_status"] == "unknown"
    assert info["implementation_sha256"] is None
    assert info["execution_spec_sha256"] is None
    assert info["run_sha256"] == request["run_sha256"]
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_current_request_read_and_validation_are_unchanged(tmp_path):
    repo, root, request, result = bundle(tmp_path, legacy=False)
    validate_document(request)
    assert repo.load_result(root) == result
    assert repo.request_provenance(root)["implementation_status"] == "recorded"


def test_legacy_is_not_accepted_by_current_write_schema(tmp_path):
    _, _, request, _ = bundle(tmp_path)
    with pytest.raises(ValueError):
        validate_document(request)


@pytest.mark.parametrize("tamper", ["artifact", "request"])
def test_legacy_integrity_checks_still_reject_tampering(tmp_path, tamper):
    repo, root, request, _ = bundle(tmp_path)
    if tamper == "artifact":
        original = (root / "evidence.bin").read_bytes()
        (root / "evidence.bin").write_bytes(b"x" * len(original))
    else:
        request["run_sha256"] = "b" * 64
        (root / "request.json").write_text(json.dumps(request))
    with pytest.raises(ValueError, match="digest mismatch"):
        repo.load_result(root)


@pytest.mark.parametrize("change", [{"implementation_sha256": "a" * 64},
                                   {"provenance_status": "complete"}])
def test_hybrid_or_complete_legacy_request_is_rejected(tmp_path, change):
    repo, root, request, _ = bundle(tmp_path)
    request.update(change)
    (root / "request.json").write_text(json.dumps(request))
    with pytest.raises(ValueError):
        repo.load_request(root)
