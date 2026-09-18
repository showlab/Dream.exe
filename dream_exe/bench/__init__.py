"""Canonical benchmark storage, configuration, and runtime contracts.

The package is deliberately outside :mod:`dream_exe.video2traj` and
:mod:`dream_exe.sim`. It owns immutable benchmark inputs, workspace-owned
outputs, strict configuration composition, and the public init/generate/run
facade.
Historical layout readers and migration utilities are intentionally outside
this package.
"""

from .contracts.schemas import (
    BENCH_SCHEMA,
    CASE_PROTOCOL_ROUTES,
    CASE_PROTOCOL_SCHEMA,
    CASE_SCHEMA,
    COLLECTION_SCHEMA,
    ENVIRONMENT_SCHEMA,
    ENVIRONMENT_STATE_SCHEMA,
    GENERATION_INPUT_SCHEMA,
    INIT_RECEIPT_SCHEMA,
    PROTOCOL_SCHEMA,
    REFERENCE_SCHEMA,
    RESULT_SCHEMA,
    RESULT_REQUEST_SCHEMA,
    RESOLVED_CONFIG_SCHEMA,
    RUN_SCHEMA,
    RUN_SUMMARY_SCHEMA,
    SOURCE_SCHEMA,
    VIDEO_OUTPUT_SCHEMA,
    WORK_MATERIALIZATION_SCHEMA,
    WORKSPACE_SCHEMA,
    canonical_sha256,
    load_json_strict,
    validate_document,
)

__all__ = [
    "BENCH_SCHEMA",
    "CASE_PROTOCOL_ROUTES",
    "CASE_PROTOCOL_SCHEMA",
    "CASE_SCHEMA",
    "COLLECTION_SCHEMA",
    "ENVIRONMENT_SCHEMA",
    "ENVIRONMENT_STATE_SCHEMA",
    "GENERATION_INPUT_SCHEMA",
    "INIT_RECEIPT_SCHEMA",
    "PROTOCOL_SCHEMA",
    "REFERENCE_SCHEMA",
    "RESULT_SCHEMA",
    "RESULT_REQUEST_SCHEMA",
    "RESOLVED_CONFIG_SCHEMA",
    "RUN_SCHEMA",
    "RUN_SUMMARY_SCHEMA",
    "SOURCE_SCHEMA",
    "VIDEO_OUTPUT_SCHEMA",
    "WORK_MATERIALIZATION_SCHEMA",
    "WORKSPACE_SCHEMA",
    "canonical_sha256",
    "load_json_strict",
    "validate_document",
]
