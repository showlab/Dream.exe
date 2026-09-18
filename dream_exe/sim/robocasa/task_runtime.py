"""Explicit RoboCasa task-environment preparation for success replay.

This adapter reads only a caller-supplied simulator manifest and explicit root
options.  It does not discover a bench, UID, repository config, action,
trajectory, or output path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import json
from pathlib import Path
from typing import Any

from ..frozen.restore import resolve_scene_manifest_path
from .restore import resolve_root_reference
from .sink_runtime import prepare_sink_runtime_compat


EnvironmentFactory = Callable[..., Any]
EnvironmentRestorer = Callable[..., Any]


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"RoboCasa episode metadata must be a mapping: {source}")
    return copy.deepcopy(dict(payload))


def _task_environment_name(
    root: Mapping[str, Any],
    scene_restore: Mapping[str, Any],
) -> str:
    bootstrap = dict(
        scene_restore.get(
            "bootstrap_env_kwargs",
            {},
        )
        or {}
    )
    return str(
        scene_restore.get("dataset_env_name", "")
        or scene_restore.get("bootstrap_env_name", "")
        or bootstrap.get("env_name", "")
        or root.get("env_name", "")
        or ""
    ).strip()


def load_robocasa_task_runtime(
    simulator_config: Mapping[str, Any],
    *,
    scene_restore_options: Mapping[str, Any] | None = None,
    ep_meta_loader: Callable[[str | Path], Any] | None = None,
) -> dict[str, Any] | None:
    """Load task bootstrap inputs from one explicit simulator manifest.

    Non-RoboCasa manifests return ``None``.  For RoboCasa, path-reference
    precedence and explicit-root conflict checks match scene restoration.
    """

    root = dict(simulator_config.get("raw", simulator_config) or {})
    scene_restore = dict(root.get("scene_restore", {}) or {})
    backend = str(scene_restore.get("backend", "") or "").strip()
    if backend != "robocasa_frozen":
        return None

    environment_name = _task_environment_name(
        root,
        scene_restore,
    )
    if not environment_name:
        raise RuntimeError(
            "RoboCasa task-success replay requires a task "
            "environment name in the simulator manifest"
        )

    options = dict(scene_restore_options or {})
    paths = dict(scene_restore.get("paths", {}) or {})
    path_refs = dict(scene_restore.get("path_refs", {}) or {})
    named_roots = dict(options.get("named_roots", {}) or {})

    custom_resolver = options.get("path_resolver", None)
    if custom_resolver is not None and not callable(custom_resolver):
        raise TypeError("scene_restore_options.path_resolver must be callable")
    resolver = (
        custom_resolver
        if custom_resolver is not None
        else (
            lambda reference: resolve_root_reference(
                reference,
                named_roots=named_roots,
            )
        )
    )
    episode_metadata_path = resolve_scene_manifest_path(
        "ep_meta_json",
        paths=paths,
        path_refs=path_refs,
        path_resolver=resolver,
    )
    if not str(episode_metadata_path or "").strip():
        raise RuntimeError(
            "RoboCasa task-success replay requires raw.scene_restore.paths.ep_meta_json"
        )

    option_loader = options.get("ep_meta_loader", None)
    load_metadata = (
        ep_meta_loader
        if ep_meta_loader is not None
        else option_loader
        if option_loader is not None
        else _load_json_mapping
    )
    if not callable(load_metadata):
        raise TypeError("RoboCasa episode metadata loader must be callable")
    metadata = load_metadata(episode_metadata_path)
    if not isinstance(metadata, Mapping):
        raise ValueError("RoboCasa episode metadata loader must return a mapping")
    return {
        "backend": backend,
        "env_name": environment_name,
        "ep_meta": copy.deepcopy(dict(metadata)),
        "ep_meta_path": (Path(episode_metadata_path).expanduser().resolve().as_posix()),
    }


def apply_task_refs_to_environment(
    env: Any,
    ep_meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the current post-restore RoboCasa ``task_refs`` contract."""

    output: dict[str, Any] = {"applied": False}
    if not isinstance(ep_meta, Mapping):
        return output
    task_refs = ep_meta.get("task_refs", None)
    if not isinstance(task_refs, Mapping) or not task_refs:
        return output

    applied_keys: list[str] = []
    for raw_key, value in task_refs.items():
        key = str(raw_key or "").strip()
        if not key or key.startswith("_"):
            continue
        try:
            setattr(env, key, value)
            applied_keys.append(key)
        except Exception:
            continue
    output["applied"] = bool(applied_keys)
    output["keys"] = applied_keys
    output["task_refs"] = copy.deepcopy(dict(task_refs))
    return output


def repair_environment_object_body_ids(
    env: Any,
) -> dict[str, Any] | None:
    """Repair current RoboCasa object body IDs after XML restoration."""

    simulator = getattr(env, "sim", None)
    objects = getattr(env, "objects", None)
    object_body_ids = getattr(
        env,
        "obj_body_id",
        None,
    )
    if simulator is None or not isinstance(objects, dict) or object_body_ids is None:
        return None
    model = getattr(simulator, "model", None)
    if model is None:
        return None

    repairs: list[dict[str, Any]] = []
    for raw_key, object_model in objects.items():
        try:
            key = str(raw_key)
        except Exception:
            continue
        try:
            object_name = str(getattr(object_model, "name", "") or "")
        except Exception:
            object_name = ""
        if not key or not object_name:
            continue
        try:
            new_body_id = int(model.body_name2id(object_name))
        except Exception:
            continue
        old_body_id = (
            object_body_ids.get(key, None)
            if isinstance(object_body_ids, dict)
            else None
        )
        try:
            old_body_id_value = int(old_body_id) if old_body_id is not None else None
        except Exception:
            old_body_id_value = None
        if old_body_id_value == new_body_id:
            continue
        old_body_name = None
        if old_body_id_value is not None:
            try:
                old_body_name = str(model.body_id2name(old_body_id_value))
            except Exception:
                old_body_name = None
        try:
            new_body_name = str(model.body_id2name(new_body_id))
        except Exception:
            new_body_name = None
        try:
            object_body_ids[key] = new_body_id
        except Exception:
            continue
        repairs.append(
            {
                "key": key,
                "obj_name": object_name,
                "old_body_id": old_body_id_value,
                "old_body_name": old_body_name,
                "new_body_id": int(new_body_id),
                "new_body_name": new_body_name,
            }
        )
    return {
        "applied": bool(repairs),
        "repairs": repairs,
    }


def prepare_robocasa_task_runtime_hooks(
    task_runtime: Mapping[str, Any],
    *,
    env_factory: EnvironmentFactory,
    restorer: EnvironmentRestorer,
) -> tuple[
    EnvironmentFactory,
    EnvironmentRestorer,
    dict[str, Any],
]:
    """Wrap explicit environment hooks with current task bootstrap timing."""

    if not callable(env_factory):
        raise TypeError("env_factory must be callable")
    if not callable(restorer):
        raise TypeError("restorer must be callable")
    runtime = copy.deepcopy(dict(task_runtime or {}))
    environment_name = str(runtime.get("env_name", "") or "").strip()
    episode_metadata = runtime.get("ep_meta", None)
    if not environment_name:
        raise ValueError("task_runtime.env_name is required")
    if not isinstance(episode_metadata, Mapping):
        raise ValueError("task_runtime.ep_meta must be a mapping")
    detached_metadata = copy.deepcopy(dict(episode_metadata))
    runtime_meta: dict[str, Any] = {}

    def task_environment_factory(
        **kwargs: Any,
    ) -> Any:
        options = dict(kwargs)
        options["env_name"] = environment_name
        environment = env_factory(**options)
        try:
            setattr(
                environment,
                "_ep_meta",
                copy.deepcopy(detached_metadata),
            )
        except BaseException:
            try:
                environment.close()
            except Exception:
                pass
            raise
        runtime_meta["loaded_ep_meta"] = True
        return environment

    def task_environment_restorer(
        env: Any,
        config: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        result = restorer(
            env,
            config,
            **kwargs,
        )
        try:
            setattr(
                env,
                "_ep_meta",
                copy.deepcopy(detached_metadata),
            )
            runtime_meta["loaded_ep_meta_post_restore"] = True
        except Exception:
            runtime_meta["loaded_ep_meta_post_restore"] = False
        runtime_meta["task_refs_applied_post_restore"] = apply_task_refs_to_environment(
            env,
            detached_metadata,
        )
        object_repairs = repair_environment_object_body_ids(env)
        if isinstance(object_repairs, dict):
            runtime_meta["obj_body_id_repair"] = object_repairs
        runtime_meta.update(prepare_sink_runtime_compat(env))
        return result

    return (
        task_environment_factory,
        task_environment_restorer,
        runtime_meta,
    )


__all__ = [
    "EnvironmentFactory",
    "EnvironmentRestorer",
    "apply_task_refs_to_environment",
    "load_robocasa_task_runtime",
    "prepare_robocasa_task_runtime_hooks",
    "repair_environment_object_body_ids",
]
