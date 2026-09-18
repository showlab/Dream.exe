"""RoboCasa scene restoration from explicit, caller-owned inputs.

The helpers in this module are deliberately independent of repository data
discovery.  Importing the module does not import a simulator backend, and the
default restore path does not create or modify asset files.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import importlib
import json
from pathlib import Path
import re
from types import MethodType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_CAMERA_PROFILE = "cotraining"
DEFAULT_ACTIVE_CAMERA_NAME = "robot0_agentview_left"

_GLOBAL_ROBOCASA_RESTORE_WARNINGS_SEEN: set[tuple[str, str]] = set()


def default_robocasa_scene_override() -> dict[str, Any]:
    """Return a detached default scene-camera override."""

    return {
        "camera_profile": DEFAULT_CAMERA_PROFILE,
        "active_camera_name": DEFAULT_ACTIVE_CAMERA_NAME,
        "cam_configs": {},
        "camera_adjustments": {},
        "camera_reviewed": False,
        "camera_usable": False,
        "camera_override_metadata": {},
    }


def resolve_root_reference(
    reference: Mapping[str, Any],
    *,
    named_roots: Mapping[str, Any] | None = None,
) -> str:
    """Resolve one portable root reference without discovering any roots."""

    root_name = str(reference.get("root_key", "") or "").strip()
    configured_roots = named_roots or {}
    selected_root = str(configured_roots.get(root_name, "") or "").strip()
    if not selected_root:
        selected_root = str(reference.get("root_path", "") or "").strip()

    relative = str(reference.get("relative_path", "") or "").strip()
    if selected_root and relative:
        return (Path(selected_root) / relative).resolve().as_posix()

    absolute = str(reference.get("absolute_path", "") or "").strip()
    if absolute:
        return Path(absolute).resolve().as_posix()
    return ""


def _backend_cameras(robot_name: str) -> Mapping[str, Any]:
    """Load cotraining cameras only when that profile is requested."""

    importlib.import_module("robocasa")
    camera_utils = importlib.import_module("robocasa.utils.camera_utils")
    return camera_utils.get_robot_cam_configs(
        robot_name,
        use_cotraining_cameras=True,
    )


def _profile_name(scene_override: Mapping[str, Any]) -> str:
    requested = str(
        scene_override.get(
            "camera_profile",
            DEFAULT_CAMERA_PROFILE,
        )
        or DEFAULT_CAMERA_PROFILE
    ).strip()
    if requested in {"dataset", "custom"}:
        return requested
    return DEFAULT_CAMERA_PROFILE


def _overlay_camera_definitions(
    cameras: dict[str, Any],
    scene_override: Mapping[str, Any],
    *,
    profile: str,
) -> None:
    if profile == "dataset":
        return
    additions = scene_override.get("cam_configs", {}) or {}
    for name, supplied in dict(additions).items():
        previous = copy.deepcopy(dict(cameras.get(name, {}) or {}))
        previous.update(copy.deepcopy(dict(supplied or {})))
        cameras[name] = previous


def _euler_triplet(value: Any) -> np.ndarray:
    if isinstance(value, Mapping):
        components = (
            value.get("roll", value.get("x", 0.0)),
            value.get("pitch", value.get("y", 0.0)),
            value.get("yaw", value.get("z", 0.0)),
        )
        return np.asarray(components, dtype=float)
    components = np.asarray(value, dtype=float).reshape(-1)
    if components.size != 3:
        raise ValueError(f"camera euler override must have 3 values, got {components}")
    return components


def _rotated_wxyz(
    quaternion_wxyz: np.ndarray,
    delta_euler_degrees: Any,
) -> list[float]:
    current_xyzw = np.asarray(
        [
            quaternion_wxyz[1],
            quaternion_wxyz[2],
            quaternion_wxyz[3],
            quaternion_wxyz[0],
        ],
        dtype=float,
    )
    increment = Rotation.from_euler(
        "xyz",
        _euler_triplet(delta_euler_degrees),
        degrees=True,
    )
    rotated_xyzw = (Rotation.from_quat(current_xyzw) * increment).as_quat()
    return [
        float(rotated_xyzw[3]),
        float(rotated_xyzw[0]),
        float(rotated_xyzw[1]),
        float(rotated_xyzw[2]),
    ]


def _apply_camera_adjustments(
    cameras: dict[str, Any],
    scene_override: Mapping[str, Any],
) -> None:
    adjustments = scene_override.get("camera_adjustments", {}) or {}
    for name, raw_adjustment in dict(adjustments).items():
        if not isinstance(raw_adjustment, Mapping):
            continue
        adjustment = dict(raw_adjustment)
        copied_from = str(adjustment.get("copy_from", "") or "").strip()
        if copied_from and copied_from != name:
            if copied_from not in cameras:
                raise KeyError(
                    f"camera_adjustments[{name!r}].copy_from "
                    "refers to unknown camera "
                    f"{copied_from!r}"
                )
            camera = copy.deepcopy(dict(cameras[copied_from]))
        else:
            camera = copy.deepcopy(dict(cameras.get(name, {}) or {}))

        position_source = (
            adjustment["pos"]
            if adjustment.get("pos", None) is not None
            else camera.get("pos", [0.0, 0.0, 0.0])
        )
        quaternion_source = (
            adjustment["quat"]
            if adjustment.get("quat", None) is not None
            else camera.get("quat", [1.0, 0.0, 0.0, 0.0])
        )
        position = np.asarray(
            position_source,
            dtype=float,
        ).reshape(3)
        quaternion = np.asarray(
            quaternion_source,
            dtype=float,
        ).reshape(4)

        if adjustment.get("delta_pos", None) is not None:
            position = position + np.asarray(
                adjustment["delta_pos"],
                dtype=float,
            ).reshape(3)

        new_attributes = adjustment.get("camera_attribs", None)
        if isinstance(new_attributes, Mapping):
            attributes = copy.deepcopy(dict(camera.get("camera_attribs", {}) or {}))
            attributes.update(copy.deepcopy(dict(new_attributes)))
            camera["camera_attribs"] = attributes

        camera["pos"] = position.tolist()
        if adjustment.get("delta_euler_deg", None) is not None:
            camera["quat"] = _rotated_wxyz(
                quaternion,
                adjustment["delta_euler_deg"],
            )
        else:
            camera["quat"] = quaternion.tolist()
        cameras[name] = camera


def _choose_camera_names(
    cameras: Mapping[str, Any],
    bootstrap_names: list[str],
    requested_active: str,
) -> tuple[str, str]:
    if requested_active in cameras:
        active = requested_active
    elif DEFAULT_ACTIVE_CAMERA_NAME in cameras:
        active = DEFAULT_ACTIVE_CAMERA_NAME
    else:
        active = sorted(cameras)[0]

    if active in bootstrap_names:
        bootstrap = active
    elif DEFAULT_ACTIVE_CAMERA_NAME in bootstrap_names:
        bootstrap = DEFAULT_ACTIVE_CAMERA_NAME
    elif bootstrap_names:
        bootstrap = bootstrap_names[0]
    else:
        bootstrap = active
    return active, bootstrap


def resolve_robocasa_camera_setup(
    *,
    robot_name: str,
    ep_meta: Mapping[str, Any],
    scene_override: Mapping[str, Any],
    cotraining_camera_loader: (Callable[[str], Mapping[str, Any]] | None) = None,
) -> dict[str, Any]:
    """Resolve dataset, custom, or cotraining camera configuration."""

    profile = _profile_name(scene_override)
    dataset_cameras = copy.deepcopy(dict(ep_meta.get("cam_configs", {}) or {}))
    if profile == DEFAULT_CAMERA_PROFILE:
        load_cameras = (
            _backend_cameras
            if cotraining_camera_loader is None
            else cotraining_camera_loader
        )
        cameras = copy.deepcopy(dict(load_cameras(robot_name) or {}))
        uses_cotraining = True
    else:
        cameras = dataset_cameras
        uses_cotraining = False

    bootstrap_names = sorted(cameras)
    _overlay_camera_definitions(
        cameras,
        scene_override,
        profile=profile,
    )
    _apply_camera_adjustments(cameras, scene_override)
    if not cameras:
        raise RuntimeError(
            "No camera configs available after resolving RoboCasa camera profile."
        )

    requested_active = str(scene_override.get("active_camera_name", "") or "").strip()
    active_name, bootstrap_name = _choose_camera_names(
        cameras,
        bootstrap_names,
        requested_active,
    )
    return {
        "camera_profile": profile,
        "active_camera_name": active_name,
        "bootstrap_camera_name": bootstrap_name,
        "bootstrap_camera_names": bootstrap_names,
        "cam_configs": copy.deepcopy(cameras),
        "active_camera_cfg": copy.deepcopy(cameras[active_name]),
        "use_cotraining_cameras": uses_cotraining,
    }


def _json_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return copy.deepcopy(dict(payload))


def resolve_robocasa_camera_setup_from_manifest(
    *,
    scene_restore: Mapping[str, Any],
    ep_meta: Mapping[str, Any],
    path_resolver: (Callable[[Mapping[str, Any]], str] | None) = None,
    named_roots: Mapping[str, Any] | None = None,
    scene_override_loader: (Callable[[str | Path], Mapping[str, Any]] | None) = None,
    cotraining_camera_loader: (Callable[[str], Mapping[str, Any]] | None) = None,
) -> dict[str, Any]:
    """Resolve camera setup using an optional explicit manifest override."""

    manifest = copy.deepcopy(dict(scene_restore))
    override = default_robocasa_scene_override()
    override["cam_configs"] = copy.deepcopy(
        dict(manifest.get("resolved_cam_configs", {}) or {})
    )

    reference = manifest.get("scene_override_ref", {}) or {}
    if isinstance(reference, Mapping) and reference:
        resolver = (
            (
                lambda item: resolve_root_reference(
                    item,
                    named_roots=named_roots,
                )
            )
            if path_resolver is None
            else path_resolver
        )
        override_path = resolver(reference)
        if override_path and Path(override_path).exists():
            load_override = (
                _json_mapping
                if scene_override_loader is None
                else scene_override_loader
            )
            override = copy.deepcopy(dict(load_override(override_path)))

    bootstrap = dict(manifest.get("bootstrap_env_kwargs", {}) or {})
    robot_name = str(bootstrap.get("robots", "PandaOmron"))
    return resolve_robocasa_camera_setup(
        robot_name=robot_name,
        ep_meta=ep_meta,
        scene_override=override,
        cotraining_camera_loader=cotraining_camera_loader,
    )


def apply_robocasa_episode_to_env(
    env: Any,
    *,
    ep_meta: Mapping[str, Any],
    cam_configs: Mapping[str, Any],
) -> dict[str, Any]:
    """Install detached episode metadata and camera definitions on an env."""

    effective_meta = copy.deepcopy(dict(ep_meta))
    effective_meta["cam_configs"] = copy.deepcopy(dict(cam_configs))

    preferred_setter = getattr(env, "set_ep_meta", None)
    alternate_setter = getattr(env, "set_attrs_from_ep_meta", None)
    if callable(preferred_setter):
        preferred_setter(effective_meta)
    elif callable(alternate_setter):
        alternate_setter(effective_meta)

    env._ep_meta = effective_meta
    env._cam_configs = copy.deepcopy(dict(cam_configs))
    return effective_meta


def _model_element_name(
    name_getter: Callable[[int], Any],
    element_id: int,
) -> str:
    try:
        value = name_getter(int(element_id))
    except Exception:
        value = None
    return str(value or "")


def _body_path(model: Any, body_id: int) -> list[int]:
    """Return a cycle-safe root-to-leaf body path."""

    parent_ids = np.asarray(
        getattr(model, "body_parentid", ()),
        dtype=np.int64,
    ).reshape(-1)
    current = int(body_id)
    leaf_to_root: list[int] = []
    seen: set[int] = set()
    while 0 <= current < parent_ids.size and current not in seen:
        seen.add(current)
        leaf_to_root.append(current)
        parent = int(parent_ids[current])
        if parent == current:
            break
        current = parent
    return list(reversed(leaf_to_root))


def _top_level_body_name(model: Any, body_id: int) -> str:
    path = _body_path(model, body_id)
    selected_id = path[1] if len(path) >= 2 else int(body_id)
    return _model_element_name(model.body_id2name, selected_id)


def _semantic_body_name(model: Any, body_id: int) -> str:
    path = _body_path(model, body_id)
    names = [
        _model_element_name(model.body_id2name, path_body_id) for path_body_id in path
    ]
    names = [name for name in names if name]
    fallback = _top_level_body_name(model, body_id) or (
        names[-1] if names else f"body_{int(body_id)}"
    )
    normalized_path = " ".join(name.lower() for name in names)
    if any(
        token in normalized_path
        for token in ("robot", "gripper", "mobilebase", "torso")
    ):
        preferred_grippers = [
            name
            for name in names
            if "gripper" in name.lower()
            and not any(
                excluded in name.lower()
                for excluded in (
                    "_eef",
                    "leftfinger",
                    "rightfinger",
                    "joint",
                    "_tip",
                )
            )
        ]
        if preferred_grippers:
            return preferred_grippers[-1]
        grippers = [name for name in names if "gripper" in name.lower()]
        if grippers:
            return grippers[-1]
        hands = [name for name in names if "hand" in name.lower()]
        if hands:
            return hands[-1]
        return fallback

    main_bodies = [name for name in names if name.endswith("_main") or "_main_" in name]
    return main_bodies[-1] if main_bodies else fallback


def _semantic_class_name(instance_name: str) -> str:
    name = str(instance_name or "").strip()
    if not name:
        return "unnamed"
    name = re.sub(r"_(left|right)finger$", "_finger", name)
    return re.sub(r"_\d+$", "", name)


def _regenerate_visual_id_maps(owner: Any, sim: Any) -> None:
    model = sim.model
    owner._instances_to_ids = {}
    owner._classes_to_ids = {}
    owner._geom_ids_to_instances = {}
    owner._geom_ids_to_classes = {}
    owner._site_ids_to_instances = {}
    owner._site_ids_to_classes = {}

    element_groups = (
        (
            "geom",
            getattr(model, "geom_bodyid", ()),
            owner._geom_ids_to_instances,
            owner._geom_ids_to_classes,
        ),
        (
            "site",
            getattr(model, "site_bodyid", ()),
            owner._site_ids_to_instances,
            owner._site_ids_to_classes,
        ),
    )
    for (
        element_kind,
        raw_body_ids,
        instance_reverse,
        class_reverse,
    ) in element_groups:
        body_ids = np.asarray(raw_body_ids, dtype=np.int64).reshape(-1)
        for element_id, body_id in enumerate(body_ids):
            instance_name = _semantic_body_name(model, int(body_id))
            class_name = _semantic_class_name(instance_name)
            owner._instances_to_ids.setdefault(
                instance_name,
                {"geom": [], "site": []},
            )
            owner._classes_to_ids.setdefault(
                class_name,
                {"geom": [], "site": []},
            )
            owner._instances_to_ids[instance_name][element_kind].append(int(element_id))
            owner._classes_to_ids[class_name][element_kind].append(int(element_id))
            instance_reverse[int(element_id)] = instance_name
            class_reverse[int(element_id)] = class_name


def _install_visual_id_rebuilder(env: Any) -> None:
    owner = getattr(env, "model", None)
    if owner is None:
        return
    owner.generate_id_mappings = MethodType(
        _regenerate_visual_id_maps,
        owner,
    )


def _restore_warning_summary(
    error: BaseException | str,
    *,
    max_length: int = 240,
) -> str:
    text = str(error).replace("\n", " ").strip()
    available_marker = ". Available "
    marker_index = text.find(available_marker)
    if marker_index >= 0:
        text = text[:marker_index].rstrip(". ")
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _warn_once_for_stale_hook(
    env: Any,
    name: str,
    error: BaseException,
) -> None:
    summary = _restore_warning_summary(error)
    warning_key = (name, summary)
    seen = getattr(env, "_robocasa_seen_restore_warnings", set())
    if warning_key in seen or warning_key in _GLOBAL_ROBOCASA_RESTORE_WARNINGS_SEEN:
        return
    seen.add(warning_key)
    env._robocasa_seen_restore_warnings = seen
    _GLOBAL_ROBOCASA_RESTORE_WARNINGS_SEEN.add(warning_key)
    print(
        "[ROBOCASA][WARN] Skipping stale "
        f"{name} reference during scene-backed restore: {summary}"
    )


def _guard_stale_environment_hooks(env: Any) -> None:
    if not hasattr(env, "_robocasa_seen_restore_warnings"):
        env._robocasa_seen_restore_warnings = set()

    for hook_name in ("visualize", "update_sites", "update_state"):
        installed_flag = f"_robocasa_safe_{hook_name}_installed"
        if getattr(env, installed_flag, False):
            continue
        original = getattr(env, hook_name, None)
        if not callable(original):
            continue

        def guarded(
            _self: Any,
            *args: Any,
            _hook: Callable[..., Any] = original,
            _name: str = hook_name,
            **kwargs: Any,
        ) -> Any:
            try:
                return _hook(*args, **kwargs)
            except ValueError as error:
                if _name == "visualize" and not any(
                    marker in str(error)
                    for marker in (
                        'No "site" with name ',
                        'No "geom" with name ',
                    )
                ):
                    raise
                _warn_once_for_stale_hook(_self, _name, error)
                return None
            except (AssertionError, AttributeError, KeyError) as error:
                if _name == "visualize":
                    raise
                _warn_once_for_stale_hook(_self, _name, error)
                return None

        setattr(env, hook_name, MethodType(guarded, env))
        setattr(env, installed_flag, True)


def _integer_leaves(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        values = value
    else:
        values = (value,)

    output: list[int] = []
    for item in values:
        if isinstance(item, Mapping) or isinstance(
            item,
            (list, tuple, set, np.ndarray),
        ):
            output.extend(_integer_leaves(item))
            continue
        try:
            output.append(int(item))
        except (TypeError, ValueError):
            continue
    return output


def _hide_robot_helper_sites(env: Any) -> list[str]:
    simulator = getattr(env, "sim", None)
    model = getattr(simulator, "model", None)
    if model is None:
        return []
    site_names = list(getattr(model, "site_names", ()) or ())
    hidden_ids: set[int] = set()
    for robot in list(getattr(env, "robots", ()) or ()):
        for attribute in ("eef_site_id", "eef_cylinder_id"):
            if hasattr(robot, attribute):
                hidden_ids.update(_integer_leaves(getattr(robot, attribute)))

    hidden_names: list[str] = []
    for site_id in sorted(hidden_ids):
        if site_id < 0 or site_id >= len(site_names):
            continue
        try:
            model.site_rgba[site_id, 3] = 0.0
            model.site_size[site_id] = 1e-8
        except (IndexError, TypeError, ValueError):
            continue
        hidden_names.append(str(site_names[site_id]))
    return hidden_names


def _apply_ripe_fruit_appearance(
    env: Any,
    episode_metadata: Mapping[str, Any],
) -> list[str]:
    instruction = str(episode_metadata.get("lang", "") or "").lower()
    if "non-rotten fruit" not in instruction:
        return []
    references = episode_metadata.get("refs", {}) or {}
    if not isinstance(references, Mapping) or not references.get("chosen_inst", None):
        return []
    object_configs = episode_metadata.get("object_cfgs", ()) or ()
    if not isinstance(object_configs, (list, tuple)) or not object_configs:
        return []
    first_config = object_configs[0]
    if not isinstance(first_config, Mapping):
        return []
    rotten_name = str(first_config.get("name", "") or "").strip()
    if not rotten_name:
        return []

    model = env.sim.model
    changed = 0
    for geom_id in range(int(getattr(model, "ngeom", 0))):
        geom_name = model.geom_id2name(geom_id)
        if (
            not geom_name
            or not str(geom_name).startswith(f"{rotten_name}_")
            or str(geom_name) == f"{rotten_name}_reg_bbox"
        ):
            continue
        model.geom_rgba[geom_id] = np.asarray(
            [0.4, 0.26, 0.14, 1.0],
            dtype=float,
        )
        changed += 1
    if changed == 0:
        return []
    env.sim.forward()
    return [f"choose_ripe_fruit_rotten_{rotten_name}:{changed}"]


def _restore_xml(
    env: Any,
    model_xml: str,
    *,
    asset_preparer: Callable[[str], list[str]] | None,
) -> list[str]:
    edited_xml = env.edit_model_xml(model_xml)
    extracted: list[str] = []
    if asset_preparer is not None:
        extracted = sorted(list(asset_preparer(edited_xml)))
        if extracted:
            print(f"[ROBOCASA] extracted objaverse assets on demand: {extracted}")
    _install_visual_id_rebuilder(env)
    _guard_stale_environment_hooks(env)
    env.reset_from_xml_string(edited_xml)
    env.sim.reset()
    env.sim.forward()
    return extracted


def _verification_error(
    env: Any,
    target: np.ndarray,
    tolerance: float,
) -> float:
    current = np.asarray(
        env.sim.get_state().flatten(),
        dtype=np.float64,
    ).reshape(-1)
    if current.shape != target.shape:
        raise RuntimeError(
            "RoboCasa restore verification dim mismatch: "
            f"current={current.shape} target={target.shape}"
        )
    difference = 0.0 if target.size == 0 else float(np.max(np.abs(current - target)))
    if difference > tolerance:
        raise RuntimeError(
            "RoboCasa restore verification failed: "
            f"max|cur-target|={difference:.3e} > "
            f"{tolerance:.3e}"
        )
    return difference


def restore_robocasa_scene(
    env: Any,
    *,
    model_xml: str,
    state0_flat: Any,
    ep_meta: Mapping[str, Any],
    cam_configs: Mapping[str, Any],
    verify: bool = True,
    verify_atol: float = 1e-9,
    asset_preparer: Callable[[str], list[str]] | None = None,
) -> dict[str, Any]:
    """Restore one RoboCasa XML scene and its first flattened state."""

    effective_meta = apply_robocasa_episode_to_env(
        env,
        ep_meta=ep_meta,
        cam_configs=cam_configs,
    )
    extracted_categories = _restore_xml(
        env,
        model_xml,
        asset_preparer=asset_preparer,
    )

    target = np.asarray(
        state0_flat,
        dtype=np.float64,
    ).reshape(-1)
    env.sim.set_state_from_flattened(target)
    env.sim.forward()

    hidden_sites = _hide_robot_helper_sites(env)
    visual_effects = _apply_ripe_fruit_appearance(
        env,
        effective_meta,
    )
    update_sites = getattr(env, "update_sites", None)
    if callable(update_sites):
        update_sites()
    update_state = getattr(env, "update_state", None)
    if callable(update_state):
        update_state()
    hidden_sites = _hide_robot_helper_sites(env)

    tolerance = float(verify_atol)
    verified_error = _verification_error(env, target, tolerance) if verify else 0.0
    return {
        "verified_max_abs_err": verified_error,
        "camera_names": sorted(dict(cam_configs)),
        "active_ep_meta_keys": sorted(effective_meta),
        "lazy_extracted_objaverse_categories": extracted_categories,
        "hidden_robot_sites": hidden_sites,
        "visual_effects": visual_effects,
    }


__all__ = [
    "DEFAULT_ACTIVE_CAMERA_NAME",
    "DEFAULT_CAMERA_PROFILE",
    "apply_robocasa_episode_to_env",
    "default_robocasa_scene_override",
    "resolve_robocasa_camera_setup",
    "resolve_robocasa_camera_setup_from_manifest",
    "resolve_root_reference",
    "restore_robocasa_scene",
]
