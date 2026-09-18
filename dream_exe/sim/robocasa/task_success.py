"""Explicit RoboCasa task-success compatibility adapters.

The functions in this module inspect only a caller-owned simulator
environment.  They do not resolve samples, UIDs, dataset roots, trajectory
paths, bench paths, or output destinations.
"""

from __future__ import annotations

import copy
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from dream_exe.evaluation.execution import (
    TaskSuccessCheckResult,
    TaskSuccessObservation,
    build_task_success_observation,
    cheesybread_metrics_from_components,
)


_JOINT_ALIAS_PATCH_LOCK = threading.RLock()


def resolve_robocasa_joint_alias(
    name: str,
    joint_names: Sequence[str],
) -> str | None:
    """Resolve the current RoboCasa group/stack joint aliases."""

    raw = str(name)
    normalized_joint_names = tuple(str(item) for item in joint_names)
    if raw in normalized_joint_names:
        return raw

    candidates = {raw}

    def expand_group_variants(value: str) -> list[str]:
        output: list[str] = []
        replacements = (
            ("_left_group_", "_right_group_"),
            ("_left_group_", "_main_group_"),
            ("_left_group_", "_island_group_"),
            ("_right_group_", "_left_group_"),
            ("_right_group_", "_main_group_"),
            ("_right_group_", "_island_group_"),
            ("_main_group_", "_left_group_"),
            ("_main_group_", "_right_group_"),
            ("_main_group_", "_island_group_"),
            ("_island_group_", "_left_group_"),
            ("_island_group_", "_right_group_"),
            ("_island_group_", "_main_group_"),
        )
        for source, target in replacements:
            if source in value:
                output.append(value.replace(source, target))

        loose_replacements = (
            ("left_group", "right_group"),
            ("left_group", "main_group"),
            ("left_group", "island_group"),
            ("right_group", "left_group"),
            ("right_group", "main_group"),
            ("right_group", "island_group"),
            ("main_group", "left_group"),
            ("main_group", "right_group"),
            ("main_group", "island_group"),
            ("island_group", "left_group"),
            ("island_group", "right_group"),
            ("island_group", "main_group"),
        )
        for source, target in loose_replacements:
            protected = f"_{source}_"
            if source in value and protected not in value:
                output.append(value.replace(source, target))
        return output

    def expand_stack_variants(value: str) -> list[str]:
        match = re.match(r"^(stack_)0?(\d+)(_.*)$", value)
        if match is None:
            return []
        number = int(match.group(2))
        return [f"{match.group(1)}{number:02d}{match.group(3)}"]

    frontier = [raw]
    for _ in range(2):
        next_frontier: list[str] = []
        for current in frontier:
            for candidate in expand_group_variants(current) + expand_stack_variants(
                current
            ):
                if candidate in candidates:
                    continue
                candidates.add(candidate)
                next_frontier.append(candidate)
        frontier = next_frontier

    for candidate in candidates:
        if candidate in normalized_joint_names:
            return candidate

    tokens = raw.split("_")
    maximum_suffix = min(8, len(tokens))
    for length in range(maximum_suffix, 3, -1):
        suffix = "_".join(tokens[-length:])
        matches = [
            joint_name
            for joint_name in normalized_joint_names
            if joint_name.endswith(suffix)
        ]
        if not matches:
            continue
        if len(matches) == 1:
            return matches[0]

        wanted_tokens = [
            key
            for key in (
                "stack",
                "fridge",
                "oven",
                "microwave",
                "dishwasher",
                "sink",
                "toaster",
                "cabinet",
            )
            if key in raw.lower()
        ]
        raw_tokens = {token.lower() for token in tokens if token}

        def score(joint_name: str) -> tuple[int, int]:
            value = 0
            lowered = joint_name.lower()
            for wanted in wanted_tokens:
                if wanted in lowered:
                    value += 5
            raw_stack = re.search(
                r"(stack)_0?(\d+)",
                raw.lower(),
            )
            candidate_stack = re.search(
                r"(stack)_0?(\d+)",
                lowered,
            )
            if (
                raw_stack is not None
                and candidate_stack is not None
                and int(raw_stack.group(2)) == int(candidate_stack.group(2))
            ):
                value += 20
            value += len(raw_tokens.intersection(set(lowered.split("_"))))
            return value, -len(joint_name)

        return sorted(
            matches,
            key=score,
            reverse=True,
        )[0]

    stack_match = re.search(
        r"(stack)_0?(\d+)",
        raw.lower(),
    )
    if stack_match is not None and raw.lower().endswith("_slidejoint"):
        stack_number = int(stack_match.group(2))
        drawer_index = None
        drawer_match = re.search(
            r"_(\d+)_slidejoint$",
            raw.lower(),
        )
        if drawer_match is not None:
            drawer_index = int(drawer_match.group(1))
        matches = []
        for joint_name in normalized_joint_names:
            lowered = joint_name.lower()
            if (
                f"stack_{stack_number}" not in lowered
                and f"stack_{stack_number:02d}" not in lowered
            ):
                continue
            if not lowered.endswith("_slidejoint"):
                continue
            if drawer_index is not None and not lowered.endswith(
                f"_{drawer_index}_slidejoint"
            ):
                continue
            matches.append(joint_name)
        if matches:
            return sorted(matches, key=len)[0]
    return None


def check_task_success_with_joint_aliases(
    env: Any,
    *,
    binding_model_type: type[Any] | None = None,
) -> TaskSuccessCheckResult:
    """Call ``env._check_success`` under the current temporary alias patch."""

    if binding_model_type is None:
        import robosuite.utils.binding_utils as binding_utils

        model_type = binding_utils.MjModel
    else:
        model_type = binding_model_type

    original_joint_name2id = getattr(
        model_type,
        "joint_name2id",
        None,
    )
    original_get_joint_qpos_addr = getattr(
        model_type,
        "get_joint_qpos_addr",
        None,
    )

    def patched_joint_name2id(model: Any, name: Any) -> Any:
        assert callable(original_joint_name2id)
        try:
            return original_joint_name2id(model, name)
        except ValueError as error:
            if 'No "joint" with name' not in str(error):
                raise
            alias = resolve_robocasa_joint_alias(
                str(name),
                tuple(getattr(model, "joint_names", []) or []),
            )
            if alias is None:
                raise
            return original_joint_name2id(model, alias)

    def patched_get_joint_qpos_addr(
        model: Any,
        name: Any,
    ) -> Any:
        assert callable(original_get_joint_qpos_addr)
        try:
            return original_get_joint_qpos_addr(model, name)
        except ValueError as error:
            if 'No "joint" with name' not in str(error):
                raise
            alias = resolve_robocasa_joint_alias(
                str(name),
                tuple(getattr(model, "joint_names", []) or []),
            )
            if alias is None:
                raise
            return original_get_joint_qpos_addr(model, alias)

    with _JOINT_ALIAS_PATCH_LOCK:
        try:
            if callable(original_joint_name2id):
                model_type.joint_name2id = patched_joint_name2id
            if callable(original_get_joint_qpos_addr):
                model_type.get_joint_qpos_addr = patched_get_joint_qpos_addr
            check_success = getattr(
                env,
                "_check_success",
                None,
            )
            if check_success is None:
                return TaskSuccessCheckResult(
                    raw_success=False,
                    error="env_missing__check_success",
                )
            return TaskSuccessCheckResult(
                raw_success=bool(check_success()),
                error=None,
            )
        except Exception as error:
            return TaskSuccessCheckResult(
                raw_success=False,
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            if callable(original_joint_name2id):
                model_type.joint_name2id = original_joint_name2id
            if callable(original_get_joint_qpos_addr):
                model_type.get_joint_qpos_addr = original_get_joint_qpos_addr


def _geom_ids_for_body(
    sim: Any,
    body_name: str,
) -> np.ndarray:
    model = sim.model
    try:
        body_id = int(model.body_name2id(body_name))
    except Exception:
        return np.zeros((0,), dtype=np.int32)
    parent_ids = np.asarray(
        model.body_parentid,
        dtype=np.int32,
    ).reshape(-1)
    in_subtree = np.zeros(
        (parent_ids.shape[0],),
        dtype=bool,
    )
    stack = [body_id]
    in_subtree[body_id] = True
    while stack:
        current = int(stack.pop())
        children = np.where(parent_ids == current)[0]
        for child in children:
            child_id = int(child)
            if not in_subtree[child_id]:
                in_subtree[child_id] = True
                stack.append(child_id)
    geom_body_ids = np.asarray(
        model.geom_bodyid,
        dtype=np.int32,
    ).reshape(-1)
    return np.where(in_subtree[geom_body_ids])[0].astype(np.int32)


def _has_any_contact(
    sim: Any,
    first_geom_ids: np.ndarray,
    second_geom_ids: np.ndarray,
) -> bool:
    if first_geom_ids.size == 0 or second_geom_ids.size == 0:
        return False
    first = {int(value) for value in first_geom_ids.reshape(-1)}
    second = {int(value) for value in second_geom_ids.reshape(-1)}
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        first_id = int(contact.geom1)
        second_id = int(contact.geom2)
        if (first_id in first and second_id in second) or (
            second_id in first and first_id in second
        ):
            return True
    return False


def _max_box_half_xy(
    sim: Any,
    geom_ids: np.ndarray,
) -> float:
    if geom_ids.size == 0:
        return 0.0
    sizes = np.asarray(
        sim.model.geom_size,
        dtype=np.float64,
    )
    maximum = 0.0
    for geom_id in geom_ids.reshape(-1):
        half_size = sizes[int(geom_id)]
        if half_size.size >= 2:
            maximum = max(
                maximum,
                float(half_size[0]),
                float(half_size[1]),
            )
    return float(maximum)


def _geom_world_corners(
    sim: Any,
    geom_ids: np.ndarray,
) -> np.ndarray:
    if geom_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    sizes = np.asarray(
        sim.model.geom_size,
        dtype=np.float64,
    )
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    output: list[np.ndarray] = []
    for geom_id in geom_ids.reshape(-1):
        index = int(geom_id)
        try:
            position = np.asarray(
                sim.data.geom_xpos[index],
                dtype=np.float64,
            ).reshape(3)
            rotation = np.asarray(
                sim.data.geom_xmat[index],
                dtype=np.float64,
            ).reshape(3, 3)
            size = np.asarray(
                sizes[index],
                dtype=np.float64,
            ).reshape(-1)
        except Exception:
            continue
        if size.size == 0:
            continue
        size_x = abs(float(size[0]))
        size_y = (
            abs(float(size[1])) if size.size > 1 and float(size[1]) > 0.0 else size_x
        )
        size_z = (
            abs(float(size[2]))
            if size.size > 2 and float(size[2]) > 0.0
            else max(size_x, size_y)
        )
        half_size = np.asarray(
            [size_x, size_y, size_z],
            dtype=np.float64,
        )
        local = signs * half_size.reshape(1, 3)
        output.append(position.reshape(1, 3) + local @ rotation.T)
    if not output:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(output, axis=0)


def _geom_center_world(
    sim: Any,
    geom_ids: np.ndarray,
    body_name: str = "",
) -> np.ndarray | None:
    positions: list[np.ndarray] = []
    for geom_id in np.asarray(
        geom_ids,
        dtype=np.int32,
    ).reshape(-1):
        try:
            position = np.asarray(
                sim.data.geom_xpos[int(geom_id)],
                dtype=np.float64,
            ).reshape(3)
            if np.all(np.isfinite(position)):
                positions.append(position)
        except Exception:
            continue
    if positions:
        return np.mean(
            np.stack(positions, axis=0),
            axis=0,
        )
    if body_name:
        try:
            position = np.asarray(
                sim.data.get_body_xpos(str(body_name)),
                dtype=np.float64,
            ).reshape(3)
            if np.all(np.isfinite(position)):
                return position
        except Exception:
            pass
    return None


def _point_in_oriented_xy_footprint(
    point_xy: np.ndarray,
    footprint_points_xy: np.ndarray,
    *,
    margin_m: float,
) -> tuple[bool, dict[str, Any]]:
    points = np.asarray(
        footprint_points_xy,
        dtype=np.float64,
    ).reshape(-1, 2)
    point = np.asarray(
        point_xy,
        dtype=np.float64,
    ).reshape(2)
    details: dict[str, Any] = {"margin_m": float(margin_m)}
    if (
        points.shape[0] < 3
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(point))
    ):
        details["error"] = "insufficient_or_nonfinite_points"
        return False, details

    center = np.mean(points, axis=0)
    centered = points - center.reshape(1, 2)
    basis = np.eye(2, dtype=np.float64)
    try:
        covariance = centered.T @ centered / max(1, int(points.shape[0] - 1))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        basis = np.asarray(
            eigenvectors[:, order],
            dtype=np.float64,
        )
        if abs(float(np.linalg.det(basis))) < 1.0e-9:
            basis = np.eye(2, dtype=np.float64)
    except Exception:
        basis = np.eye(2, dtype=np.float64)

    projection = centered @ basis
    point_projection = (point - center) @ basis
    lower = np.min(projection, axis=0) - float(margin_m)
    upper = np.max(projection, axis=0) + float(margin_m)
    signed_margin = np.minimum(
        point_projection - lower,
        upper - point_projection,
    )
    inside = bool(np.all(signed_margin >= 0.0))
    details.update(
        {
            "center_xy": center.tolist(),
            "basis_xy": basis.tolist(),
            "bounds_lo": lower.tolist(),
            "bounds_hi": upper.tolist(),
            "point_proj": point_projection.tolist(),
            "signed_margin_m": signed_margin.tolist(),
            "inside": bool(inside),
        }
    )
    return inside, details


def cheese_on_bread_oriented_footprint(
    env: Any,
    *,
    margin_m: float = 0.015,
) -> dict[str, Any]:
    """Extract the current oriented bread-footprint evidence."""

    sim = env.sim
    bread_body = "bread_main"
    cheese_body = "cheese_main"
    try:
        if hasattr(env, "objects") and isinstance(env.objects, dict):
            bread_object = env.objects.get("bread", None)
            cheese_object = env.objects.get("cheese", None)
            bread_name = (
                str(
                    getattr(
                        bread_object,
                        "name",
                        "",
                    )
                    or ""
                ).strip()
                if bread_object is not None
                else ""
            )
            cheese_name = (
                str(
                    getattr(
                        cheese_object,
                        "name",
                        "",
                    )
                    or ""
                ).strip()
                if cheese_object is not None
                else ""
            )
            if bread_name:
                bread_body = bread_name
            if cheese_name:
                cheese_body = cheese_name
        if hasattr(env, "obj_body_id") and isinstance(env.obj_body_id, dict):
            if "bread" in env.obj_body_id:
                bread_body = str(sim.model.body_id2name(int(env.obj_body_id["bread"])))
            if "cheese" in env.obj_body_id:
                cheese_body = str(
                    sim.model.body_id2name(int(env.obj_body_id["cheese"]))
                )
    except Exception:
        pass

    bread_geometries = _geom_ids_for_body(
        sim,
        bread_body,
    )
    cheese_geometries = _geom_ids_for_body(
        sim,
        cheese_body,
    )
    bread_corners = _geom_world_corners(
        sim,
        bread_geometries,
    )
    cheese_corners = _geom_world_corners(
        sim,
        cheese_geometries,
    )
    cheese_center = _geom_center_world(
        sim,
        cheese_geometries,
        cheese_body,
    )

    output: dict[str, Any] = {
        "body_names": {
            "bread": bread_body,
            "cheese": cheese_body,
        },
        "bread_geom_count": int(bread_geometries.size),
        "cheese_geom_count": int(cheese_geometries.size),
        "margin_m": float(margin_m),
    }
    if cheese_center is not None:
        output["cheese_geom_center_world"] = cheese_center.tolist()
    if bread_corners.size == 0 or cheese_center is None:
        output["ok"] = False
        output["error"] = "missing_bread_corners_or_cheese_center"
        return output

    inside, details = _point_in_oriented_xy_footprint(
        cheese_center[:2],
        bread_corners[:, :2],
        margin_m=float(margin_m),
    )
    output["oriented_footprint"] = details

    z_ok = None
    if cheese_corners.size > 0:
        bread_z_min = float(np.min(bread_corners[:, 2]))
        bread_z_max = float(np.max(bread_corners[:, 2]))
        cheese_z_min = float(np.min(cheese_corners[:, 2]))
        cheese_z_max = float(np.max(cheese_corners[:, 2]))
        center_over_top = float(cheese_center[2] - bread_z_max)
        bottom_over_top = float(cheese_z_min - bread_z_max)
        z_ok = bool(bottom_over_top <= 0.25 and center_over_top >= -0.05)
        output.update(
            {
                "bread_z_min_m": bread_z_min,
                "bread_z_max_m": bread_z_max,
                "cheese_z_min_m": cheese_z_min,
                "cheese_z_max_m": cheese_z_max,
                "cheese_center_over_bread_top_m": (center_over_top),
                "cheese_bottom_over_bread_top_m": (bottom_over_top),
                "z_ok": bool(z_ok),
                "z_policy": (
                    "cheese_bottom_within_25cm_above_bread_top_and_center_not_below"
                ),
            }
        )
    else:
        output["z_ok"] = None
        output["z_policy"] = "not_evaluated_missing_cheese_corners"
    output["ok"] = bool(inside and z_ok is not False)
    return output


def cheesybread_geometry_fallback(
    env: Any,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate the current hardcoded-body CheesyBread fallback."""

    sim = env.sim
    bread_body = "bread_main"
    cheese_body = "cheese_main"
    container_body = "bread_container_main"
    bread_geometries = _geom_ids_for_body(
        sim,
        bread_body,
    )
    cheese_geometries = _geom_ids_for_body(
        sim,
        cheese_body,
    )
    container_geometries = _geom_ids_for_body(
        sim,
        container_body,
    )
    bread_container_contact = _has_any_contact(
        sim,
        bread_geometries,
        container_geometries,
    )
    cheese_bread_contact = _has_any_contact(
        sim,
        cheese_geometries,
        bread_geometries,
    )
    try:
        bread_position = np.asarray(
            sim.data.get_body_xpos(bread_body),
            dtype=np.float64,
        ).reshape(3)
        container_position = np.asarray(
            sim.data.get_body_xpos(container_body),
            dtype=np.float64,
        ).reshape(3)
        cheese_position = np.asarray(
            sim.data.get_body_xpos(cheese_body),
            dtype=np.float64,
        ).reshape(3)
    except Exception:
        return False, {"error": "missing_required_bodies"}

    distance_xy = float(np.linalg.norm((bread_position - container_position)[:2]))
    threshold_xy = 0.7 * _max_box_half_xy(
        sim,
        container_geometries,
    )
    bread_distance_ok = bool(distance_xy < float(threshold_xy))
    bread_in_receptacle = bool(bread_container_contact and bread_distance_ok)
    try:
        eef_site_id = env.robots[0].eef_site_id["right"]
        eef_position = np.asarray(
            sim.data.site_xpos[eef_site_id],
            dtype=np.float64,
        ).reshape(3)
        gripper_distance = float(np.linalg.norm(eef_position - cheese_position))
    except Exception:
        gripper_distance = float("inf")
    gripper_far = bool(gripper_distance > 0.25)
    success = bool(bread_in_receptacle and gripper_far and cheese_bread_contact)
    return success, {
        "components_mode": "geom_fallback",
        "body_names": {
            "bread": bread_body,
            "cheese": cheese_body,
            "bread_container": container_body,
        },
        "bread_receptacle_contact": bool(bread_container_contact),
        "bread_vs_receptacle_dist_xy_m": distance_xy,
        "bread_receptacle_dist_xy_threshold_m": float(threshold_xy),
        "bread_in_receptacle_and": bool(bread_in_receptacle),
        "gripper_to_cheese_dist_m": float(gripper_distance),
        "gripper_far_threshold_m": 0.25,
        "gripper_far_from_cheese": bool(gripper_far),
        "cheese_bread_contact": bool(cheese_bread_contact),
        "all_three_and": bool(success),
    }


def extract_cheesybread_components(
    env: Any,
    *,
    object_utils: Any | None = None,
) -> dict[str, Any]:
    """Extract current CheesyBread components from one explicit env."""

    output: dict[str, Any] = {"task": "CheesyBread"}
    try:
        output["env_objects_keys"] = sorted(
            list((getattr(env, "objects", {}) or {}).keys())
        )
    except Exception:
        output["env_objects_keys"] = None
    try:
        output["env_obj_body_id_keys"] = sorted(
            list((getattr(env, "obj_body_id", {}) or {}).keys())
        )
    except Exception:
        output["env_obj_body_id_keys"] = None

    bread_in_receptacle = None
    bread_in_receptacle_error = None
    bread_contact = None
    bread_distance_xy = None
    bread_threshold_xy = None
    bread_container_radius = None
    bread_body = None
    bread_container_body = None
    bread_object_name = None
    bread_container_object_name = None
    bread_body_id_from_environment = None
    container_body_id_from_environment = None
    bread_body_id_from_model = None
    container_body_id_from_model = None
    bread_body_name_from_model = None
    container_body_name_from_model = None
    bread_body_id_mismatch = None
    container_body_id_mismatch = None
    bread_in_receptacle_like_object_utils = None
    try:
        if object_utils is None:
            import robocasa.utils.object_utils as runtime_utils
        else:
            runtime_utils = object_utils
        bread_in_receptacle = bool(
            runtime_utils.check_obj_in_receptacle(
                env,
                "bread",
                "bread_container",
            )
        )
        try:
            if (
                hasattr(env, "objects")
                and "bread" in env.objects
                and "bread_container" in env.objects
            ):
                bread_contact = bool(
                    env.check_contact(
                        env.objects["bread"],
                        env.objects["bread_container"],
                    )
                )
        except Exception:
            bread_contact = None
        try:
            sim = env.sim
            if (
                hasattr(env, "objects")
                and "bread" in env.objects
                and "bread_container" in env.objects
            ):
                bread_object_name = str(
                    getattr(
                        env.objects["bread"],
                        "name",
                        "",
                    )
                    or ""
                )
                bread_container_object_name = str(
                    getattr(
                        env.objects["bread_container"],
                        "name",
                        "",
                    )
                    or ""
                )
            if hasattr(env, "obj_body_id") and isinstance(env.obj_body_id, dict):
                if "bread" in env.obj_body_id:
                    bread_body_id_from_environment = int(env.obj_body_id["bread"])
                    bread_body = str(
                        sim.model.body_id2name(bread_body_id_from_environment)
                    )
                if "bread_container" in env.obj_body_id:
                    container_body_id_from_environment = int(
                        env.obj_body_id["bread_container"]
                    )
                    bread_container_body = str(
                        sim.model.body_id2name(container_body_id_from_environment)
                    )
            if bread_object_name:
                bread_body_id_from_model = int(
                    sim.model.body_name2id(bread_object_name)
                )
                bread_body_name_from_model = str(
                    sim.model.body_id2name(bread_body_id_from_model)
                )
            if bread_container_object_name:
                container_body_id_from_model = int(
                    sim.model.body_name2id(bread_container_object_name)
                )
                container_body_name_from_model = str(
                    sim.model.body_id2name(container_body_id_from_model)
                )
            if (
                bread_body_id_from_environment is not None
                and bread_body_id_from_model is not None
            ):
                bread_body_id_mismatch = bool(
                    bread_body_id_from_environment != bread_body_id_from_model
                )
            if (
                container_body_id_from_environment is not None
                and container_body_id_from_model is not None
            ):
                container_body_id_mismatch = bool(
                    container_body_id_from_environment != container_body_id_from_model
                )
            if (
                bread_body_id_from_environment is not None
                and container_body_id_from_environment is not None
            ):
                bread_position = np.asarray(
                    sim.data.body_xpos[bread_body_id_from_environment],
                    dtype=np.float64,
                ).reshape(3)
                container_position = np.asarray(
                    sim.data.body_xpos[container_body_id_from_environment],
                    dtype=np.float64,
                ).reshape(3)
                bread_distance_xy = float(
                    np.linalg.norm((bread_position - container_position)[:2])
                )
            try:
                receptacle = env.objects["bread_container"]
                bread_container_radius = float(
                    getattr(
                        receptacle,
                        "horizontal_radius",
                    )
                )
                bread_threshold_xy = float(bread_container_radius * 0.7)
            except Exception:
                bread_container_radius = None
                bread_threshold_xy = None
            if (
                bread_contact is not None
                and bread_threshold_xy is not None
                and bread_object_name
                and bread_container_object_name
            ):
                bread_model_id = int(sim.model.body_name2id(bread_object_name))
                container_model_id = int(
                    sim.model.body_name2id(bread_container_object_name)
                )
                bread_position = np.asarray(
                    sim.data.body_xpos[bread_model_id],
                    dtype=np.float64,
                ).reshape(3)
                container_position = np.asarray(
                    sim.data.body_xpos[container_model_id],
                    dtype=np.float64,
                ).reshape(3)
                model_distance_xy = float(
                    np.linalg.norm((bread_position - container_position)[:2])
                )
                bread_in_receptacle_like_object_utils = bool(
                    bool(bread_contact)
                    and model_distance_xy < float(bread_threshold_xy)
                )
        except Exception:
            bread_distance_xy = bread_distance_xy
    except Exception as error:
        bread_in_receptacle_error = f"{type(error).__name__}: {error}"

    output["bread_in_receptacle"] = bread_in_receptacle
    if bread_in_receptacle_error is not None:
        output["bread_in_receptacle_error"] = bread_in_receptacle_error
    optional_values = (
        ("bread_obj_name", bread_object_name),
        (
            "bread_container_obj_name",
            bread_container_object_name,
        ),
        (
            "bread_body_id_from_obj_body_id",
            bread_body_id_from_environment,
        ),
        (
            "bread_container_body_id_from_obj_body_id",
            container_body_id_from_environment,
        ),
        (
            "bread_body_id_from_model_name2id",
            bread_body_id_from_model,
        ),
        (
            "bread_container_body_id_from_model_name2id",
            container_body_id_from_model,
        ),
        (
            "bread_body_name_from_model_name2id",
            bread_body_name_from_model,
        ),
        (
            "bread_container_body_name_from_model_name2id",
            container_body_name_from_model,
        ),
        (
            "bread_obj_body_id_mismatch",
            bread_body_id_mismatch,
        ),
        (
            "bread_container_obj_body_id_mismatch",
            container_body_id_mismatch,
        ),
        ("bread_container_contact", bread_contact),
        (
            "bread_body_name_from_obj_body_id",
            bread_body,
        ),
        (
            "bread_container_body_name_from_obj_body_id",
            bread_container_body,
        ),
        (
            "bread_vs_bread_container_dist_xy_m",
            bread_distance_xy,
        ),
        (
            "bread_container_horizontal_radius_m",
            bread_container_radius,
        ),
        (
            "bread_in_receptacle_threshold_xy_m",
            bread_threshold_xy,
        ),
    )
    for key, value in optional_values:
        if value is not None and value != "":
            output[key] = value
    if (
        bread_contact is not None
        and bread_distance_xy is not None
        and bread_threshold_xy is not None
    ):
        output["bread_in_receptacle_contact_and_dist"] = bool(
            bread_contact and bread_distance_xy < bread_threshold_xy
        )
    if bread_in_receptacle_like_object_utils is not None:
        output["bread_in_receptacle_like_ou_model_ids"] = bool(
            bread_in_receptacle_like_object_utils
        )

    gripper_far = None
    gripper_distance = None
    gripper_error = None
    try:
        eef_site_id = env.robots[0].eef_site_id["right"]
        eef_position = np.asarray(
            env.sim.data.site_xpos[eef_site_id],
            dtype=np.float64,
        ).reshape(3)
        if hasattr(env, "obj_body_id") and "cheese" in env.obj_body_id:
            cheese_body_id = int(env.obj_body_id["cheese"])
            cheese_position = np.asarray(
                env.sim.data.body_xpos[cheese_body_id],
                dtype=np.float64,
            ).reshape(3)
        else:
            cheese_body_id = int(env.sim.model.body_name2id("cheese_main"))
            cheese_position = np.asarray(
                env.sim.data.body_xpos[cheese_body_id],
                dtype=np.float64,
            ).reshape(3)
        gripper_distance = float(np.linalg.norm(eef_position - cheese_position))
        gripper_far = bool(gripper_distance > 0.25)
    except Exception as error:
        gripper_error = f"{type(error).__name__}: {error}"
    output["gripper_far_from_cheese"] = gripper_far
    output["gripper_far_threshold_m"] = 0.25
    if gripper_distance is not None:
        output["gripper_to_cheese_dist_m"] = float(gripper_distance)
    if gripper_error is not None:
        output["gripper_to_cheese_error"] = gripper_error

    contact_ok = None
    contact_error = None
    try:
        if (
            hasattr(env, "objects")
            and "cheese" in env.objects
            and "bread" in env.objects
        ):
            contact_ok = bool(
                env.check_contact(
                    env.objects["cheese"],
                    env.objects["bread"],
                )
            )
        else:
            cheese_geometries = _geom_ids_for_body(
                env.sim,
                "cheese_main",
            )
            bread_geometries = _geom_ids_for_body(
                env.sim,
                "bread_main",
            )
            contact_ok = bool(
                _has_any_contact(
                    env.sim,
                    cheese_geometries,
                    bread_geometries,
                )
            )
    except Exception as error:
        contact_error = f"{type(error).__name__}: {error}"
    output["cheese_bread_contact"] = contact_ok
    if contact_error is not None:
        output["cheese_bread_contact_error"] = contact_error

    try:
        sim = env.sim
        bread_position = None
        cheese_position = None
        if hasattr(env, "obj_body_id") and isinstance(env.obj_body_id, dict):
            if "bread" in env.obj_body_id:
                bread_position = np.asarray(
                    sim.data.body_xpos[int(env.obj_body_id["bread"])],
                    dtype=np.float64,
                ).reshape(3)
            if "cheese" in env.obj_body_id:
                cheese_position = np.asarray(
                    sim.data.body_xpos[int(env.obj_body_id["cheese"])],
                    dtype=np.float64,
                ).reshape(3)
        if bread_position is None:
            bread_position = np.asarray(
                sim.data.get_body_xpos("bread_main"),
                dtype=np.float64,
            ).reshape(3)
        if cheese_position is None:
            cheese_position = np.asarray(
                sim.data.get_body_xpos("cheese_main"),
                dtype=np.float64,
            ).reshape(3)
        cheese_to_bread_xy = float(
            np.linalg.norm((cheese_position - bread_position)[:2])
        )
        cheese_to_bread_z = float(cheese_position[2] - bread_position[2])
        output["cheese_to_bread_xy_m"] = float(cheese_to_bread_xy)
        output["cheese_to_bread_z_m"] = float(cheese_to_bread_z)

        alignment_threshold = None
        try:
            if hasattr(env, "objects") and isinstance(env.objects, dict):
                bread_object = env.objects.get(
                    "bread",
                    None,
                )
                cheese_object = env.objects.get(
                    "cheese",
                    None,
                )
                bread_radius = (
                    float(
                        getattr(
                            bread_object,
                            "horizontal_radius",
                        )
                    )
                    if (
                        bread_object is not None
                        and getattr(
                            bread_object,
                            "horizontal_radius",
                            None,
                        )
                        is not None
                    )
                    else None
                )
                cheese_radius = (
                    float(
                        getattr(
                            cheese_object,
                            "horizontal_radius",
                        )
                    )
                    if (
                        cheese_object is not None
                        and getattr(
                            cheese_object,
                            "horizontal_radius",
                            None,
                        )
                        is not None
                    )
                    else None
                )
                if bread_radius is not None and cheese_radius is not None:
                    alignment_threshold = max(
                        0.03,
                        0.5 * (bread_radius + cheese_radius),
                    )
                elif bread_radius is not None:
                    alignment_threshold = max(
                        0.03,
                        float(bread_radius),
                    )
                elif cheese_radius is not None:
                    alignment_threshold = max(
                        0.03,
                        float(cheese_radius),
                    )
        except Exception:
            alignment_threshold = None
        if alignment_threshold is None:
            alignment_threshold = 0.03
        output["cheese_alignment_threshold_xy_m"] = float(alignment_threshold)
    except Exception as error:
        output["cheese_to_bread_error"] = f"{type(error).__name__}: {error}"

    try:
        _success, geometry_details = cheesybread_geometry_fallback(env)
        output["geom_fallback"] = geometry_details
    except Exception as error:
        output["geom_fallback_error"] = f"{type(error).__name__}: {error}"

    try:
        footprint = cheese_on_bread_oriented_footprint(
            env,
            margin_m=0.015,
        )
        output["cheese_bread_oriented_footprint"] = footprint
        output["cheese_center_in_bread_oriented_footprint"] = bool(footprint.get("ok"))
        output["cheese_bread_oriented_footprint_margin_m"] = float(
            footprint.get("margin_m", 0.015)
        )
        if footprint.get("z_ok", None) is not None:
            output["cheese_bread_oriented_footprint_z_ok"] = bool(footprint.get("z_ok"))
        if "cheese_geom_center_world" in footprint:
            output["cheese_geom_center_world"] = footprint["cheese_geom_center_world"]
    except Exception as error:
        output["cheese_bread_oriented_footprint_error"] = (
            f"{type(error).__name__}: {error}"
        )

    if (
        bread_in_receptacle is not None
        and gripper_far is not None
        and contact_ok is not None
    ):
        output["all_three_and"] = bool(
            bread_in_receptacle and gripper_far and contact_ok
        )
    else:
        output["all_three_and"] = None
    return output


def extract_cheesybread_success_evidence(
    env: Any,
    *,
    object_utils: Any | None = None,
) -> dict[str, Any]:
    """Return components plus current pure metrics for evidence_reader."""

    components = extract_cheesybread_components(
        env,
        object_utils=object_utils,
    )
    return {
        "task_success_components": components,
        "metrics": cheesybread_metrics_from_components(components),
    }


def observe_robocasa_cheesybread_success(
    env: Any,
    *,
    trajectory_objects: Mapping[str, Any] | None = None,
    object_utils: Any | None = None,
    binding_model_type: type[Any] | None = None,
    checker: (Callable[[Any], TaskSuccessCheckResult] | None) = None,
) -> TaskSuccessObservation:
    """Run the current CheesyBread checker/component fallback contract."""

    if checker is None:
        check_result = check_task_success_with_joint_aliases(
            env,
            binding_model_type=binding_model_type,
        )
    else:
        check_result = checker(env)
    if not isinstance(check_result, TaskSuccessCheckResult):
        raise TypeError("checker must return TaskSuccessCheckResult")

    if check_result.error is None:
        components = extract_cheesybread_components(
            env,
            object_utils=object_utils,
        )
        evidence: dict[str, Any] = {"task_success_components": components}
        try:
            environment_components = getattr(
                env,
                "_last_success_components",
                None,
            )
            if isinstance(environment_components, dict) and environment_components:
                evidence["task_success_components_env"] = copy.deepcopy(
                    environment_components
                )
        except Exception:
            pass
        if trajectory_objects:
            evidence["trajectory_objects"] = copy.deepcopy(dict(trajectory_objects))
        return build_task_success_observation(
            "CheesyBread",
            raw_success=check_result.raw_success,
            meta=evidence,
        )

    try:
        components = extract_cheesybread_components(
            env,
            object_utils=object_utils,
        )
        fallback_success = components.get(
            "all_three_and",
            None,
        )
        if fallback_success is None:
            (
                fallback_success,
                geometry_details,
            ) = cheesybread_geometry_fallback(env)
            components["geom_fallback"] = geometry_details
        return build_task_success_observation(
            "CheesyBread",
            raw_success=False,
            checker_error=check_result.error,
            meta={
                "task_success_components": components,
            },
            calibration_raw_success=bool(fallback_success),
            success_impl="CheesyBread.components",
        )
    except Exception as error:
        return TaskSuccessObservation(
            task_check_success=False,
            strict_task_check_success=None,
            calibrated_task_check_success=None,
            meta={"success_impl": ("CheesyBread.geom_fallback")},
            error=f"{type(error).__name__}: {error}",
        )


__all__ = [
    "check_task_success_with_joint_aliases",
    "cheese_on_bread_oriented_footprint",
    "cheesybread_geometry_fallback",
    "extract_cheesybread_components",
    "extract_cheesybread_success_evidence",
    "observe_robocasa_cheesybread_success",
    "resolve_robocasa_joint_alias",
]
