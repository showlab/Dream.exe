"""Detached, side-effect-free RoboCasa task-success component readers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUPPORTED_ROBOCASA_COMPONENT_TASKS = frozenset(
    {
        "SlideOvenRack",
        "SlideToasterOvenRack",
        "OpenStandMixerHead",
        "CloseStandMixerHead",
        "TurnOnToaster",
        "TurnSinkSpout",
        "CloseDrawer",
        "CloseMicrowave",
        "CloseOven",
        "CloseFridgeDrawer",
        "PickPlaceCounterToOven",
        "PickPlaceCounterToBlender",
        "MakeIcedCoffee",
        "BreadAndCheese",
        "BreadSetupSlicing",
        "PlaceVegetablesEvenly",
        "CoffeeServeMug",
        "PackDessert",
        "OpenBlenderLid",
    }
)


def extract_robocasa_task_success_components(
    env: Any,
    task_name: str,
) -> dict[str, Any]:
    """Read supported RoboCasa component inputs without calling the checker."""
    task = str(task_name or "").strip()
    if task not in SUPPORTED_ROBOCASA_COMPONENT_TASKS:
        raise ValueError(f"unsupported RoboCasa component task: {task!r}")
    if task in {"OpenStandMixerHead", "CloseStandMixerHead"}:
        return _extract_stand_mixer_head_components(env, task)
    if task == "TurnOnToaster":
        return _extract_toaster_components(env)
    if task == "TurnSinkSpout":
        return _extract_sink_spout_components(env)
    if task == "CloseDrawer":
        return _extract_close_drawer_components(env)
    if task in {"CloseMicrowave", "CloseOven"}:
        return _extract_close_door_components(env, task)
    if task == "CloseFridgeDrawer":
        return _extract_close_fridge_drawer_components(env)
    if task == "PickPlaceCounterToOven":
        return _extract_pick_place_counter_to_oven_components(env)
    if task == "PickPlaceCounterToBlender":
        return _extract_pick_place_counter_to_blender_components(env)
    if task == "MakeIcedCoffee":
        return _extract_make_iced_coffee_components(env)
    if task == "BreadAndCheese":
        return _extract_bread_and_cheese_components(env)
    if task == "BreadSetupSlicing":
        return _extract_bread_setup_slicing_components(env)
    if task == "PlaceVegetablesEvenly":
        return _extract_place_vegetables_evenly_components(env)
    if task == "CoffeeServeMug":
        return _extract_coffee_serve_mug_components(env)
    if task == "PackDessert":
        return _extract_pack_dessert_components(env)
    if task == "OpenBlenderLid":
        return _extract_open_blender_lid_components(env)
    fixture_name = "oven" if task == "SlideOvenRack" else "toaster_oven"
    fixture = getattr(env, fixture_name)
    should_pull = bool(getattr(env, "should_pull"))
    rack_level = int(getattr(env, "rack_level"))
    state = fixture.get_state(rack_level=rack_level)
    prefixes = ("rack",) if task == "SlideOvenRack" else ("rack", "tray")
    key = next((key for key in state if str(key).startswith(prefixes)), None)
    current = None if key is None else state[key]
    numeric_current = None
    if current is not None:
        try:
            numeric_current = float(current)
        except (TypeError, ValueError):
            # The current checker raises during float conversion and generic
            # replay absorbs that checker error. Fail closed here so optional
            # detached evidence never fabricates a primary payload or
            # re-escalates that already-contained failure.
            raise ValueError("rack state value is not numeric") from None
    progress = None
    if numeric_current is not None:
        try:
            value = max(0.0, min(1.0, numeric_current))
            progress = value if should_pull else 1.0 - value
        except (TypeError, ValueError):
            pass
    success = False
    if numeric_current is not None:
        try:
            success = (
                numeric_current >= 0.95 if should_pull else numeric_current <= 0.05
            )
        except (TypeError, ValueError):
            pass
    return {
        "task": task,
        "should_pull": should_pull,
        "rack_level": rack_level,
        "setup_scene_errors": getattr(env, "_setup_scene_errors", None) or None,
        "state_key": None if key is None else str(key),
        "current_pos": numeric_current,
        "progress_fraction": progress,
        "progress_pct": None if progress is None else float(progress * 100.0),
        "threshold_pull_ge": 0.95,
        "threshold_push_le": 0.05,
        "all_and": bool(success),
    }


def _extract_stand_mixer_head_components(
    env: Any,
    task: str,
) -> dict[str, Any]:
    fixture = getattr(env, "stand_mixer")
    state = dict(fixture.get_state(env) or {})
    head = state.get("head", None)
    joint_used = None
    if head is None:
        try:
            names = list(getattr(env.sim.model, "joint_names", []) or [])
            candidates = [
                str(name)
                for name in names
                if str(name).endswith("head_joint") and "mixer" in str(name).lower()
            ]
            if candidates:
                joint_used = sorted(candidates)[0]
                head = float(fixture.get_joint_state(env, [joint_used])[joint_used])
        except Exception:
            head = None
    if head is not None:
        head = float(head)
    opening = task == "OpenStandMixerHead"
    threshold = 0.99 if opening else 0.01
    success = bool(
        head is not None and (head > threshold if opening else head < threshold)
    )
    output = {
        "task": task,
        "head": head,
        "head_joint_used": joint_used,
        "all_and": success,
    }
    if opening:
        output.update({"threshold_gt": threshold, "head_open": success})
    else:
        output.update({"threshold_lt": threshold, "head_closed": success})
    return output


def _extract_toaster_components(env: Any) -> dict[str, Any]:
    toaster = getattr(env, "toaster")
    slot_keys = list(toaster.get_state(env).keys())
    contacts: dict[str, bool] = {}
    selected = 0
    any_contact = False
    for slot_pair in range(len(slot_keys)):
        contact = bool(toaster.check_slot_contact(env, "obj", slot_pair))
        contacts[str(slot_pair)] = contact
        if contact and not any_contact:
            selected, any_contact = slot_pair, True
    turned_on = bool(toaster.get_state(env, slot_pair=selected)["turned_on"])
    return {
        "task": "TurnOnToaster",
        "num_slot_pairs": int(len(slot_keys)),
        "contacts_by_slot_pair": contacts,
        "toast_in_any_slot": bool(any_contact),
        "selected_slot_pair": int(selected),
        "turned_on_selected_slot": bool(turned_on),
        "all_and": bool(turned_on),
    }


def _extract_sink_spout_components(env: Any) -> dict[str, Any]:
    handle_state = getattr(env, "sink").get_handle_state(env=env)
    try:
        spout_ori = handle_state.get("spout_ori", None)
    except Exception:
        spout_ori = None
    filtered = {}
    if isinstance(handle_state, dict):
        for key in (
            "spout_ori",
            "spout_joint",
            "water_on",
            "handle_joint",
            "water_pressure",
            "water_temp",
        ):
            if key in handle_state:
                filtered[key] = handle_state.get(key)
    success = bool(spout_ori == getattr(env, "behavior"))
    return {
        "task": "TurnSinkSpout",
        "target_spout_ori": getattr(env, "behavior"),
        "spout_ori": spout_ori,
        "handle_state": filtered,
        "all_and": success,
    }


def _extract_close_drawer_components(env: Any) -> dict[str, Any]:
    drawer = getattr(env, "drawer")
    door_state = drawer.get_door_state(env=env)
    behavior = getattr(env, "behavior")
    open_threshold = 0.95
    close_threshold = 0.05
    close_threshold_relaxed = 0.25
    per_joint_ok: dict[str, bool] = {}
    per_joint_ok_strict: dict[str, bool] = {}
    per_joint_ok_relaxed: dict[str, bool] = {}
    open_fraction_max = None
    closed_pct_min = None
    for joint_name, joint_p in door_state.items():
        position = float(joint_p)
        if open_fraction_max is None:
            open_fraction_max = position
        else:
            open_fraction_max = max(float(open_fraction_max), position)
        closed_pct = 1.0 - position
        if closed_pct_min is None:
            closed_pct_min = closed_pct
        else:
            closed_pct_min = min(float(closed_pct_min), closed_pct)
        if behavior == "open":
            per_joint_ok[str(joint_name)] = bool(position >= open_threshold)
        elif behavior == "close":
            strict_ok = bool(position <= close_threshold)
            relaxed_ok = bool(position <= close_threshold_relaxed)
            per_joint_ok_strict[str(joint_name)] = strict_ok
            per_joint_ok_relaxed[str(joint_name)] = relaxed_ok
            per_joint_ok[str(joint_name)] = bool(strict_ok or relaxed_ok)
        else:
            per_joint_ok[str(joint_name)] = False
    success = bool(all(per_joint_ok.values())) if per_joint_ok else False
    return {
        "task": "ManipulateDrawer",
        "behavior": behavior,
        "drawer_side": getattr(env, "drawer_side", None),
        "open_threshold": float(open_threshold),
        "close_threshold": float(close_threshold),
        "close_threshold_relaxed": float(close_threshold_relaxed),
        "open_fraction_max": (
            None if open_fraction_max is None else float(open_fraction_max)
        ),
        "closed_pct_min": (None if closed_pct_min is None else float(closed_pct_min)),
        "door_state": {str(key): float(value) for key, value in door_state.items()},
        "per_joint_ok": per_joint_ok,
        "per_joint_ok_strict": (per_joint_ok_strict if per_joint_ok_strict else None),
        "per_joint_ok_relaxed": (
            per_joint_ok_relaxed if per_joint_ok_relaxed else None
        ),
        "all_and": bool(success),
    }


def _extract_close_door_components(env: Any, task: str) -> dict[str, Any]:
    fixture = getattr(env, "fxtr")
    behavior = getattr(env, "behavior")
    open_threshold = 0.90
    closed_threshold_strict = 0.005
    closed_threshold_relaxed = 0.08 if task == "CloseOven" else closed_threshold_strict
    is_open = bool(fixture.is_open(env=env, th=open_threshold))
    is_closed_strict = bool(fixture.is_closed(env=env, th=closed_threshold_strict))
    is_closed_relaxed = bool(fixture.is_closed(env=env, th=closed_threshold_relaxed))
    is_closed = (
        bool(is_closed_relaxed) if behavior == "close" else bool(is_closed_strict)
    )
    door_joint_state = None
    open_fraction_max = None
    closed_fraction_min = None
    closed_pct_min = None
    try:
        joint_state = fixture.get_joint_state(
            env=env,
            joint_names=fixture.door_joint_names,
        )
        if isinstance(joint_state, dict) and joint_state:
            door_joint_state = {
                str(key): float(value) for key, value in joint_state.items()
            }
            open_fraction_max = float(max(door_joint_state.values()))
            closed_fraction_min = float(1.0 - open_fraction_max)
            closed_pct_min = float(closed_fraction_min * 100.0)
    except Exception:
        door_joint_state = None
    if behavior == "open":
        success = is_open
    elif behavior == "close":
        success = is_closed
    else:
        success = False
    try:
        fixture_name = str(
            getattr(fixture, "name", None)
            or getattr(fixture, "_name", None)
            or type(fixture).__name__
        )
    except Exception:
        fixture_name = type(fixture).__name__
    return {
        "task": "ManipulateDoor",
        "behavior": behavior,
        "fixture_id": getattr(env, "fixture_id"),
        "fixture_name": fixture_name,
        "door_joint_state": door_joint_state,
        "open_fraction_max": open_fraction_max,
        "closed_fraction_min": closed_fraction_min,
        "closed_pct_min": closed_pct_min,
        "open_threshold": float(open_threshold),
        "closed_threshold_strict": float(closed_threshold_strict),
        "closed_threshold_relaxed": float(closed_threshold_relaxed),
        "is_open": bool(is_open),
        "is_closed_strict": bool(is_closed_strict),
        "is_closed_relaxed": bool(is_closed_relaxed),
        "is_closed": bool(is_closed),
        "all_and": bool(success),
    }


def _extract_close_fridge_drawer_components(env: Any) -> dict[str, Any]:
    fridge = getattr(env, "fridge")
    threshold = 0.05
    closed = bool(
        fridge.is_closed(
            env,
            compartment="fridge",
            reg_type="drawer",
            drawer_rack_index=-1,
            th=threshold,
        )
    )
    drawer_open_fraction = None
    drawer_closed_fraction = None
    drawer_joint_names = None
    try:
        joints = []
        if hasattr(fridge, "_get_drawer_joints"):
            joints = list(
                fridge._get_drawer_joints(
                    compartment="fridge",
                    drawer_rack_index=-1,
                )
            )
        drawer_joint_names = [str(joint) for joint in joints if str(joint)]
        fractions = []
        for joint_name in drawer_joint_names:
            try:
                joint_id = int(env.sim.model.joint_name2id(joint_name))
                lower, upper = [
                    float(value) for value in env.sim.model.jnt_range[joint_id]
                ]
                address = int(env.sim.model.get_joint_qpos_addr(joint_name))
                qpos = float(env.sim.data.qpos[address])
                if upper == lower:
                    continue
                fraction = (qpos - lower) / (upper - lower)
                fraction = (
                    0.0
                    if fraction < 0.0
                    else 1.0
                    if fraction > 1.0
                    else float(fraction)
                )
                fractions.append(float(fraction))
            except Exception:
                continue
        if fractions:
            drawer_open_fraction = float(max(fractions))
            drawer_closed_fraction = float(1.0 - drawer_open_fraction)
    except Exception:
        drawer_open_fraction = None
    return {
        "task": "CloseFridgeDrawer",
        "compartment": "fridge",
        "reg_type": "drawer",
        "drawer_rack_index": -1,
        "th": float(threshold),
        "is_closed": bool(closed),
        "drawer_joint_names": drawer_joint_names,
        "drawer_open_fraction": drawer_open_fraction,
        "drawer_closed_fraction": drawer_closed_fraction,
        "all_and": bool(closed),
    }


def _robocasa_object_utils() -> Any:
    """Load runtime-only RoboCasa predicates without a package import dependency."""
    from robocasa.utils import object_utils

    return object_utils


def _robocasa_counter_type() -> type[Any]:
    """Load the runtime Counter type without a module-import dependency."""
    from robocasa.models.fixtures.counter import Counter

    return Counter


def _extract_pick_place_counter_to_oven_components(env: Any) -> dict[str, Any]:
    object_utils = _robocasa_object_utils()
    obj_container_contact = object_utils.check_obj_in_receptacle(
        env,
        "obj",
        "oven_tray",
    )
    on_rack = getattr(env, "oven").check_rack_contact(
        env,
        "oven_tray",
        rack_level=getattr(env, "rack_level"),
    )
    gripper_far = object_utils.gripper_obj_far(env, "obj")
    success = bool(on_rack and obj_container_contact and gripper_far)
    return {
        "task": "PickPlaceCounterToOven",
        "rack_level": int(getattr(env, "rack_level")),
        "on_rack": bool(on_rack),
        "obj_in_oven_tray": bool(obj_container_contact),
        "gripper_obj_far": bool(gripper_far),
        "gripper_far_threshold_m": 0.25,
        "all_and": bool(success),
    }


def _extract_pick_place_counter_to_blender_components(env: Any) -> dict[str, Any]:
    object_utils = _robocasa_object_utils()
    obj_in_blender = object_utils.obj_inside_of(
        env,
        "obj",
        getattr(env, "blender"),
        th=0.01,
    )
    gripper_obj_far = object_utils.gripper_obj_far(env)
    success = bool(obj_in_blender and gripper_obj_far)
    return {
        "task": "PickPlaceCounterToBlender",
        "obj_inside_blender": bool(obj_in_blender),
        "obj_inside_blender_th": 0.01,
        "gripper_obj_far": bool(gripper_obj_far),
        "gripper_far_threshold_m": 0.25,
        "all_and": bool(success),
    }


def _extract_make_iced_coffee_components(env: Any) -> dict[str, Any]:
    object_utils = _robocasa_object_utils()
    ice_cube1_in_cup = object_utils.check_obj_in_receptacle(
        env,
        "ice_cube1",
        "cup",
        th=0.5,
    )
    ice_cube2_in_cup = object_utils.check_obj_in_receptacle(
        env,
        "ice_cube2",
        "cup",
        th=0.5,
    )
    ice_in_cup = ice_cube1_in_cup or ice_cube2_in_cup
    gripper_far_from_ice_cube1 = object_utils.gripper_obj_far(
        env,
        "ice_cube1",
        th=0.15,
    )
    gripper_far_from_ice_cube2 = object_utils.gripper_obj_far(
        env,
        "ice_cube2",
        th=0.15,
    )
    gripper_far = gripper_far_from_ice_cube1 and gripper_far_from_ice_cube2
    success = bool(ice_in_cup and gripper_far)
    return {
        "task": "MakeIcedCoffee",
        "ice_cube1_in_cup": bool(ice_cube1_in_cup),
        "ice_cube2_in_cup": bool(ice_cube2_in_cup),
        "ice_in_cup_or": bool(ice_in_cup),
        "receptacle_threshold": 0.5,
        "gripper_far_from_ice_cube1": bool(gripper_far_from_ice_cube1),
        "gripper_far_from_ice_cube2": bool(gripper_far_from_ice_cube2),
        "gripper_far_threshold_m": 0.15,
        "gripper_far_and": bool(gripper_far),
        "all_and": bool(success),
    }


def _normalize_receptacle_result(raw: Any) -> bool:
    import numpy as np

    try:
        return bool(np.all(raw))
    except Exception:
        return bool(raw)


def _extract_bread_and_cheese_components(env: Any) -> dict[str, Any]:
    object_utils = _robocasa_object_utils()
    bread_on_board_raw = object_utils.check_obj_in_receptacle(
        env,
        "obj",
        "container",
    )
    cheese_on_board_raw = object_utils.check_obj_in_receptacle(
        env,
        "obj2",
        "container",
    )
    bread_on_board = _normalize_receptacle_result(bread_on_board_raw)
    cheese_on_board = _normalize_receptacle_result(cheese_on_board_raw)
    food_on_cutting_board = bool(bread_on_board and cheese_on_board)
    gripper_obj_far = bool(object_utils.gripper_obj_far(env))
    success = bool(food_on_cutting_board and gripper_obj_far)
    return {
        "task": "BreadAndCheese",
        "bread_on_cutting_board": bool(bread_on_board),
        "cheese_on_cutting_board": bool(cheese_on_board),
        "food_on_cutting_board_and": bool(food_on_cutting_board),
        "gripper_obj_far": bool(gripper_obj_far),
        "gripper_far_threshold_m": 0.25,
        "all_and": bool(success),
    }


def _extract_bread_setup_slicing_components(env: Any) -> dict[str, Any]:
    object_utils = _robocasa_object_utils()
    bread_checks: dict[str, bool] = {}
    num_bread = getattr(env, "num_bread")
    for index in range(num_bread):
        name = f"obj_{index}"
        bread_checks[name] = bool(
            object_utils.check_obj_in_receptacle(env, name, "receptacle")
        )
    bread_on_board = bool(all(bread_checks.values())) if bread_checks else False
    gripper_far = bool(object_utils.gripper_obj_far(env, "obj_0"))
    success = bool(bread_on_board and gripper_far)
    return {
        "task": "BreadSetupSlicing",
        "num_bread": int(num_bread),
        "bread_on_board_by_obj": bread_checks,
        "bread_on_board_and": bool(bread_on_board),
        "gripper_obj_far_obj_0": bool(gripper_far),
        "gripper_far_threshold_m": 0.25,
        "all_and": bool(success),
    }


def _extract_place_vegetables_evenly_components(env: Any) -> dict[str, Any]:
    import numpy as np

    object_utils = _robocasa_object_utils()
    vegetables = ["veg1", "veg2"]
    in_pan = {
        vegetable: bool(object_utils.check_obj_in_receptacle(env, vegetable, "pan"))
        for vegetable in vegetables
    }
    all_in_pan = bool(all(in_pan.values()))
    min_z_distance = 0.02
    min_xy_distance = 0.06
    z_sep = None
    xy_sep = None
    z_distance = None
    xy_distance = None
    gripper_far = None
    if all_in_pan:
        positions = [
            np.array(env.sim.data.body_xpos[env.obj_body_id[vegetable]])
            for vegetable in vegetables
        ]
        z_values = [float(position[2]) for position in positions]
        xy_values = [np.array(position[:2], dtype=np.float64) for position in positions]
        z_distance = float(abs(z_values[0] - z_values[1]))
        xy_distance = float(np.linalg.norm(xy_values[0] - xy_values[1]))
        z_sep = bool(z_distance < min_z_distance)
        xy_sep = bool(xy_distance > min_xy_distance)
        gripper_far = bool(
            all(
                object_utils.gripper_obj_far(env, vegetable) for vegetable in vegetables
            )
        )
    success = bool(all_in_pan and bool(z_sep) and bool(xy_sep) and bool(gripper_far))
    return {
        "task": "PlaceVegetablesEvenly",
        "in_pan": in_pan,
        "all_in_pan": bool(all_in_pan),
        "min_z_distance": float(min_z_distance),
        "min_xy_distance": float(min_xy_distance),
        "z_distance": z_distance,
        "xy_distance": xy_distance,
        "z_sep": None if z_sep is None else bool(z_sep),
        "xy_sep": None if xy_sep is None else bool(xy_sep),
        "gripper_far": None if gripper_far is None else bool(gripper_far),
        "gripper_far_threshold_m": 0.25,
        "all_and": bool(success),
    }


def _extract_coffee_serve_mug_components(env: Any) -> dict[str, Any]:
    object_utils = _robocasa_object_utils()
    gripper_obj_far = object_utils.gripper_obj_far(env)
    behavior = getattr(env, "behavior")
    if behavior == "counter_to_machine":
        contact_check = getattr(
            env, "coffee_machine"
        ).check_receptacle_placement_for_pouring(
            env,
            "obj",
        )
    elif behavior == "machine_to_counter":
        contact_check = object_utils.check_obj_fixture_contact(
            env,
            "obj",
            getattr(env, "counter"),
        )
    else:
        contact_check = False
    success = bool(contact_check and gripper_obj_far)
    return {
        "task": "PickPlaceCoffee",
        "behavior": behavior,
        "contact_check": bool(contact_check),
        "gripper_obj_far": bool(gripper_obj_far),
        "all_and": bool(success),
    }


def _extract_pack_dessert_components(env: Any) -> dict[str, Any]:
    import numpy as np

    object_utils = _robocasa_object_utils()
    cooked_food_in_container_strict = object_utils.check_obj_in_receptacle(
        env,
        "cooked_food",
        "cooked_food_container",
    )
    dessert_in_container_strict = object_utils.check_obj_in_receptacle(
        env,
        "dessert",
        "cooked_food_container",
    )
    gripper_far_from_dessert = object_utils.gripper_obj_far(env, "dessert")
    dessert_contact_container = None
    dessert_contact_cooked_food = None
    dessert_xy_dist_m = None
    dessert_xy_th_m = None
    dessert_xy_th_relaxed_m = None
    dessert_xy_within_th = None
    dessert_xy_ratio = None
    dessert_xy_pct_of_th = None
    dessert_xy_pct_of_radius = None
    dessert_in_container_loose = None
    dessert_in_container_relaxed = None
    dessert_in_container = bool(dessert_in_container_strict)
    try:
        dessert_obj = env.objects.get("dessert", None)
        cooked_food_obj = env.objects.get("cooked_food", None)
        container_obj = env.objects.get("cooked_food_container", None)
        if dessert_obj is not None and cooked_food_obj is not None:
            dessert_contact_cooked_food = bool(
                env.check_contact(dessert_obj, cooked_food_obj)
            )
        if dessert_obj is not None and container_obj is not None:
            dessert_contact_container = bool(
                env.check_contact(dessert_obj, container_obj)
            )
        if container_obj is not None:
            radius = float(getattr(container_obj, "horizontal_radius", 0.0))
            dessert_xy_th_m = radius * 0.7
            dessert_xy_th_relaxed_m = radius * 0.8
        if dessert_xy_th_m is not None and dessert_xy_th_m > 0.0:
            dessert_pos = np.array(
                env.sim.data.body_xpos[env.obj_body_id["dessert"]],
                dtype=np.float64,
            ).reshape(3)
            container_pos = np.array(
                env.sim.data.body_xpos[env.obj_body_id["cooked_food_container"]],
                dtype=np.float64,
            ).reshape(3)
            dessert_xy_dist_m = float(
                np.linalg.norm(dessert_pos[:2] - container_pos[:2])
            )
            dessert_xy_within_th = bool(dessert_xy_dist_m < float(dessert_xy_th_m))
            dessert_xy_ratio = float(dessert_xy_dist_m / float(dessert_xy_th_m))
            dessert_xy_pct_of_th = float(dessert_xy_ratio * 100.0)
            try:
                if radius > 0.0:
                    dessert_xy_pct_of_radius = float(
                        (dessert_xy_dist_m / radius) * 100.0
                    )
            except Exception:
                dessert_xy_pct_of_radius = None
            if (
                dessert_contact_container is not None
                and dessert_xy_th_relaxed_m is not None
            ):
                dessert_in_container_relaxed = bool(
                    dessert_contact_container
                    and (dessert_xy_dist_m < float(dessert_xy_th_relaxed_m))
                )
                dessert_in_container = bool(
                    dessert_in_container or dessert_in_container_relaxed
                )
        elif dessert_contact_container is not None:
            dessert_in_container_relaxed = bool(dessert_contact_container)
            dessert_in_container = bool(
                dessert_in_container or dessert_in_container_relaxed
            )
        if (
            dessert_xy_within_th is not None
            and cooked_food_in_container_strict is not None
            and dessert_contact_cooked_food is not None
        ):
            dessert_in_container_loose = bool(
                dessert_xy_within_th
                and cooked_food_in_container_strict
                and dessert_contact_cooked_food
            )
            dessert_in_container = bool(
                dessert_in_container or dessert_in_container_loose
            )
    except Exception:
        pass
    success = bool(
        cooked_food_in_container_strict
        and dessert_in_container
        and gripper_far_from_dessert
    )
    return {
        "task": "PackDessert",
        "cooked_food_in_container_strict": bool(cooked_food_in_container_strict),
        "dessert_in_container_strict": bool(dessert_in_container_strict),
        "dessert_contact_container": dessert_contact_container,
        "dessert_contact_cooked_food": dessert_contact_cooked_food,
        "dessert_xy_dist_m": dessert_xy_dist_m,
        "dessert_xy_th_m": dessert_xy_th_m,
        "dessert_xy_th_relaxed_m": dessert_xy_th_relaxed_m,
        "dessert_xy_within_th": dessert_xy_within_th,
        "dessert_xy_pct_of_th": dessert_xy_pct_of_th,
        "dessert_xy_pct_of_radius": dessert_xy_pct_of_radius,
        "dessert_in_container_loose": dessert_in_container_loose,
        "dessert_in_container_relaxed": dessert_in_container_relaxed,
        "dessert_in_container": bool(dessert_in_container),
        "gripper_far_from_dessert": bool(gripper_far_from_dessert),
        "gripper_far_threshold_m": 0.25,
        "all_and": bool(success),
    }


def _extract_open_blender_lid_components(env: Any) -> dict[str, Any]:
    import numpy as np

    object_utils = _robocasa_object_utils()
    lid_on_blender = None
    lid_off_blender = None
    dist_to_closed_m = None
    closed_thresh_m = None
    gripper_lid_far = None
    gripper_lid_far_thresh_m = 0.15
    lid_on_any_counter = False
    lid_on_counters: dict[str, bool] = {}
    lid_body_name = None
    lid_body_id = None
    lid_pos = None
    lid_closed_pos = None
    candidate_lid_bodies = None
    errors: dict[str, str] = {}

    try:
        env.blender.update_state(env)
    except Exception as error:
        errors["blender_update_state_error"] = f"{type(error).__name__}: {error}"

    try:
        state = env.blender.get_state()
        lid_on_blender = bool(state.get("lid_on_blender", True))
        lid_off_blender = bool(not lid_on_blender)
    except Exception as error:
        errors["blender_state_error"] = f"{type(error).__name__}: {error}"

    try:
        closed_thresh_m = float(getattr(env.blender, "_BLENDER_LID_POS_THRESH"))
    except Exception:
        closed_thresh_m = None

    try:
        current = env.blender.get_curr_lid_pos(env)
        closed = env.blender.get_lid_closed_pos(env)
        if current is not None and closed is not None:
            dist_to_closed_m = float(
                np.linalg.norm(np.array(current) - np.array(closed))
            )
    except Exception as error:
        errors["lid_pose_error"] = f"{type(error).__name__}: {error}"

    try:
        if getattr(env.blender, "blender_lid", None) is not None:
            lid_body_name = f"{env.blender.blender_lid.name}_main"
    except Exception:
        lid_body_name = None

    try:
        body_names = list(getattr(env.sim.model, "body_names", []) or [])
        candidate_lid_bodies = sorted(
            [
                str(name)
                for name in body_names
                if isinstance(name, str)
                and "blender" in name.lower()
                and "lid" in name.lower()
            ]
        )
    except Exception as error:
        errors["candidate_lid_bodies_error"] = f"{type(error).__name__}: {error}"

    if lid_body_name is None and candidate_lid_bodies:
        lid_body_name = candidate_lid_bodies[0]

    try:
        if lid_body_name is not None:
            gripper_lid_far = bool(
                object_utils.gripper_fxtr_far(
                    env,
                    lid_body_name,
                    th=gripper_lid_far_thresh_m,
                )
            )
    except Exception as error:
        errors["gripper_lid_far_error"] = f"{type(error).__name__}: {error}"

    try:
        counter_type = _robocasa_counter_type()
        lid_object = getattr(env.blender, "blender_lid", None)
        if lid_object is not None:
            for fixture in env.fixtures.values():
                if not isinstance(fixture, counter_type):
                    continue
                try:
                    key = str(
                        getattr(fixture, "name", None)
                        or getattr(fixture, "_name", None)
                        or type(fixture).__name__
                    )
                except Exception:
                    key = type(fixture).__name__
                try:
                    in_contact = bool(env.check_contact(lid_object, fixture))
                except Exception:
                    in_contact = False
                lid_on_counters[key] = in_contact
                lid_on_any_counter = bool(lid_on_any_counter or in_contact)
    except Exception as error:
        errors["counter_contact_error"] = f"{type(error).__name__}: {error}"

    lid_off_by_dist = None
    if dist_to_closed_m is not None and closed_thresh_m is not None:
        lid_off_by_dist = bool(dist_to_closed_m > closed_thresh_m)

    try:
        if lid_body_name is not None:
            lid_body_id = int(env.sim.model.body_name2id(lid_body_name))
            lid_pos = [float(value) for value in env.sim.data.body_xpos[lid_body_id]]
    except Exception as error:
        errors["lid_body_pose_error"] = f"{type(error).__name__}: {error}"

    try:
        if getattr(env.blender, "blender_lid", None) is not None:
            lid_closed_pos = [
                float(value) for value in env.blender.get_lid_closed_pos(env)
            ]
    except Exception as error:
        errors["lid_closed_pos_error"] = f"{type(error).__name__}: {error}"

    moved_off_blender = bool(lid_off_blender) if lid_off_blender is not None else False
    moved_off_by_dist = bool(lid_off_by_dist) if lid_off_by_dist is not None else False
    placed_on_counter = bool(lid_on_any_counter)
    success_signal = bool(
        placed_on_counter and (moved_off_blender or moved_off_by_dist)
    )
    released_ok = bool(gripper_lid_far) if gripper_lid_far is not None else False
    success = bool(success_signal and released_ok)
    return {
        "task": "OpenBlenderLid",
        "lid_on_blender": lid_on_blender,
        "lid_off_blender": lid_off_blender,
        "dist_to_closed_m": dist_to_closed_m,
        "closed_thresh_m": closed_thresh_m,
        "lid_off_by_dist": lid_off_by_dist,
        "lid_body_name": lid_body_name,
        "lid_body_id": lid_body_id,
        "lid_pos": lid_pos,
        "lid_closed_pos": lid_closed_pos,
        "candidate_lid_bodies": candidate_lid_bodies,
        "gripper_lid_far": gripper_lid_far,
        "gripper_lid_far_threshold_m": float(gripper_lid_far_thresh_m),
        "lid_on_any_counter": bool(lid_on_any_counter),
        "lid_on_counters": lid_on_counters,
        "moved_off_blender": bool(moved_off_blender),
        "moved_off_by_dist": bool(moved_off_by_dist),
        "placed_on_counter": bool(placed_on_counter),
        "success_signal": bool(success_signal),
        "released_ok": bool(released_ok),
        "all_and": bool(success),
        "errors": errors or None,
    }


def extract_robocasa_task_success_evidence(
    env: Any,
    task_name: str,
) -> Mapping[str, Any] | None:
    """Adapt optional replay evidence without escalating fixture read errors."""
    try:
        components = extract_robocasa_task_success_components(env, task_name)
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ):
        return None
    return {"task_success_components": components}
