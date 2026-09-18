"""Simulator contact and support-surface queries used by execution gates."""

from __future__ import annotations

from typing import Any


def _all_robot_contact_pairs(
    env: Any,
    robot_prefix: tuple[str, ...] = ("robot0_",),
) -> list[tuple[str, str, float]]:
    names = list(env.sim.model.geom_names)
    name_count = len(names)

    def safe_name(geometry_id: int) -> str:
        geometry_id = int(geometry_id)
        if 0 <= geometry_id < name_count:
            return names[geometry_id] or ""
        return ""

    def is_robot(geometry_name: str) -> bool:
        normalized = geometry_name.lower()
        return any(normalized.startswith(prefix) for prefix in robot_prefix)

    pairs: list[tuple[str, str, float]] = []
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        first_name = safe_name(contact.geom1)
        second_name = safe_name(contact.geom2)
        if not (is_robot(first_name) or is_robot(second_name)):
            continue
        pairs.append(
            (first_name, second_name, float(getattr(contact, "dist", 0.0)))
        )
    pairs.sort(key=lambda item: item[2])
    return pairs


def robot_contacts_any(
    env: Any,
    robot_prefix: tuple[str, ...] = ("robot0_",),
    topk: int = 5,
) -> list[tuple[str, str, float]]:
    """Return the closest current contacts involving a robot geometry."""

    return _all_robot_contact_pairs(env, robot_prefix=robot_prefix)[:topk]


def robot_non_support_contacts(
    env: Any,
    robot_prefix: tuple[str, ...] = ("robot0_",),
    exclude_keywords: tuple[str, ...] = (
        "table",
        "counter",
        "workspace",
        "bin",
        "floor",
        "wall",
    ),
    topk: int = 5,
) -> list[tuple[str, str, float]]:
    """Return robot contacts after excluding robot/self and support contacts."""

    def is_robot(geometry_name: str) -> bool:
        normalized = geometry_name.lower()
        return any(normalized.startswith(prefix) for prefix in robot_prefix)

    def is_support(geometry_name: str) -> bool:
        normalized = geometry_name.lower()
        return any(keyword in normalized for keyword in exclude_keywords)

    filtered: list[tuple[str, str, float]] = []
    for first_name, second_name, distance in _all_robot_contact_pairs(
        env,
        robot_prefix=robot_prefix,
    ):
        if is_robot(first_name) and is_robot(second_name):
            continue
        other_name = second_name if is_robot(first_name) else first_name
        if is_support(other_name):
            continue
        filtered.append((first_name, second_name, distance))
    return filtered[:topk]


def find_support_top_z(
    env: Any,
    prefer: tuple[str, ...] = ("table", "counter", "workspace", "bin"),
) -> float | None:
    """Return the highest preferred support-geometry top surface."""

    candidates: list[float] = []
    for geometry_id, name in enumerate(env.sim.model.geom_names):
        if not name:
            continue
        normalized = name.lower()
        if "floor" in normalized or not any(
            keyword in normalized for keyword in prefer
        ):
            continue
        position = env.sim.model.geom_pos[geometry_id]
        size = env.sim.model.geom_size[geometry_id]
        candidates.append(float(position[2] + size[2]))
    return max(candidates) if candidates else None


__all__ = [
    "find_support_top_z",
    "robot_contacts_any",
    "robot_non_support_contacts",
]
