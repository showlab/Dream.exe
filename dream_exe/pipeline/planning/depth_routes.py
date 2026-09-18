"""Pure benchmark-only routing for per-sample depth providers.

The current bench stores the requested depth preset in each sample's
``pipeline/trajectory.json``.  A shared batch profile therefore cannot name
one concrete provider without silently overriding most samples.  This module
keeps that selection in the bench layer: callers may declare both supported
provider routes, and the stage adapter selects exactly one concrete route
before the simulator-independent runtime sees the configuration.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from ...video2traj.depth.estimator import (
    SUPPORTED_DEPTH_BACKENDS,
    default_depth_estimator_config,
)


BENCH_RESOLVED_DEPTH_BACKEND = "bench_resolved"
BENCH_RESOLVED_DEPTH_PRESET = "bench_resolved"
BENCH_DEPTH_PROVIDERS_FIELD = "providers"

_SUPPORTED_PROVIDER_FAMILIES = frozenset(SUPPORTED_DEPTH_BACKENDS)
_ROUTER_FIELDS = frozenset(
    {
        "backend",
        "preset",
        BENCH_DEPTH_PROVIDERS_FIELD,
    }
)


def benchmark_depth_provider_routes(
    value: Mapping[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return concrete routes declared by one benchmark depth block.

    A normal concrete runtime remains a one-route result for compatibility.
    The benchmark router itself is deliberately strict and has no implicit
    defaults: every selectable provider must be fully represented in config.
    """

    depth = copy.deepcopy(dict(value))
    backend = str(depth.get("backend", "") or "").strip().lower()
    raw_providers = depth.get(BENCH_DEPTH_PROVIDERS_FIELD)
    if backend != BENCH_RESOLVED_DEPTH_BACKEND:
        if raw_providers is not None:
            raise ValueError(
                "depth.providers is valid only when depth.backend='bench_resolved'"
            )
        return ((backend, depth),)

    unknown = sorted(set(depth).difference(_ROUTER_FIELDS))
    if unknown:
        raise ValueError(
            "bench-resolved depth router contains unsupported fields: "
            + ", ".join(unknown)
        )
    preset = str(depth.get("preset", "") or "").strip().lower()
    if preset != BENCH_RESOLVED_DEPTH_PRESET:
        raise ValueError(
            "bench-resolved depth router requires depth.preset='bench_resolved'"
        )
    if not isinstance(raw_providers, Mapping) or not raw_providers:
        raise TypeError(
            "bench-resolved depth router requires a non-empty depth.providers mapping"
        )

    unsupported = sorted(
        str(name)
        for name in raw_providers
        if str(name).strip().lower() not in _SUPPORTED_PROVIDER_FAMILIES
    )
    if unsupported:
        raise ValueError(
            "bench-resolved depth router contains unsupported providers: "
            + ", ".join(unsupported)
        )

    routes: list[tuple[str, dict[str, Any]]] = []
    seen_families: set[str] = set()
    for raw_name in sorted(raw_providers, key=lambda item: str(item)):
        name = str(raw_name).strip().lower()
        if name in seen_families:
            raise ValueError(
                "bench-resolved depth router contains duplicate canonical "
                f"provider {name!r}"
            )
        seen_families.add(name)
        raw_route = raw_providers[raw_name]
        if not isinstance(raw_route, Mapping):
            raise TypeError(f"depth.providers[{name!r}] must be a mapping")
        route = copy.deepcopy(dict(raw_route))
        route_backend = str(route.get("backend", "") or "").strip().lower()
        if route_backend != name:
            raise ValueError(f"depth.providers[{name!r}].backend must equal {name!r}")
        if BENCH_DEPTH_PROVIDERS_FIELD in route:
            raise ValueError("nested depth provider routers are not supported")
        routes.append((name, route))
    return tuple(routes)


def benchmark_depth_provider_family(preset: str) -> str:
    """Resolve one supported model preset to its concrete provider family."""

    config = default_depth_estimator_config(preset)
    family = str(config.get("model_name", "") or "").strip().lower()
    if family not in _SUPPORTED_PROVIDER_FAMILIES:
        raise ValueError(
            f"depth preset {preset!r} resolves to unsupported family {family!r}"
        )
    return family


def select_benchmark_depth_provider(
    value: Mapping[str, Any],
    *,
    requested_preset: str,
) -> dict[str, Any]:
    """Select one concrete provider route for a resolved sample preset."""

    depth = copy.deepcopy(dict(value))
    backend = str(depth.get("backend", "") or "").strip().lower()
    routes = benchmark_depth_provider_routes(depth)
    if backend != BENCH_RESOLVED_DEPTH_BACKEND:
        return routes[0][1]

    family = benchmark_depth_provider_family(requested_preset)
    by_family = dict(routes)
    if family not in by_family:
        raise ValueError(
            "bench-resolved depth router has no provider for "
            f"preset {requested_preset!r} (family {family!r})"
        )
    return copy.deepcopy(by_family[family])


__all__ = [
    "BENCH_DEPTH_PROVIDERS_FIELD",
    "BENCH_RESOLVED_DEPTH_BACKEND",
    "BENCH_RESOLVED_DEPTH_PRESET",
    "benchmark_depth_provider_family",
    "benchmark_depth_provider_routes",
    "select_benchmark_depth_provider",
]
