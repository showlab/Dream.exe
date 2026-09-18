"""Lazy simulator backend loading and environment construction.

This module preserves the current ``pipeline.core.env.create_env`` behavior
without importing RoboSuite when :mod:`dream_exe.sim` is imported.  Concrete
backend loading happens only when :func:`create_env` is called and can be
injected for tests.  Scene restoration and artifact persistence deliberately
remain outside this boundary.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, MutableSequence
from functools import partial
import importlib
import os
from pathlib import Path
import sys
from typing import Any


_BACKEND_NOT_LOADED = object()
_robosuite_backend: Any = _BACKEND_NOT_LOADED


def _query_mujoco_egl_devices() -> Any:
    """Use the same device enumeration backend as RoboSuite's EGL context."""

    from mujoco.egl import egl_ext as egl

    return egl.eglQueryDevicesEXT()


def preflight_mujoco_egl_runtime(
    *,
    environ: MutableMapping[str, str] | None = None,
    device_query: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Fail before simulator work when the selected EGL device is unusable.

    RoboSuite discovers the EGL device only while constructing an offscreen
    context.  A long benchmark can otherwise materialize inputs and load
    models before learning that its node exposes no EGL devices.  This pure
    environment/device check mirrors that selection without creating a
    simulator environment.
    """

    env = os.environ if environ is None else environ
    backend = env.get("MUJOCO_GL", "").strip().lower()
    if backend != "egl":
        return {
            "status": "not_requested",
            "backend": backend,
        }

    selected_raw = env.get("MUJOCO_EGL_DEVICE_ID", "0").strip() or "0"
    try:
        selected = int(selected_raw)
    except ValueError as error:
        raise RuntimeError(
            "MuJoCo EGL preflight requires MUJOCO_EGL_DEVICE_ID to be one "
            f"integer after CUDA visibility remapping, got {selected_raw!r}."
        ) from error
    if selected < 0:
        raise RuntimeError(
            "MuJoCo EGL preflight requires a non-negative logical "
            f"MUJOCO_EGL_DEVICE_ID, got {selected}."
        )

    query = _query_mujoco_egl_devices if device_query is None else device_query
    try:
        devices = query()
        device_count = len(devices)
    except Exception as error:
        raise RuntimeError(
            "MuJoCo EGL preflight could not enumerate EGL devices on this "
            "node. Select an EGL-capable node before running init or exec."
        ) from error
    if device_count == 0:
        raise RuntimeError(
            "MuJoCo EGL preflight found no EGL devices on this node while "
            "MUJOCO_GL=egl. Select an EGL-capable node before running init "
            "or exec."
        )
    if selected >= device_count:
        raise RuntimeError(
            "MuJoCo EGL preflight found "
            f"{device_count} EGL device(s), but MUJOCO_EGL_DEVICE_ID="
            f"{selected}. Select an EGL-capable node and a logical device in "
            f"the range 0..{device_count - 1} before running init or exec."
        )
    return {
        "status": "verified",
        "backend": "egl",
        "device_count": device_count,
        "logical_device_id": selected,
    }


def prepare_mujoco_egl_device_id(
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Expose the physical CUDA id expected during RoboSuite import.

    This is the import-time half of the current EGL compatibility workaround.
    It intentionally preserves ``-1``, accepts the existing physical id when
    it is visible, and prints only when replacing a non-empty incompatible id.
    """

    env = os.environ if environ is None else environ
    if env.get("MUJOCO_GL", "").strip().lower() != "egl":
        return

    visible_raw = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_raw:
        return

    visible = [item.strip() for item in visible_raw.split(",") if item.strip()]
    if not visible:
        return

    egl_raw = env.get("MUJOCO_EGL_DEVICE_ID", "").strip()
    if egl_raw == "-1":
        return
    if egl_raw and egl_raw in visible:
        return

    first_visible = visible[0]
    env["MUJOCO_EGL_DEVICE_ID"] = first_visible
    if egl_raw:
        print(
            f"[env] Reset MUJOCO_EGL_DEVICE_ID from {egl_raw} to {first_visible} "
            f"to satisfy robosuite CUDA_VISIBLE_DEVICES={visible_raw}"
        )


def finalize_mujoco_egl_device_id(
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Keep the physical EGL id selected before RoboSuite import.

    The RoboSuite renderer currently indexes the unmasked list returned by
    ``eglQueryDevicesEXT()``.  Therefore replacing a physical id such as
    ``5`` with logical id ``0`` would silently send every masked worker to
    EGL device zero.  Import-time mutations are repaired only when the value
    is no longer one of the visible physical ids.
    """

    env = os.environ if environ is None else environ
    if env.get("MUJOCO_GL", "").strip().lower() != "egl":
        return

    visible_raw = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_raw:
        return

    visible = [item.strip() for item in visible_raw.split(",") if item.strip()]
    egl_raw = env.get("MUJOCO_EGL_DEVICE_ID", "").strip()
    if not visible or egl_raw == "-1":
        return
    if egl_raw not in visible:
        env["MUJOCO_EGL_DEVICE_ID"] = visible[0]


def load_robosuite_backend(
    *,
    importer: Callable[[str], Any] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> Any:
    """Load and cache RoboSuite lazily with the current EGL transition.

    Finalization intentionally occurs only after a successful import, matching
    the observable ordering of the current module-level import.  The real
    default load is cached so later environment creation does not touch EGL
    again.  Explicit importer or environment injection remains uncached.
    """

    global _robosuite_backend

    use_default_cache = importer is None and environ is None
    if use_default_cache and _robosuite_backend is not _BACKEND_NOT_LOADED:
        return _robosuite_backend

    module_importer = importlib.import_module if importer is None else importer
    prepare_mujoco_egl_device_id(environ=environ)
    suite = module_importer("robosuite")
    finalize_mujoco_egl_device_id(environ=environ)
    if use_default_cache:
        _robosuite_backend = suite
    return suite


def _validated_robocasa_source_root(source_root: str | Path) -> Path:
    """Validate one caller-owned RoboCasa checkout root.

    The root is deliberately not inferred from the Dream.exe checkout.  An
    explicit source checkout must expose the importable ``robocasa`` package
    directly beneath it, as the reviewed ``external/RoboCasa`` convention
    does.
    """

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboCasa source root is not a directory: {root}")
    package_init = root / "robocasa" / "__init__.py"
    if not package_init.is_file():
        raise ValueError(
            f"RoboCasa source root must contain robocasa/__init__.py: {root}"
        )
    return root


def _require_robocasa_origin(module: Any, *, source_root: Path) -> None:
    """Fail when an explicit checkout resolves to a different installation."""

    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(
            "Explicit RoboCasa source root produced a module without an "
            f"inspectable origin: {source_root}"
        )
    origin = Path(module_file).expanduser().resolve()
    try:
        origin.relative_to(source_root)
    except ValueError as error:
        raise RuntimeError(
            "Imported RoboCasa escaped the explicit source root: "
            f"origin={origin}, source_root={source_root}. Remove a conflicting "
            "cached/installed robocasa module or use the installed dependency "
            "without source_root."
        ) from error


def ensure_robocasa_imported(
    *,
    source_root: str | Path | None = None,
    importer: Callable[[str], Any] | None = None,
    module_search_path: MutableSequence[str] | None = None,
) -> None:
    """Import RoboCasa so that its environments register with RoboSuite.

    By default RoboCasa must already be installed in the active environment;
    Dream.exe never searches its own parents or a neighboring repository.  A
    caller may instead provide an explicit checkout root such as
    ``external/RoboCasa``.  Only that validated root is prepended, and real
    imports are checked to originate inside it.  Injection hooks keep this
    behavior testable without importing either simulator.
    """

    search_path = sys.path if module_search_path is None else module_search_path
    root = None if source_root is None else _validated_robocasa_source_root(source_root)
    inserted_root: str | None = None
    if root is not None:
        root_text = root.as_posix()
        if root_text not in search_path:
            search_path.insert(0, root_text)
            inserted_root = root_text

    use_default_importer = importer is None
    module_importer = importlib.import_module if importer is None else importer
    try:
        module = module_importer("robocasa")
        if root is not None and use_default_importer:
            _require_robocasa_origin(module, source_root=root)
    except BaseException:
        if inserted_root is not None:
            search_path.remove(inserted_root)
        raise


def preflight_robocasa_runtime(
    *,
    backend_loader: Callable[[], Any] | None = None,
    robocasa_loader: Callable[[], Any] | None = None,
    robocasa_source_root: str | Path | None = None,
    egl_device_query: Callable[[], Any] | None = None,
) -> None:
    """Validate the concrete RoboCasa runtime without creating an environment.

    The import order matches :func:`create_env`: RoboSuite is loaded before
    RoboCasa performs its environment registration.  Installed-package and
    explicit-checkout behavior remains owned by :func:`ensure_robocasa_imported`.
    """

    if robocasa_loader is not None and robocasa_source_root is not None:
        raise ValueError(
            "Provide either robocasa_loader or robocasa_source_root, not both."
        )
    load_backend = load_robosuite_backend if backend_loader is None else backend_loader
    try:
        load_backend()
    except Exception as error:
        raise RuntimeError(
            "RoboCasa runtime preflight failed before a simulator environment "
            "was created: could not import RoboSuite. Install the simulator "
            "dependencies in the active environment (for this source checkout: "
            "`python -m pip install '.[sim]'`)."
        ) from error

    if backend_loader is None or egl_device_query is not None:
        preflight_mujoco_egl_runtime(device_query=egl_device_query)

    register_robocasa = robocasa_loader
    if register_robocasa is None:
        register_robocasa = partial(
            ensure_robocasa_imported,
            source_root=robocasa_source_root,
        )
    try:
        register_robocasa()
    except Exception as error:
        raise RuntimeError(
            "RoboCasa runtime preflight failed before a simulator environment "
            "was created: could not import RoboCasa. Install the simulator and "
            "RoboCasa import dependencies (for this source checkout: "
            "`python -m pip install '.[sim,robocasa-runtime]'`), then install a "
            "reviewed RoboCasa package in the active environment or pass an "
            "explicit RoboCasa checkout root (for example `external/RoboCasa`) "
            "through robocasa_source_root. Dream.exe does not search parent "
            "repositories or infer RoboCasa source, revision, or assets."
        ) from error


def create_env(
    env_name: str,
    robots: str,
    camera_name: str,
    frame_size: tuple[int, int],
    num_steps: int,
    seed: int | None = None,
    *,
    render: bool = False,
    offscreen: bool = True,
    use_camera_obs: bool = False,
    camera_depths: bool = False,
    camera_segmentations: Any | None = None,
    controller_cfg: dict[str, Any] | None = None,
    policy_hz: int | None = None,
    backend: str = "robosuite",
    camera_names: Any | None = None,
    camera_widths: Any | None = None,
    camera_heights: Any | None = None,
    extra_make_kwargs: dict[str, Any] | None = None,
    backend_loader: Callable[[], Any] | None = None,
    suite_make: Callable[..., Any] | None = None,
    robocasa_loader: Callable[[], Any] | None = None,
    robocasa_source_root: str | Path | None = None,
) -> Any:
    """Create a RoboSuite-compatible environment with current make fallbacks.

    ``suite_make`` bypasses backend loading entirely.  Otherwise,
    ``backend_loader`` (or the lazy default loader) must return an object with a
    ``make`` callable.  These hooks are runtime seams, not alternate behavior:
    all environment kwargs, conversions, precedence, and TypeError retries
    match the current implementation.
    """

    normalized_backend = str(backend or "robosuite").strip().lower()
    if (
        normalized_backend == "robocasa"
        and robocasa_loader is not None
        and robocasa_source_root is not None
    ):
        raise ValueError(
            "Provide either robocasa_loader or robocasa_source_root, not both."
        )

    suite_backend = None
    if suite_make is None:
        loader = load_robosuite_backend if backend_loader is None else backend_loader
        suite_backend = loader()

    if normalized_backend == "robocasa":
        registration_loader = robocasa_loader
        if registration_loader is None:
            registration_loader = partial(
                ensure_robocasa_imported,
                source_root=robocasa_source_root,
            )
        registration_loader()

    names = camera_names if camera_names is not None else [camera_name]
    widths = camera_widths if camera_widths is not None else frame_size[0]
    heights = camera_heights if camera_heights is not None else frame_size[1]

    kwargs: dict[str, Any] = dict(extra_make_kwargs or {})
    kwargs.update(
        {
            "env_name": env_name,
            "robots": robots,
            "has_renderer": render,
            "has_offscreen_renderer": offscreen,
            "use_camera_obs": use_camera_obs,
            "render_camera": camera_name,
            "camera_names": names,
            "camera_depths": camera_depths,
            "camera_heights": heights,
            "camera_widths": widths,
            "horizon": num_steps,
        }
    )
    if camera_segmentations is not None:
        kwargs["camera_segmentations"] = camera_segmentations

    if seed is not None:
        kwargs["seed"] = int(seed)
    if policy_hz is not None:
        kwargs["control_freq"] = int(policy_hz)
    if controller_cfg is not None:
        kwargs["controller_configs"] = controller_cfg
    elif "controller_configs" in kwargs and kwargs["controller_configs"] is None:
        kwargs.pop("controller_configs", None)

    while True:
        try:
            make = suite_make if suite_make is not None else suite_backend.make
            return make(**kwargs)
        except TypeError as error:
            message = str(error)
            removed = False
            if "seed" in message and "seed" in kwargs:
                kwargs.pop("seed", None)
                removed = True
            if "camera_segmentations" in message and "camera_segmentations" in kwargs:
                kwargs.pop("camera_segmentations", None)
                removed = True
            if removed:
                continue
            raise
