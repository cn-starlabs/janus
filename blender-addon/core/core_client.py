"""ctypes wrapper around libjanus_ffi."""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import bpy

ADDON_PACKAGE = Path(__file__).resolve().parents[1].name


def _addon_module_candidates() -> list[str]:
    base_name = ADDON_PACKAGE
    candidates = [base_name, base_name.replace("-", "_"), base_name.replace("-", "")]
    if base_name != "blender_addon":
        candidates.append("blender_addon")
    if base_name != "janus_addon":
        candidates.append("janus_addon")
    return [name for name in candidates if name]


class JanusFieldView(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)), ("len", ctypes.c_size_t)]


class JanusGridInfo(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_size_t),
        ("ny", ctypes.c_size_t),
        ("dx", ctypes.c_double),
        ("dy", ctypes.c_double),
        ("origin_x", ctypes.c_double),
        ("origin_y", ctypes.c_double),
        ("ncells", ctypes.c_size_t),
    ]


def _platform_lib_name() -> str:
    if sys.platform == "win32":
        return "janus_ffi.dll"
    if sys.platform == "darwin":
        return "libjanus_ffi.dylib"
    return "libjanus_ffi.so"


def _default_search_paths() -> list[Path]:
    addon_dir = Path(__file__).resolve().parents[1]
    return [
        addon_dir / "bin" / _platform_lib_name(),
        addon_dir.parent / "target" / "release" / _platform_lib_name(),
        addon_dir.parent / "target" / "debug" / _platform_lib_name(),
    ]


def resolve_library_path() -> str | None:
    for module_name in _addon_module_candidates():
        addon = bpy.context.preferences.addons.get(module_name)
        if addon and addon.preferences.library_path:
            p = Path(bpy.path.abspath(addon.preferences.library_path))
            if p.is_file():
                return str(p)
    for candidate in _default_search_paths():
        if candidate.is_file():
            return str(candidate)
    return None


class JanusLibrary:
    def __init__(self, path: str):
        self._lib = ctypes.CDLL(path)
        self._setup_signatures()

    def _setup_signatures(self):
        lib = self._lib
        lib.janus_ffi_api_version.restype = ctypes.c_uint32
        lib.janus_last_error.restype = ctypes.c_char_p
        lib.janus_solver_create.argtypes = [ctypes.c_char_p]
        lib.janus_solver_create.restype = ctypes.c_void_p
        lib.janus_solver_destroy.argtypes = [ctypes.c_void_p]
        lib.janus_solver_step.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.janus_solver_step.restype = ctypes.c_int
        lib.janus_solver_cfl_dt.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.janus_solver_cfl_dt.restype = ctypes.c_double
        lib.janus_solver_grid_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(JanusGridInfo)]
        lib.janus_solver_grid_info.restype = ctypes.c_int
        lib.janus_solver_set_scheme.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.janus_solver_set_scheme.restype = ctypes.c_int
        lib.janus_solver_field_view.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(JanusFieldView)]
        lib.janus_solver_field_view.restype = ctypes.c_int
        lib.janus_solver_write_jvtk.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_uint64]
        lib.janus_solver_write_jvtk.restype = ctypes.c_int
        lib.janus_default_case_json.restype = ctypes.c_char_p

    @property
    def api_version(self) -> int:
        return int(self._lib.janus_ffi_api_version())

    def last_error(self) -> str:
        msg = self._lib.janus_last_error()
        return msg.decode("utf-8") if msg else ""

    def default_case_json(self) -> str:
        raw = self._lib.janus_default_case_json()
        return raw.decode("utf-8") if raw else "{}"

    def create_solver(self, config: dict | str) -> int:
        payload = json.dumps(config).encode("utf-8") if isinstance(config, dict) else config.encode("utf-8")
        handle = self._lib.janus_solver_create(payload)
        if not handle:
            raise RuntimeError(self.last_error() or "janus_solver_create failed")
        return handle

    def destroy_solver(self, handle: int):
        if handle:
            self._lib.janus_solver_destroy(handle)

    def step(self, handle: int, dt: float):
        if self._lib.janus_solver_step(handle, dt) != 0:
            raise RuntimeError(self.last_error() or "janus_solver_step failed")

    def cfl_dt(self, handle: int, cfl: float) -> float:
        return float(self._lib.janus_solver_cfl_dt(handle, cfl))

    def set_scheme(self, handle: int, scheme: int):
        if self._lib.janus_solver_set_scheme(handle, scheme) != 0:
            raise RuntimeError(self.last_error() or "janus_solver_set_scheme failed")

    def grid_info(self, handle: int) -> JanusGridInfo:
        out = JanusGridInfo()
        if self._lib.janus_solver_grid_info(handle, ctypes.byref(out)) != 0:
            raise RuntimeError(self.last_error() or "janus_solver_grid_info failed")
        return out

    def field_view(self, handle: int, name: str) -> tuple[int, int]:
        view = JanusFieldView()
        if self._lib.janus_solver_field_view(handle, name.encode("utf-8"), ctypes.byref(view)) != 0:
            raise RuntimeError(self.last_error() or f"field '{name}' unavailable")
        return ctypes.addressof(view.data.contents), view.len

    def write_jvtk(self, handle: int, path: str, time: float, step: int):
        if self._lib.janus_solver_write_jvtk(handle, path.encode("utf-8"), time, step) != 0:
            raise RuntimeError(self.last_error() or "janus_solver_write_jvtk failed")


_LIB: JanusLibrary | None = None


def get_library() -> JanusLibrary:
    global _LIB
    if _LIB is not None:
        return _LIB
    path = resolve_library_path()
    if not path:
        raise FileNotFoundError(
            "Janus FFI library not found. Set path in Edit > Preferences > Add-ons > Janus CFD."
        )
    _LIB = JanusLibrary(path)
    return _LIB
