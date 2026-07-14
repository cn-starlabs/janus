"""Grid setup operators."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
from bpy.types import Operator

from ..core.jvtk_reader import JvtkHeader
from ..core.mesh_builder import ensure_field_mesh


def _scene_grid_defaults(scene: bpy.types.Scene) -> dict:
    props = scene.janus
    grid = {"nx": 20, "ny": 20, "dx": 1.0e-3 / 20.0, "dy": 1.0e-3 / 20.0, "origin": [0.0, 0.0]}

    if props.mesh_object_name:
        obj = bpy.data.objects.get(props.mesh_object_name)
        if obj is not None:
            bounds = obj.bound_box
            if bounds:
                width = max(1e-6, abs(bounds[-1][0] - bounds[0][0]))
                height = max(1e-6, abs(bounds[-1][1] - bounds[0][1]))
                grid["dx"] = width / max(1, grid["nx"])
                grid["dy"] = height / max(1, grid["ny"])
                grid["origin"] = [bounds[0][0], bounds[0][1]]

    return grid


def _scene_domain_bounds(scene: bpy.types.Scene) -> tuple[float, float, float, float, float, float]:
    props = scene.janus
    ref_obj = bpy.data.objects.get(props.mesh_object_name)
    if ref_obj is not None and getattr(ref_obj, "bound_box", None):
        bounds = ref_obj.bound_box
        if bounds:
            return (
                min(v[0] for v in bounds),
                max(v[0] for v in bounds),
                min(v[1] for v in bounds),
                max(v[1] for v in bounds),
                min(v[2] for v in bounds),
                max(v[2] for v in bounds),
            )
    return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)


def _build_case_payload(scene: bpy.types.Scene) -> dict:
    props = scene.janus
    default_case = {
        "grid": _scene_grid_defaults(scene),
        "bcs": {
            "west": "Periodic",
            "east": "Periodic",
            "south": {"DiffuseWall": {"temperature": 300.0, "wall_velocity": [0.0, 0.0]}},
            "north": {"DiffuseWall": {"temperature": 300.0, "wall_velocity": [0.0, 0.0]}},
        },
        "gas": {"r_gas": 208.13, "molar_mass": 0.039948, "vhs_omega": 0.81, "mu_ref": 2.117e-5, "t_ref": 273.15, "prandtl": 2.0 / 3.0},
    }

    if props.manifest_path:
        try:
            from ..core.core_client import get_library

            lib = get_library()
            case = json.loads(lib.default_case_json())
            default_case = case
        except (FileNotFoundError, RuntimeError):
            pass

    # Extract bcs from either flat format {"bcs": {...}} or FFI nested format
    # {"config": {"bcs": {...}}}.  `lib.default_case_json()` returns the nested
    # form; the plain Python default above uses the flat form.  Both must be
    # handled so we never send an empty or incomplete bcs object to the solver
    # (serde will reject it with "missing field `west`").
    bcs: dict = (
        default_case.get("bcs")
        or default_case.get("config", {}).get("bcs")
        or {}
    )

    # Guarantee every required edge key is present. serde's BoundaryAssignment
    # has no #[serde(default)] on its fields so every key is mandatory.
    _EDGE_DEFAULT = {
        "west": "Periodic",
        "east": "Periodic",
        "south": "Periodic",
        "north": "Periodic",
    }
    for edge in ("west", "east", "south", "north"):
        if edge not in bcs:
            bcs[edge] = _EDGE_DEFAULT[edge]

    # Override with any boundary objects the user tagged in the scene.
    for obj in bpy.data.objects:
        if "janus_boundary_role" not in obj:
            continue
        role = str(obj["janus_boundary_role"])
        if role in {"west", "east", "south", "north"}:
            temperature = float(obj.get("janus_boundary_temperature", props.bc_temperature))
            velocity = list(obj.get("janus_boundary_velocity", [props.bc_wall_velocity_x, props.bc_wall_velocity_y]))
            bcs[role] = {"DiffuseWall": {"temperature": temperature, "wall_velocity": velocity}}

    # Extract grid and gas — handle both flat and nested FFI formats.
    cfg = default_case.get("config", {})
    grid = cfg.get("grid") or default_case.get("grid") or _scene_grid_defaults(scene)
    gas = cfg.get("gas") or default_case.get("gas") or {
        "r_gas": 208.13, "molar_mass": 0.039948, "vhs_omega": 0.81,
        "mu_ref": 2.117e-5, "t_ref": 273.15, "prandtl": 2.0 / 3.0,
    }

    payload = {
        "config": {
            "grid": grid,
            "bcs": bcs,
            "gas": gas,
        },
        "initial": {
            "rho": 1.0,
            "temperature": 300.0,
            "velocity": [0.0, 0.0],
        },
        "velocity_grid": {
            "v_max": 1800.0,
            "n_per_dim": 25,
        },
        "time_scheme": {
            "scheme": ("euler", "rk2", "rk4")[int(props.sim_scheme)]
        },
        "scene": {
            "mesh_object": props.mesh_object_name,
            "boundary_tags": [
                {
                    "object": obj.name,
                    "role": obj.get("janus_boundary_role", ""),
                    "kind": obj.get("janus_boundary_kind", ""),
                    "temperature": obj.get("janus_boundary_temperature", props.bc_temperature),
                    "velocity": obj.get("janus_boundary_velocity", [props.bc_wall_velocity_x, props.bc_wall_velocity_y]),
                    "domain_bounds": list(_scene_domain_bounds(scene)),
                }
                for obj in bpy.data.objects
                if "janus_boundary_role" in obj
            ],
        },
    }
    return payload


def _addon_pref_entry(context) -> tuple[object, str] | tuple[None, None]:
    addon_dir = Path(__file__).resolve().parents[1]
    module_candidates = [
        addon_dir.name,
        addon_dir.name.replace("-", "_"),
        addon_dir.name.replace("-", ""),
        "blender_addon",
        "janus_addon",
    ]
    for module_name in module_candidates:
        addon = context.preferences.addons.get(module_name)
        if addon is not None:
            return addon, module_name
    return None, None


def _get_addon_preferences(context):
    addon, _ = _addon_pref_entry(context)
    if addon is None:
        return None
    return addon.preferences


class JANUS_OT_build_ffi(Operator):
    bl_idname = "janus.build_ffi"
    bl_label = "Build Janus FFI"
    bl_description = "Build the Janus Rust FFI library and copy it into the add-on bundle"
    bl_options = {"REGISTER"}

    def execute(self, context):
        addon_dir = Path(__file__).resolve().parents[1]
        target_dir = addon_dir / "bin"
        target_dir.mkdir(parents=True, exist_ok=True)

        prefs = _get_addon_preferences(context)
        repo_root_value = ""
        if prefs is not None and getattr(prefs, "repository_path", ""):
            repo_root_value = getattr(prefs, "repository_path")
        repo_root = Path(bpy.path.abspath(repo_root_value)) if repo_root_value else addon_dir.parent
        if not repo_root.exists():
            self.report({"ERROR"}, f"Repository path does not exist: {repo_root}")
            return {"CANCELLED"}
        if not (repo_root / "Cargo.toml").is_file():
            self.report({"ERROR"}, f"Repository path does not look like the Janus root: {repo_root}")
            return {"CANCELLED"}

        if sys.platform == "win32":
            lib_name = "janus_ffi.dll"
        elif sys.platform == "darwin":
            lib_name = "libjanus_ffi.dylib"
        else:
            lib_name = "libjanus_ffi.so"

        try:
            subprocess.run(
                ["cargo", "build", "-p", "janus-ffi", "--release"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            self.report({"ERROR"}, "cargo was not found on PATH")
            return {"CANCELLED"}
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or exc.stdout or "cargo build failed").strip()
            self.report({"ERROR"}, err[:400])
            return {"CANCELLED"}

        src = repo_root / "target" / "release" / lib_name
        if not src.is_file():
            self.report({"ERROR"}, f"Built library not found at {src}")
            return {"CANCELLED"}

        dst = target_dir / lib_name
        shutil.copy2(src, dst)

        prefs = _get_addon_preferences(context)
        if prefs is not None:
            prefs.library_path = str(dst)

        self.report({"INFO"}, f"Built and installed {lib_name} -> {dst}")
        return {"FINISHED"}


class JANUS_OT_build_grid(Operator):
    bl_idname = "janus.build_grid"
    bl_label = "Build Demo Grid"
    bl_description = "Create a placeholder domain mesh from the default FFI case JSON"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.janus
        try:
            from ..core.core_client import get_library

            lib = get_library()
            case = json.loads(lib.default_case_json())
            payload = _build_case_payload(context.scene)
            grid = payload["config"]["grid"]
            header = JvtkHeader(
                dims=(grid["nx"], grid["ny"], 1),
                spacing=(grid["dx"], grid["dy"], 1.0),
                origin=(grid["origin"][0], grid["origin"][1], 0.0),
                time=0.0,
                step=0,
                kn_range=(0.0, 0.0),
                cell_fields=(),
            )
            ensure_field_mesh(context.scene, props.mesh_object_name, header)
            out_path = Path(bpy.path.abspath(props.sim_output_dir)) / "case.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.report({"INFO"}, f"Built {grid['nx']}x{grid['ny']} domain mesh and wrote {out_path}")
            return {"FINISHED"}
        except (FileNotFoundError, RuntimeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
