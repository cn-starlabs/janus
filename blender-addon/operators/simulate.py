"""Live simulation operators (FFI solver + viewport streaming)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import bpy
from bpy.types import Operator

from ..core.jvtk_reader import JvtkHeader
from ..core.mesh_builder import (
    apply_live_field_from_solver,
    ensure_field_mesh,
    ensure_preview_material,
)


class _SimRuntime:
    """Module-level state for the active live simulation."""

    handle: int | None = None
    lib = None
    step_count: int = 0
    sim_time: float = 0.0
    dt: float = 0.0
    frame_index: int = 0
    manifest_frames: list[dict] = []
    write_every: int = 50
    output_dir: str = ""


def _stop_runtime():
    rt = _SimRuntime
    if rt.handle and rt.lib:
        rt.lib.destroy_solver(rt.handle)
    rt.handle = None
    rt.lib = None
    rt.step_count = 0
    rt.sim_time = 0.0
    rt.frame_index = 0
    rt.manifest_frames = []


def _write_manifest(output_dir: str, grid_info, write_every: int):
    manifest = {
        "case_name": "live",
        "write_every": write_every,
        "dims": [grid_info.nx, grid_info.ny, 1],
        "spacing": [grid_info.dx, grid_info.dy, 1.0],
        "origin": [grid_info.origin_x, grid_info.origin_y, 0.0],
        "frames": _SimRuntime.manifest_frames,
    }
    path = Path(output_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(path)


def _sim_timer():
    """Called on Blender's main thread each tick while sim is running."""
    rt = _SimRuntime
    if not rt.handle or not rt.lib:
        return None

    props = bpy.context.scene.janus
    try:
        for _ in range(props.sim_steps_per_tick):
            rt.lib.step(rt.handle, rt.dt)
            rt.step_count += 1
            rt.sim_time += rt.dt

            if rt.step_count % rt.write_every == 0:
                fname = f"live_{rt.frame_index:04d}.jvtk"
                fpath = os.path.join(rt.output_dir, fname)
                rt.lib.write_jvtk(rt.handle, fpath, rt.sim_time, rt.step_count)
                rt.manifest_frames.append(
                    {
                        "index": rt.frame_index,
                        "step": rt.step_count,
                        "time": rt.sim_time,
                        "file": fname,
                    }
                )
                rt.frame_index += 1

        scene = bpy.context.scene
        grid = rt.lib.grid_info(rt.handle)
        header = JvtkHeader(
            dims=(grid.nx, grid.ny, 1),
            spacing=(grid.dx, grid.dy, 1.0),
            origin=(grid.origin_x, grid.origin_y, 0.0),
            time=rt.sim_time,
            step=rt.step_count,
            kn_range=(0.0, 0.0),
            cell_fields=(),
        )
        mesh = ensure_field_mesh(scene, props.mesh_object_name, header)
        apply_live_field_from_solver(
            mesh, rt.lib, rt.handle, props.active_field, props.show_regime_overlay
        )
        obj = bpy.data.objects.get(props.mesh_object_name)
        if obj:
            ensure_preview_material(obj, props.active_field)

        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    except RuntimeError as exc:
        props.sim_running = False
        _stop_runtime()
        print(f"Janus simulation error: {exc}")
        return None

    return 0.05 if props.sim_running else None


class JANUS_OT_simulate(Operator):
    bl_idname = "janus.simulate"
    bl_label = "Start Simulation"
    bl_description = "Run the Janus FFI solver with live viewport updates"

    def execute(self, context):
        props = context.scene.janus
        if props.sim_running:
            self.report({"WARNING"}, "Simulation already running")
            return {"CANCELLED"}

        try:
            from ..core.core_client import get_library

            lib = get_library()
        except FileNotFoundError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        _stop_runtime()
        rt = _SimRuntime
        rt.lib = lib
        rt.write_every = max(1, props.sim_steps_per_tick * 10)
        rt.output_dir = bpy.path.abspath(props.sim_output_dir)
        os.makedirs(rt.output_dir, exist_ok=True)

        try:
            case = json.loads(lib.default_case_json())
            rt.handle = lib.create_solver(case)
            lib.set_scheme(rt.handle, int(props.sim_scheme))
            rt.dt = lib.cfl_dt(rt.handle, props.sim_cfl)
            if rt.dt <= 0.0:
                raise RuntimeError("CFL timestep is zero — check grid and CFL setting")

            grid = lib.grid_info(rt.handle)
            header = JvtkHeader(
                dims=(grid.nx, grid.ny, 1),
                spacing=(grid.dx, grid.dy, 1.0),
                origin=(grid.origin_x, grid.origin_y, 0.0),
                time=0.0,
                step=0,
                kn_range=(0.0, 0.0),
                cell_fields=(),
            )
            mesh = ensure_field_mesh(context.scene, props.mesh_object_name, header)
            apply_live_field_from_solver(
                mesh, lib, rt.handle, props.active_field, props.show_regime_overlay
            )
            obj = bpy.data.objects.get(props.mesh_object_name)
            if obj:
                ensure_preview_material(obj, props.active_field)

            # Initial frame at t=0
            init_path = os.path.join(rt.output_dir, "live_0000.jvtk")
            lib.write_jvtk(rt.handle, init_path, 0.0, 0)
            rt.manifest_frames = [
                {"index": 0, "step": 0, "time": 0.0, "file": "live_0000.jvtk"}
            ]
            rt.frame_index = 1
        except RuntimeError as exc:
            _stop_runtime()
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        props.sim_running = True
        props.data_root = rt.output_dir
        bpy.app.timers.register(_sim_timer, first_interval=0.05)
        self.report({"INFO"}, f"Simulation started (dt={rt.dt:.3e})")
        return {"FINISHED"}


class JANUS_OT_simulate_stop(Operator):
    bl_idname = "janus.simulate_stop"
    bl_label = "Stop Simulation"
    bl_description = "Stop the live solver and write manifest.json"

    def execute(self, context):
        props = context.scene.janus
        rt = _SimRuntime

        if rt.handle and rt.lib and rt.output_dir:
            try:
                grid = rt.lib.grid_info(rt.handle)
                manifest_path = _write_manifest(rt.output_dir, grid, rt.write_every)
                props.manifest_path = manifest_path
                props.data_root = rt.output_dir
                props.cached_frame_index = -1
            except (RuntimeError, OSError) as exc:
                self.report({"WARNING"}, f"Manifest write failed: {exc}")

        props.sim_running = False
        _stop_runtime()
        self.report({"INFO"}, "Simulation stopped")
        return {"FINISHED"}
