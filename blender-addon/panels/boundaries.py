"""Boundary condition panel (M3: default-case overview; geometry tagging later)."""

import json

import bpy
from bpy.types import Panel


def _bc_label(kind) -> str:
    if isinstance(kind, str):
        return kind
    if not isinstance(kind, dict):
        return str(kind)
    if "DiffuseWall" in kind:
        dw = kind["DiffuseWall"]
        u, v = dw.get("wall_velocity", [0.0, 0.0])
        return f"Diffuse wall T={dw.get('temperature', 0):.0f}K u=({u:.1f},{v:.1f})"
    if "Periodic" in kind:
        return "Periodic"
    if "SpecularWall" in kind:
        return "Specular wall"
    if "VelocityInlet" in kind:
        vi = kind["VelocityInlet"]
        return f"Velocity inlet rho={vi.get('density', 0):.2g}"
    if "Outlet" in kind:
        return "Outlet (Neumann)"
    if "Symmetry" in kind:
        return "Symmetry"
    return str(kind)


class JANUS_PT_boundaries(Panel):
    bl_label = "Boundaries"
    bl_idname = "JANUS_PT_boundaries"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Janus"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.janus

        try:
            from ..core.core_client import get_library

            lib = get_library()
            case = json.loads(lib.default_case_json())
        except (FileNotFoundError, RuntimeError) as exc:
            layout.label(text=str(exc), icon="ERROR")
            layout.label(text="Build: cargo build -p janus-ffi --release")
            return

        grid = case["config"]["grid"]
        bcs = case["config"]["bcs"]

        col = layout.column(align=True)
        col.label(text=f"Grid: {grid['nx']} x {grid['ny']}", icon="MESH_GRID")
        col.label(text=f"dx={grid['dx']:.2e}  dy={grid['dy']:.2e}")

        layout.separator()
        for edge in ("west", "east", "south", "north"):
            box = layout.box()
            box.label(text=edge.capitalize(), icon="MOD_WAVE")
            box.label(text=_bc_label(bcs[edge]))

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Geometry-driven BC tagging", icon="MOD_WAVE")
        col.prop(props, "bc_role")
        col.prop_search(props, "bc_object_name", bpy.data, "objects")
        col.prop(props, "bc_temperature")
        col.prop(props, "bc_wall_velocity_x")
        col.prop(props, "bc_wall_velocity_y")
        col.operator("janus.assign_boundary", icon="ADD")

        layout.separator()
        tagged = []
        for obj in bpy.data.objects:
            if "janus_boundary_role" in obj:
                tagged.append((obj.name, obj["janus_boundary_role"]))
        if tagged:
            box = layout.box()
            box.label(text="Tagged objects", icon="OBJECT_DATA")
            for name, role in tagged:
                box.label(text=f"{name}: {role}")
        else:
            layout.label(text="No tagged boundary objects yet", icon="INFO")
