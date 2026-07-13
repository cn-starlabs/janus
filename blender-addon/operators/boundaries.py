"""Geometry-driven boundary assignment operators."""

import bpy
from bpy.types import Operator

from ..core.boundary_flow import infer_boundary_role_from_geometry


class JANUS_OT_assign_boundary(Operator):
    bl_idname = "janus.assign_boundary"
    bl_label = "Assign Boundary"
    bl_description = "Tag the selected Blender object as a Janus boundary source"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.janus
        obj = bpy.data.objects.get(props.bc_object_name)
        if obj is None:
            self.report({"ERROR"}, "Boundary object not found")
            return {"CANCELLED"}

        if not obj.data:
            self.report({"ERROR"}, "Boundary object has no data")
            return {"CANCELLED"}

        role = props.bc_role
        if role == "wall":
            role_name = "DiffuseWall"
        else:
            role_name = role.capitalize()

        # Infer a domain edge from the object's geometry relative to the field mesh.
        inferred_role = role
        scene = context.scene
        ref_obj = bpy.data.objects.get(getattr(scene.janus, "mesh_object_name", ""))
        if ref_obj is not None and getattr(ref_obj, "bound_box", None):
            bounds = ref_obj.bound_box
            if bounds:
                domain_bounds = (
                    min(v[0] for v in bounds),
                    max(v[0] for v in bounds),
                    min(v[1] for v in bounds),
                    max(v[1] for v in bounds),
                    min(v[2] for v in bounds),
                    max(v[2] for v in bounds),
                )
                obj_bounds = (
                    min(v[0] for v in obj.bound_box),
                    max(v[0] for v in obj.bound_box),
                    min(v[1] for v in obj.bound_box),
                    max(v[1] for v in obj.bound_box),
                    min(v[2] for v in obj.bound_box),
                    max(v[2] for v in obj.bound_box),
                )
                try:
                    inferred_role = infer_boundary_role_from_geometry(obj_bounds, domain_bounds)
                except ValueError:
                    inferred_role = role

        if inferred_role == "wall":
            role_name = "DiffuseWall"
        else:
            role_name = inferred_role.capitalize()

        obj["janus_boundary_role"] = inferred_role
        obj["janus_boundary_kind"] = role_name
        obj["janus_boundary_temperature"] = props.bc_temperature
        obj["janus_boundary_velocity"] = [props.bc_wall_velocity_x, props.bc_wall_velocity_y]

        self.report({"INFO"}, f"Assigned {obj.name} -> {role_name} (inferred: {inferred_role})")
        return {"FINISHED"}
