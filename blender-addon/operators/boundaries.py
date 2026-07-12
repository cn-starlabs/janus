"""Geometry-driven boundary assignment operators."""

import bpy
from bpy.types import Operator


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

        obj["janus_boundary_role"] = role
        obj["janus_boundary_kind"] = role_name
        obj["janus_boundary_temperature"] = props.bc_temperature
        obj["janus_boundary_velocity"] = [props.bc_wall_velocity_x, props.bc_wall_velocity_y]

        self.report({"INFO"}, f"Assigned {obj.name} -> {role_name}")
        return {"FINISHED"}
