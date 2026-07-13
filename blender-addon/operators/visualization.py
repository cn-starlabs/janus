"""Visualization operators: material setup, field display, regime overlay."""

import bpy
from ..core.visualization import (
    setup_field_material,
    setup_regime_overlay_material,
    ensure_attribute_domain,
)


class JANUS_OT_setup_visualization(bpy.types.Operator):
    """Set up high-performance materials and geometry nodes for field visualization."""

    bl_idname = "janus.setup_visualization"
    bl_label = "Setup Visualization Materials"

    def execute(self, context):
        props = context.scene.janus
        mesh_name = props.mesh_object_name

        if mesh_name not in bpy.data.objects:
            self.report({"ERROR"}, f"Mesh object '{mesh_name}' not found")
            return {"CANCELLED"}

        obj = bpy.data.objects[mesh_name]
        if not isinstance(obj.data, bpy.types.Mesh):
            self.report({"ERROR"}, f"Object '{mesh_name}' is not a mesh")
            return {"CANCELLED"}

        # Ensure key attributes exist
        for attr in ["rho", "temperature", "kn_loc", "mom_x", "mom_y", "energy"]:
            ensure_attribute_domain(obj, attr, domain="FACE", dtype="FLOAT")

        # Set up field material for current active field
        active_field = props.active_field
        if active_field in ["rho", "temperature", "kn_loc", "mom_x", "mom_y", "energy"]:
            try:
                setup_field_material(obj, field_name=active_field)
                self.report({"INFO"}, f"Set up material for field '{active_field}'")
            except Exception as e:
                self.report({"ERROR"}, f"Failed to set up material: {e}")
                return {"CANCELLED"}

        # Set up regime overlay material if requested
        if props.show_regime_overlay:
            try:
                ensure_attribute_domain(obj, "kn_loc", domain="FACE", dtype="FLOAT")
                setup_regime_overlay_material(obj)
                self.report({"INFO"}, "Set up regime overlay material")
            except Exception as e:
                self.report({"ERROR"}, f"Failed to set up regime overlay: {e}")
                return {"CANCELLED"}

        # Enable viewport shading to see materials
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"
                        break

        return {"FINISHED"}


class JANUS_OT_update_field_material(bpy.types.Operator):
    """Update visualization material for a different field."""

    bl_idname = "janus.update_field_material"
    bl_label = "Update Field Material"

    def execute(self, context):
        props = context.scene.janus
        mesh_name = props.mesh_object_name

        if mesh_name not in bpy.data.objects:
            self.report({"ERROR"}, f"Mesh object '{mesh_name}' not found")
            return {"CANCELLED"}

        obj = bpy.data.objects[mesh_name]
        if not isinstance(obj.data, bpy.types.Mesh):
            self.report({"ERROR"}, f"Object '{mesh_name}' is not a mesh")
            return {"CANCELLED"}

        # Switch to the new active field material
        field = props.active_field
        if field in ["rho", "temperature", "kn_loc", "mom_x", "mom_y", "energy"]:
            try:
                setup_field_material(obj, field_name=field)
                self.report({"INFO"}, f"Updated material for field '{field}'")
            except Exception as e:
                self.report({"ERROR"}, f"Material update failed: {e}")
                return {"CANCELLED"}
        elif field == "kn_regime":
            # Show regime overlay instead
            try:
                ensure_attribute_domain(obj, "kn_loc", domain="FACE", dtype="FLOAT")
                setup_regime_overlay_material(obj)
                self.report({"INFO"}, "Switched to regime overlay view")
            except Exception as e:
                self.report({"ERROR"}, f"Regime overlay failed: {e}")
                return {"CANCELLED"}

        return {"FINISHED"}


def register():
    """Register operators."""
    bpy.utils.register_class(JANUS_OT_setup_visualization)
    bpy.utils.register_class(JANUS_OT_update_field_material)


def unregister():
    """Unregister operators."""
    bpy.utils.unregister_class(JANUS_OT_setup_visualization)
    bpy.utils.unregister_class(JANUS_OT_update_field_material)
