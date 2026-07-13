"""Field visualization panel."""

import bpy
from bpy.types import Panel


class JANUS_PT_visualization(Panel):
    bl_label = "Visualization"
    bl_idname = "JANUS_PT_visualization"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Janus"

    def draw(self, context):
        layout = self.layout
        props = context.scene.janus

        col = layout.column(align=True)
        col.label(text="Mesh", icon="MESH_GRID")
        col.prop(props, "mesh_object_name")
        col.prop(props, "extrude_z")
        col.prop(props, "show_wireframe")

        layout.separator()
        row = layout.row(align=True)
        row.operator("janus.build_grid", icon="MESH_PLANE")
        row.operator("janus.refresh_viewport", icon="FILE_REFRESH")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Field", icon="RESTRICT_VIEW_OFF")
        col.prop(props, "active_field")
        col.prop(props, "show_regime_overlay")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Visualization Materials", icon="SHADING_RENDERED")
        col.operator("janus.setup_visualization", text="Setup Materials", icon="MATERIAL")
        col.operator("janus.update_field_material", text="Update Field Display", icon="MATSHADERBALL")

        obj = bpy.data.objects.get(props.mesh_object_name)
        if obj and obj.type == "MESH":
            mesh = obj.data
            layout.separator()
            layout.label(text=f"Faces: {len(mesh.polygons)}", icon="MESH_DATA")
            attrs = [a.name for a in mesh.attributes if a.domain == "FACE"]
            if attrs:
                layout.label(text=f"Attributes: {', '.join(attrs[:4])}")
            else:
                layout.label(text="No face attributes yet", icon="INFO")
        else:
            layout.separator()
            layout.label(text="No field mesh found yet", icon="INFO")
