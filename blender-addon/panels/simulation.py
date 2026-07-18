"""Simulation control panel."""

import bpy
from bpy.types import Panel


class JANUS_PT_simulation(Panel):
    bl_label = "Simulation"
    bl_idname = "JANUS_PT_simulation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Janus"

    def draw(self, context):
        layout = self.layout
        props = context.scene.janus

        col = layout.column(align=True)
        col.label(text="Data Import", icon="IMPORT")
        col.operator("janus.import_manifest", icon="FILE_FOLDER")
        col.operator("janus.import_jvtk", icon="FILE")
        col.prop(props, "manifest_path")
        col.prop(props, "data_root")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Live Solver", icon="PLAY")
        row = col.row(align=True)
        if props.sim_running:
            row.operator("janus.simulate_stop", icon="PAUSE", text="Stop")
        else:
            row.operator("janus.simulate", icon="PLAY", text="Start")
        col.prop(props, "sim_cfl")
        col.prop(props, "sim_scheme")
        col.prop(props, "sim_kernel")
        if props.sim_kernel == 'ugkwp':
            col.prop(props, "sim_seed")
            col.prop(props, "sim_kn_threshold")
        col.prop(props, "sim_steps_per_tick")
        col.prop(props, "sim_output_dir")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Live Status", icon="PLAY")
        col.label(text=props.sim_status, icon="INFO")
        if props.sim_running:
            row = col.row()
            row.label(text=f"Time: {props.sim_current_time:.3e} s")
            row = col.row()
            row.label(text=f"Step: {props.sim_current_step}")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Setup", icon="MESH_GRID")
        col.operator("janus.build_ffi", icon="SETTINGS")
        col.operator("janus.build_grid", icon="MESH_PLANE")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Timeline", icon="TIME")
        col.prop(props, "auto_sync_timeline")
        col.prop(props, "frame_offset")
        col.operator("janus.sync_timeline", icon="SORTTIME")
