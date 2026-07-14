"""Import/export and viewport refresh operators."""

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from ..core import jvtk_reader
from ..core.mesh_builder import apply_field_to_mesh, ensure_field_mesh, ensure_preview_material


class JANUS_OT_import_manifest(Operator):
    bl_idname = "janus.import_manifest"
    bl_label = "Import Manifest"
    bl_description = "Load manifest.json and display the first frame"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        scene = context.scene
        props = scene.janus
        manifest_path = bpy.path.abspath(self.filepath or props.manifest_path)
        if not manifest_path:
            self.report({"ERROR"}, "No manifest path set")
            return {"CANCELLED"}
        try:
            manifest = jvtk_reader.load_manifest(manifest_path)
        except OSError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        from pathlib import Path
        props.manifest_path = manifest_path
        props.data_root = str(Path(bpy.path.abspath(manifest_path)).parent)
        props.cached_frame_index = -1

        if manifest.get("frames"):
            frame = manifest["frames"][0]
            jvtk_path = f"{props.data_root}/{frame['file']}"
            reader = jvtk_reader.JvtkReader.open(jvtk_path)
            mesh = ensure_field_mesh(scene, props.mesh_object_name, reader.header)
            apply_field_to_mesh(mesh, reader, props.active_field, props.show_regime_overlay)
            obj = bpy.data.objects.get(props.mesh_object_name)
            if obj:
                ensure_preview_material(obj, props.active_field)
            reader.close()

        self.report({"INFO"}, f"Loaded manifest with {len(manifest.get('frames', []))} frames")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class JANUS_OT_import_jvtk(Operator):
    bl_idname = "janus.import_jvtk"
    bl_label = "Import JVTK Frame"
    bl_description = "Load a single .jvtk snapshot"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        scene = context.scene
        props = scene.janus
        path = bpy.path.abspath(self.filepath)
        try:
            reader = jvtk_reader.JvtkReader.open(path)
        except OSError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        mesh = ensure_field_mesh(scene, props.mesh_object_name, reader.header)
        apply_field_to_mesh(mesh, reader, props.active_field, props.show_regime_overlay)
        obj = bpy.data.objects.get(props.mesh_object_name)
        if obj:
            ensure_preview_material(obj, props.active_field)
        reader.close()
        props.cached_frame_index = -1
        self.report({"INFO"}, f"Loaded {path}")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class JANUS_OT_refresh_viewport(Operator):
    bl_idname = "janus.refresh_viewport"
    bl_label = "Refresh Viewport"
    bl_description = "Re-apply the active field from the current manifest frame"

    def execute(self, context):
        props = context.scene.janus
        props.cached_frame_index = -1
        from ..handlers import _on_frame_change

        _on_frame_change(context.scene, context.evaluated_depsgraph_get())
        return {"FINISHED"}
