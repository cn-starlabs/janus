"""Timeline synchronization operators."""

import bpy
from bpy.types import Operator

from ..core import jvtk_reader
from ..core.timeline_utils import resolve_manifest_frame_path


class JANUS_OT_sync_timeline(Operator):
    bl_idname = "janus.sync_timeline"
    bl_label = "Sync Timeline"
    bl_description = "Set Blender frame range to match the loaded manifest"

    def execute(self, context):
        scene = context.scene
        props = scene.janus
        if not props.manifest_path:
            self.report({"ERROR"}, "No manifest loaded")
            return {"CANCELLED"}
        try:
            manifest = jvtk_reader.load_manifest(props.manifest_path)
        except OSError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        n_frames = len(manifest.get("frames", []))
        if n_frames == 0:
            self.report({"ERROR"}, "Manifest has no frames")
            return {"CANCELLED"}

        scene.frame_start = props.frame_offset
        scene.frame_end = props.frame_offset + n_frames - 1
        scene.frame_current = props.frame_offset
        props.cached_frame_index = -1

        if n_frames > 1:
            scene.frame_step = 1

        self.report({"INFO"}, f"Timeline set to frames {scene.frame_start}–{scene.frame_end}")
        return {"FINISHED"}
