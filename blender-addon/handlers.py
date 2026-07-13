"""Frame-change handlers for timeline scrubbing."""

import bpy

from .core import jvtk_reader
from .core.mesh_builder import apply_field_to_mesh, ensure_field_mesh
from .core.timeline_utils import frame_index_from_blender_frame, resolve_manifest_frame_path


def _on_frame_change(scene, depsgraph):
    props = scene.janus
    if not props.auto_sync_timeline or not props.manifest_path:
        return
    try:
        manifest = jvtk_reader.load_manifest(props.manifest_path)
    except OSError:
        return
    blender_frame = scene.frame_current
    frame_index = frame_index_from_blender_frame(blender_frame, props.frame_offset)
    if frame_index < 0 or frame_index >= len(manifest["frames"]):
        return
    if frame_index == props.cached_frame_index:
        return
    try:
        jvtk_path = resolve_manifest_frame_path(
            manifest,
            props.manifest_path,
            props.data_root,
            frame_index,
        )
        reader = jvtk_reader.JvtkReader.open(jvtk_path)
    except (IndexError, OSError, ValueError):
        return
    mesh = ensure_field_mesh(scene, props.mesh_object_name, reader.header)
    apply_field_to_mesh(mesh, reader, props.active_field, props.show_regime_overlay)
    props.cached_frame_index = frame_index


def register_handlers():
    if _on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame_change)


def unregister_handlers():
    if _on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_on_frame_change)
