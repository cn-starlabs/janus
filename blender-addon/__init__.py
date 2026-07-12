bl_info = {
    "name": "Janus CFD",
    "author": "Janus Project",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Janus",
    "description": "Cross-scale gas dynamics simulator integration (UGKWP / .jvtk playback)",
    "category": "Physics",
}

import bpy

from . import handlers
from .operators import boundaries as boundary_ops
from .operators import import_export, setup, simulate, timeline
from .panels import boundaries, simulation, visualization
from .properties import JanusAddonPreferences, JanusSceneProperties


CLASSES = (
    JanusAddonPreferences,
    JanusSceneProperties,
    import_export.JANUS_OT_import_manifest,
    import_export.JANUS_OT_import_jvtk,
    import_export.JANUS_OT_refresh_viewport,
    setup.JANUS_OT_build_ffi,
    setup.JANUS_OT_build_grid,
    boundary_ops.JANUS_OT_assign_boundary,
    simulate.JANUS_OT_simulate,
    simulate.JANUS_OT_simulate_stop,
    timeline.JANUS_OT_sync_timeline,
    simulation.JANUS_PT_simulation,
    boundaries.JANUS_PT_boundaries,
    visualization.JANUS_PT_visualization,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.janus = bpy.props.PointerProperty(type=JanusSceneProperties)
    handlers.register_handlers()


def unregister():
    handlers.unregister_handlers()
    del bpy.types.Scene.janus
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
