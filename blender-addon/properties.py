"""Addon RNA properties."""

import bpy


class JanusAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    library_path: bpy.props.StringProperty(
        name="Janus FFI Library",
        description="Path to libjanus_ffi.so / janus_ffi.dll / libjanus_ffi.dylib",
        subtype="FILE_PATH",
        default="",
    )
    repository_path: bpy.props.StringProperty(
        name="Janus Repository Path",
        description="Path to the Janus repository root used by the Build Janus FFI action",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "library_path")
        layout.prop(self, "repository_path")
        layout.label(text="Build: cargo build -p janus-ffi --release", icon="INFO")


class JanusSceneProperties(bpy.types.PropertyGroup):
    data_root: bpy.props.StringProperty(
        name="Data Directory",
        description="Directory containing manifest.json and .jvtk frames",
        subtype="DIR_PATH",
        default="",
    )
    manifest_path: bpy.props.StringProperty(
        name="Manifest",
        description="Path to manifest.json (auto-filled on import)",
        subtype="FILE_PATH",
        default="",
    )
    mesh_object_name: bpy.props.StringProperty(
        name="Field Mesh",
        description="Name of the Blender mesh object displaying Janus fields",
        default="JanusField",
    )
    frame_offset: bpy.props.IntProperty(
        name="Frame Offset",
        description="Blender frame subtracted before indexing into the manifest",
        default=1,
        min=0,
    )
    auto_sync_timeline: bpy.props.BoolProperty(
        name="Auto Sync Timeline",
        description="Update field mesh when the timeline frame changes",
        default=True,
    )
    active_field: bpy.props.EnumProperty(
        name="Field",
        items=[
            ("rho", "Density (rho)", "Mass density"),
            ("temperature", "Temperature", "Gas temperature"),
            ("kn_loc", "Kn (local)", "Local Knudsen number"),
            ("mom_x", "Momentum X", "X momentum component"),
            ("mom_y", "Momentum Y", "Y momentum component"),
            ("energy", "Energy", "Total energy density"),
            ("kn_regime", "Kn Regime", "Color by flow regime bands"),
            ("p_free", "Particle Fraction (p_free)", "Free-transport fraction"),
            ("particle_count_density", "Particle Density", "Particle count density per cell"),
        ],
        default="rho",
    )
    show_regime_overlay: bpy.props.BoolProperty(
        name="Regime Overlay",
        description="Color by Kn regime bands",
        default=False,
    )
    extrude_z: bpy.props.FloatProperty(
        name="Z Thickness",
        description="Thin extrusion for 2D cases (meters)",
        default=0.0,
        min=0.0,
        soft_max=1.0e-2,
    )
    show_wireframe: bpy.props.BoolProperty(
        name="Show Wireframe",
        description="Display the field mesh as a wireframe for easier inspection",
        default=False,
    )
    sim_running: bpy.props.BoolProperty(name="Simulation Running", default=False)
    sim_status: bpy.props.StringProperty(
        name="Status",
        description="Current simulation status and diagnostics",
        default="Idle",
    )
    sim_current_time: bpy.props.FloatProperty(
        name="Sim Time",
        description="Current simulation time (seconds)",
        default=0.0,
        options={"HIDDEN"},
    )
    sim_current_step: bpy.props.IntProperty(
        name="Step",
        description="Current simulation step count",
        default=0,
        options={"HIDDEN"},
    )
    sim_steps_per_tick: bpy.props.IntProperty(
        name="Steps / Tick",
        default=5,
        min=1,
        max=1000,
    )
    sim_cfl: bpy.props.FloatProperty(name="CFL", default=0.4, min=0.01, max=0.95)
    sim_scheme: bpy.props.EnumProperty(
        name="Integrator",
        items=[("0", "Euler", ""), ("1", "RK2", ""), ("2", "RK4", "")],
        default="0",
    )
    sim_output_dir: bpy.props.StringProperty(
        name="Sim Output Dir",
        subtype="DIR_PATH",
        default="/home/pana/janus/output",
    )
    sim_kernel: bpy.props.EnumProperty(
        name="Solver Kernel",
        description="The kinetic solver physics kernel",
        items=[
            ("ugkwp", "UGKWP (coupled)", "Full unified gas-kinetic wave-particle coupling"),
            ("dugks", "DUGKS (wave-only)", "Pure-wave deterministic fast path"),
        ],
        default="ugkwp",
    )
    sim_seed: bpy.props.IntProperty(
        name="RNG Seed",
        description="Random number generator seed for particle sampling",
        default=12345,
        min=1,
    )
    sim_kn_threshold: bpy.props.FloatProperty(
        name="Kn Cutoff",
        description="Threshold above which particles are sampled in UGKWP",
        default=0.1,
        min=0.0,
        max=10.0,
    )

    bc_role: bpy.props.EnumProperty(
        name="Boundary Role",
        items=[
            ("west", "West", "Left boundary"),
            ("east", "East", "Right boundary"),
            ("south", "South", "Bottom boundary"),
            ("north", "North", "Top boundary"),
            ("wall", "Wall", "Generic wall"),
        ],
        default="west",
    )
    bc_object_name: bpy.props.StringProperty(
        name="Boundary Object",
        description="Blender object to tag as a boundary source",
        default="",
    )
    bc_temperature: bpy.props.FloatProperty(
        name="Wall Temperature",
        description="Temperature used for diffuse wall assignments",
        default=300.0,
        min=0.0,
    )
    bc_wall_velocity_x: bpy.props.FloatProperty(
        name="Wall Ux",
        description="Wall velocity X component",
        default=0.0,
    )
    bc_wall_velocity_y: bpy.props.FloatProperty(
        name="Wall Uy",
        description="Wall velocity Y component",
        default=0.0,
    )
    cached_frame_index: bpy.props.IntProperty(default=-1)
