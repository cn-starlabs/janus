"""Test scene generator for Janus Blender workflow validation."""

import bpy
import bmesh
from mathutils import Vector


def create_demo_grid(nx: int = 64, ny: int = 64, name: str = "JanusField") -> bpy.types.Object:
    """
    Create a rectangular grid mesh for testing.
    
    Args:
        nx: Grid cells in X direction.
        ny: Grid cells in Y direction.
        name: Name for the mesh object.
    
    Returns:
        Blender mesh object with grid topology.
    """
    # Create mesh and object
    mesh = bpy.data.meshes.new(name=name)
    obj = bpy.data.objects.new(name=name, object_data=mesh)
    bpy.context.collection.objects.link(obj)
    
    # Build grid using bmesh
    bm = bmesh.new()
    
    # Add vertices: (nx+1) x (ny+1) grid
    dx, dy = 1.0 / nx, 1.0 / ny
    verts = []
    for j in range(ny + 1):
        row = []
        for i in range(nx + 1):
            x, y = i * dx, j * dy
            v = bm.verts.new((x, y, 0.0))
            row.append(v)
        verts.append(row)
    
    # Add faces: nx x ny quads
    for j in range(ny):
        for i in range(nx):
            v0 = verts[j][i]
            v1 = verts[j][i + 1]
            v2 = verts[j + 1][i + 1]
            v3 = verts[j + 1][i]
            bm.faces.new([v0, v1, v2, v3])
    
    bm.to_mesh(mesh)
    bm.free()
    
    mesh.update()
    return obj


def create_boundary_markers(grid_obj: bpy.types.Object) -> dict:
    """
    Create boundary marker cubes positioned at domain edges.
    
    Args:
        grid_obj: Reference grid object to determine domain bounds.
    
    Returns:
        Dictionary with keys "west", "east", "south", "north" → bpy.types.Object.
    """
    grid = grid_obj.data
    
    # Compute bounds from grid vertices
    xs = [v.co.x for v in grid.vertices]
    ys = [v.co.y for v in grid.vertices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Size of marker cubes (slightly outside domain)
    size = 0.05
    offset = 0.15
    
    boundaries = {}
    positions = {
        "west": (x_min - offset, y_center, 0.0),
        "east": (x_max + offset, y_center, 0.0),
        "south": (x_center, y_min - offset, 0.0),
        "north": (x_center, y_max + offset, 0.0),
    }
    
    for role, (x, y, z) in positions.items():
        # Create cube
        bpy.ops.mesh.primitive_cube_add(size=size, location=(x, y, z))
        obj = bpy.context.active_object
        obj.name = f"Boundary_{role.capitalize()}"
        
        # Store role as custom property
        obj["janus_boundary_role"] = role
        obj["janus_boundary_temperature"] = 300.0
        obj["janus_boundary_velocity"] = [0.0, 0.0]
        obj["janus_boundary_kind"] = "DiffuseWall"
        
        boundaries[role] = obj
    
    return boundaries


def ensure_test_collection() -> bpy.types.Collection:
    """
    Create or get a test collection in the scene.
    
    Returns:
        Blender collection for test objects.
    """
    scene_col = bpy.context.scene.collection
    col_name = "JanusWorkflowTest"
    
    if col_name in bpy.data.collections:
        col = bpy.data.collections[col_name]
    else:
        col = bpy.data.collections.new(col_name)
        scene_col.children.link(col)
    
    return col


def setup_demo_scene(nx: int = 64, ny: int = 64) -> dict:
    """
    Create a complete demo scene with grid and boundary markers.
    
    Args:
        nx: Grid X cells.
        ny: Grid Y cells.
    
    Returns:
        Dictionary with keys "grid" → grid object, "boundaries" → boundary dict.
    """
    # Get or create test collection
    col = ensure_test_collection()
    
    # Clear existing objects in collection
    for obj in col.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Save current context
    original_col = bpy.context.view_layer.active_layer_collection.collection
    
    # Switch to test collection
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[col.name]
    
    # Create grid
    grid = create_demo_grid(nx=nx, ny=ny, name="JanusField")
    grid["janus_boundary_role"] = ""  # Grid itself is not a boundary
    
    # Create boundary markers
    boundaries = create_boundary_markers(grid)
    
    # Link all to collection
    for obj in [grid] + list(boundaries.values()):
        if obj.name not in col.objects:
            col.objects.link(obj)
    
    # Restore context
    bpy.context.view_layer.active_layer_collection = original_col.children[original_col.name]
    
    return {
        "grid": grid,
        "boundaries": boundaries,
        "collection": col,
    }


def validate_scene_setup(scene_dict: dict) -> bool:
    """
    Validate that demo scene was created correctly.
    
    Args:
        scene_dict: Result from setup_demo_scene().
    
    Returns:
        True if valid, raises AssertionError otherwise.
    """
    grid = scene_dict["grid"]
    boundaries = scene_dict["boundaries"]
    
    # Check grid exists and has correct name
    assert grid.name == "JanusField", f"Grid name is {grid.name}, expected 'JanusField'"
    assert isinstance(grid.data, bpy.types.Mesh), "Grid is not a mesh"
    
    # Check grid has faces
    n_faces = len(grid.data.polygons)
    assert n_faces == 64 * 64, f"Grid has {n_faces} faces, expected 4096"
    
    # Check all 4 boundaries exist
    for role in ["west", "east", "south", "north"]:
        assert role in boundaries, f"Missing boundary: {role}"
        obj = boundaries[role]
        assert obj["janus_boundary_role"] == role, f"Boundary {role} has wrong role property"
        assert "janus_boundary_temperature" in obj, f"Boundary {role} missing temperature"
        assert "janus_boundary_velocity" in obj, f"Boundary {role} missing velocity"
    
    print("✓ Demo scene validation passed")
    return True


if __name__ == "__main__":
    # Test scene creation in Blender
    print("Creating demo scene...")
    scene = setup_demo_scene(nx=32, ny=32)
    print(f"Grid: {scene['grid'].name}")
    print(f"Boundaries: {list(scene['boundaries'].keys())}")
    
    print("Validating scene...")
    validate_scene_setup(scene)
