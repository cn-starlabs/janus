"""High-performance material node and geometry node setup for field visualization."""

import bpy
from mathutils import Color


def setup_field_material(obj: bpy.types.Object, field_name: str = "rho") -> bpy.types.Material:
    """
    Create/update a material for visualizing a field with vertex colors.
    
    The material uses a Principled BSDF with a value-to-RGB color ramp,
    allowing real-time recoloring of field data without mesh rebuilds.
    
    Args:
        obj: Blender mesh object to apply material to.
        field_name: Name of the field attribute (e.g., 'rho', 'temperature', 'kn_loc').
    
    Returns:
        The configured material object.
    """
    # Create or get material
    mat_name = f"JanusField_{field_name}"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
        mat.use_nodes = True
        # Clear existing nodes
        mat.node_tree.nodes.clear()
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Input node: grab the attribute from mesh
    attr_node = nodes.new(type="ShaderNodeAttribute")
    attr_node.attribute_name = field_name

    # Value-to-RGB color ramp
    ramp_node = nodes.new(type="ShaderNodeValToRGB")
    ramp = ramp_node.color_ramp

    # Configure ramp stops for field visualization
    # Default: 0 = black, 1 = white (linear interpolation)
    if len(ramp.elements) < 2:
        ramp.elements.new(0.5)

    # Stop 0 (value=0.0): dark blue
    ramp.elements[0].position = 0.0
    ramp.elements[0].color = Color((0.0, 0.2, 0.8, 1.0))

    # Stop 1 (value=1.0): bright yellow transitioning to red
    ramp.elements[1].position = 1.0
    ramp.elements[1].color = Color((1.0, 0.3, 0.0, 1.0))

    # Principled BSDF (modern physically-based shader)
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.inputs["Subsurface Weight"].default_value = 0.0
    bsdf_node.inputs["Coat Weight"].default_value = 0.0
    bsdf_node.inputs["Emission Strength"].default_value = 0.0

    # Output node
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Connect: attribute -> color ramp -> BSDF base color -> output
    links.new(attr_node.outputs["Fac"], ramp_node.inputs["Fac"])
    links.new(ramp_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Apply material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return mat


def setup_regime_overlay_material(obj: bpy.types.Object) -> bpy.types.Material:
    """
    Create a material for Kn regime visualization with discrete color bands.
    
    Maps local Knudsen number (kn_loc) to flow regime colors:
    - Kn < 0.01: blue (continuum, dark blue)
    - 0.01 ≤ Kn < 0.1: green (slip, light green)
    - 0.1 ≤ Kn < 10: yellow (transition, bright yellow)
    - Kn ≥ 10: red (free molecular, bright red)
    
    Uses a ColorRamp with multiple stops to create sharp regime boundaries.
    
    Args:
        obj: Blender mesh object.
    
    Returns:
        The regime overlay material.
    """
    mat_name = "JanusRegimeOverlay"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
    else:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Attribute node: read kn_loc
    attr_node = nodes.new(type="ShaderNodeAttribute")
    attr_node.attribute_name = "kn_loc"

    # Logarithmic scale: kn_loc is typically 0.001 to 100+
    # Map to [0, 1] via log10 with normalization:
    #   log10(0.001) = -3 → 0.0 (continuum)
    #   log10(0.01) = -2 → 0.25
    #   log10(0.1) = -1 → 0.5
    #   log10(1.0) = 0 → 0.75
    #   log10(100) = 2 → 1.0
    # Use a Math node to apply log10, then ramp
    log_node = nodes.new(type="ShaderNodeMath")
    log_node.operation = "LOGARITHM"
    log_node.inputs[1].default_value = 10.0  # base 10

    # Map log scale to [0, 1]: (log + 3) / 5 ≈ (log + 3) / 5
    # Clamp to [0, 1]
    scale_node = nodes.new(type="ShaderNodeMath")
    scale_node.operation = "ADD"
    scale_node.inputs[1].default_value = 3.0

    div_node = nodes.new(type="ShaderNodeMath")
    div_node.operation = "DIVIDE"
    div_node.inputs[1].default_value = 5.0

    clamp_node = nodes.new(type="ShaderNodeMath")
    clamp_node.operation = "MAXIMUM"
    clamp_node.inputs[1].default_value = 0.0

    clamp_node2 = nodes.new(type="ShaderNodeMath")
    clamp_node2.operation = "MINIMUM"
    clamp_node2.inputs[1].default_value = 1.0

    # Color ramp with discrete stops
    ramp_node = nodes.new(type="ShaderNodeValToRGB")
    ramp = ramp_node.color_ramp
    ramp.interpolation = "LINEAR"

    # Clear default elements
    while len(ramp.elements) > 0:
        ramp.elements.remove(ramp.elements[0])

    # Add stops for flow regimes (normalized to [0, 1] via log scale above)
    # log10(0.01) = -2 → (−2 + 3) / 5 = 0.2 (blue/green boundary)
    e0 = ramp.elements.new(0.0)
    e0.color = Color((0.0, 0.2, 0.9, 1.0))  # Continuum: dark blue

    e1 = ramp.elements.new(0.2)
    e1.color = Color((0.0, 0.8, 0.2, 1.0))  # Slip: green

    # log10(0.1) = -1 → (−1 + 3) / 5 = 0.4 (green/yellow boundary)
    e2 = ramp.elements.new(0.4)
    e2.color = Color((1.0, 1.0, 0.0, 1.0))  # Transition: yellow

    # log10(10) = 1 → (1 + 3) / 5 = 0.8 (yellow/red boundary)
    e3 = ramp.elements.new(0.8)
    e3.color = Color((1.0, 0.0, 0.0, 1.0))  # Free molecular: red

    e4 = ramp.elements.new(1.0)
    e4.color = Color((0.6, 0.0, 0.2, 1.0))  # Far free molecular: dark red

    # Principled BSDF
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.inputs["Subsurface Weight"].default_value = 0.0

    # Output
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Connect pipeline: attribute → log → scale → div → clamp → clamp2 → ramp → BSDF → output
    links.new(attr_node.outputs["Fac"], log_node.inputs[0])
    links.new(log_node.outputs["Value"], scale_node.inputs[0])
    links.new(scale_node.outputs["Value"], div_node.inputs[0])
    links.new(div_node.outputs["Value"], clamp_node.inputs[0])
    links.new(clamp_node.outputs["Value"], clamp_node2.inputs[0])
    links.new(clamp_node2.outputs["Value"], ramp_node.inputs["Fac"])
    links.new(ramp_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Apply material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return mat


def setup_geometry_nodes_for_field(obj: bpy.types.Object, field_name: str = "rho") -> bpy.types.GeometryNodeTree:
    """
    Set up a geometry nodes modifier for field visualization with optional vertex coloring.
    
    This is useful for future enhancements like:
    - Instancing glyphs at vertices colored by field value
    - Creating streamlines from vector fields
    - Cutting slices for 3D volumetric display
    
    Args:
        obj: Target object.
        field_name: Mesh attribute name to visualize.
    
    Returns:
        The geometry node tree.
    """
    # For now, this is a placeholder for future enhancements.
    # Current visualization uses direct material node setup (see setup_field_material).
    # Future extensions could add:
    # - Glyph instancing with scale/color based on field magnitude
    # - Volume object creation from point cloud
    # - Streamline generation from velocity fields
    
    # Check if geometry nodes modifier already exists
    geo_mod = None
    for mod in obj.modifiers:
        if mod.type == "GEOMETRY":
            geo_mod = mod
            break

    if geo_mod is None:
        geo_mod = obj.modifiers.new(name="GeometryNodes", type="GEOMETRY")

    # Ensure there's a node tree
    if geo_mod.node_group is None:
        geo_mod.node_group = bpy.data.node_groups.new(name="FieldGeoNodes", type="GeometryNodeTree")

    tree = geo_mod.node_group
    return tree


def ensure_attribute_domain(obj: bpy.types.Object, attr_name: str, domain: str = "FACE", dtype: str = "FLOAT") -> bpy.types.Attribute:
    """
    Ensure a mesh attribute exists with the given domain and data type.
    
    Args:
        obj: Blender mesh object.
        attr_name: Name of the attribute.
        domain: Attribute domain ("FACE", "VERTEX", "POINT", "CORNER", "EDGE").
        dtype: Data type ("FLOAT", "INT", "STRING", "FLOAT_VECTOR", "FLOAT_COLOR").
    
    Returns:
        The attribute (created if it did not exist).
    """
    if not isinstance(obj.data, bpy.types.Mesh):
        raise TypeError(f"Object {obj.name} is not a mesh")

    mesh = obj.data
    if attr_name not in mesh.attributes:
        mesh.attributes.new(name=attr_name, type=dtype, domain=domain)

    return mesh.attributes[attr_name]
