"""Build structured-grid mesh and apply scalar fields as face attributes."""

from __future__ import annotations

import bpy
import bmesh
import numpy as np
from mathutils import Vector

from .jvtk_reader import JvtkHeader, JvtkReader


def kn_regime_color(kn: float) -> tuple[float, float, float, float]:
    """Map local Kn to regime RGBA per janus.md §6.2."""
    if kn < 0.01:
        return (0.1, 0.2, 0.9, 1.0)  # continuum — blue
    if kn < 0.1:
        return (0.1, 0.8, 0.2, 1.0)  # slip — green
    if kn < 10.0:
        return (0.95, 0.85, 0.1, 1.0)  # transition — yellow
    return (0.95, 0.15, 0.1, 1.0)  # free molecular — red


def ensure_field_mesh(
    scene: bpy.types.Scene,
    name: str,
    header: JvtkHeader,
    extrusion_z: float | None = None,
) -> bpy.types.Mesh:
    obj = bpy.data.objects.get(name)
    nx, ny, nz = header.dims
    dx, dy, _dz = header.spacing
    ox, oy, _oz = header.origin
    z_scale = header.spacing[2] if nz > 1 else max(dx, dy) * 0.01

    if extrusion_z is None and hasattr(scene, "janus"):
        extrusion_z = float(getattr(scene.janus, "extrude_z", 0.0))
    if extrusion_z is None:
        extrusion_z = 0.0

    if obj is None:
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
    else:
        mesh = obj.data

    bm = bmesh.new()
    verts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = ox + i * dx
            y = oy + j * dy
            verts.append(bm.verts.new((x, y, 0.0)))
    bm.verts.ensure_lookup_table()
    for j in range(ny):
        for i in range(nx):
            v0 = verts[j * (nx + 1) + i]
            v1 = verts[j * (nx + 1) + i + 1]
            v2 = verts[(j + 1) * (nx + 1) + i + 1]
            v3 = verts[(j + 1) * (nx + 1) + i]
            try:
                bm.faces.new((v0, v1, v2, v3))
            except ValueError:
                pass

    if extrusion_z > 0.0:
        geom = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
        extruded_verts = [e for e in geom["geom"] if isinstance(e, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, verts=extruded_verts, vec=Vector((0.0, 0.0, extrusion_z)))

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    if obj:
        obj.location.z = 0.0
        obj.show_wire = bool(getattr(getattr(scene, "janus", None), "show_wireframe", False))

    return mesh


def _normalize_scalar_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.float64)
    arr = np.where(finite, arr, 0.0)
    vmin = np.min(arr)
    vmax = np.max(arr)
    if np.isclose(vmin, vmax):
        return np.full(arr.shape, 0.5, dtype=np.float64)
    return (arr - vmin) / (vmax - vmin)


def _set_face_scalar(mesh: bpy.types.Mesh, name: str, values: np.ndarray):
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(value_array) == 0:
        value_array = np.zeros(len(mesh.polygons), dtype=np.float64)
    elif len(value_array) != len(mesh.polygons):
        repeated = np.zeros(len(mesh.polygons), dtype=np.float64)
        if len(value_array) > 0:
            for idx in range(len(mesh.polygons)):
                repeated[idx] = value_array[idx % len(value_array)]
        value_array = repeated

    normalized = _normalize_scalar_values(value_array)
    attr = mesh.attributes.get(name)
    if attr is None:
        attr = mesh.attributes.new(name=name, type="FLOAT", domain="FACE")
    attr.data.foreach_set("value", normalized.astype(np.float64).tolist())
    mesh.update()


def _set_face_color(mesh: bpy.types.Mesh, name: str, rgba: list[tuple[float, float, float, float]]):
    attr = mesh.attributes.get(name)
    if attr is None:
        attr = mesh.attributes.new(name=name, type="BYTE_COLOR", domain="FACE")
    flat = []
    for r, g, b, a in rgba:
        flat.extend((int(r * 255), int(g * 255), int(b * 255), int(a * 255)))
    attr.data.foreach_set("color", flat)
    mesh.update()


def apply_field_array_to_mesh(
    mesh: bpy.types.Mesh,
    field_name: str,
    values: np.ndarray,
    regime_overlay: bool = False,
    kn_values: np.ndarray | None = None,
):
    if field_name == "kn_regime" or regime_overlay:
        kn = kn_values if kn_values is not None else values
        colors = [kn_regime_color(float(k)) for k in kn]
        _set_face_color(mesh, "janus_regime", colors)
        _set_face_scalar(mesh, "kn_loc", kn)
        return
    _set_face_scalar(mesh, field_name, values)


def apply_field_to_mesh(
    mesh: bpy.types.Mesh,
    reader: JvtkReader,
    field_name: str,
    regime_overlay: bool = False,
):
    if field_name == "kn_regime" or regime_overlay:
        kn = reader.cell_field("kn_loc")
        apply_field_array_to_mesh(mesh, field_name, kn, regime_overlay=True, kn_values=kn)
        return
    try:
        values = reader.cell_field(field_name)
    except KeyError:
        values = np.zeros(reader.header.ncells, dtype=np.float64)
    apply_field_array_to_mesh(mesh, field_name, values, regime_overlay=False)


def apply_live_field_from_solver(
    mesh: bpy.types.Mesh,
    lib,
    handle: int,
    field_name: str,
    regime_overlay: bool = False,
):
    from .numpy_bridge import array_from_ptr

    ffi_name = "kn_loc" if field_name == "kn_regime" else field_name
    ptr, length = lib.field_view(handle, ffi_name)
    values = array_from_ptr(ptr, length)
    kn_values = None
    if field_name == "kn_regime" or regime_overlay:
        kn_ptr, kn_len = lib.field_view(handle, "kn_loc")
        kn_values = array_from_ptr(kn_ptr, kn_len)
    apply_field_array_to_mesh(mesh, field_name, values, regime_overlay, kn_values)


def ensure_preview_material(obj: bpy.types.Object, field_name: str = "rho"):
    mat_name = "JanusFieldPreview"
    mat = bpy.data.materials.get(mat_name)
    attr_name = "janus_regime" if field_name == "kn_regime" else field_name
    use_color_attr = field_name == "kn_regime"

    if mat is None:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = attr_name
        if use_color_attr:
            links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            ramp = nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].position = 0.0
            ramp.color_ramp.elements[1].position = 1.0
            ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.2, 1.0)
            ramp.color_ramp.elements[1].color = (1.0, 0.2, 0.0, 1.0)
            links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
            links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        mat.blend_method = "BLEND"
    else:
        for node in mat.node_tree.nodes:
            if node.type == "ATTRIBUTE":
                node.attribute_name = attr_name

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
