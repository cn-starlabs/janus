"""Helpers for converting Blender geometry into Janus boundary assignments."""

from __future__ import annotations

from typing import Iterable


def infer_boundary_role_from_geometry(obj_bounds: Iterable[float], reference_bounds: Iterable[float]) -> str:
    """Infer the nearest domain edge for a geometry object.

    The heuristic uses the object's axis-aligned bounds and the reference domain
    bounds. The closest edge wins, with a small tolerance so tiny objects do not
    flip between adjacent edges.
    """

    obj_min_x, obj_max_x, obj_min_y, obj_max_y, _obj_min_z, _obj_max_z = list(obj_bounds)
    ref_min_x, ref_max_x, ref_min_y, ref_max_y, _ref_min_z, _ref_max_z = list(reference_bounds)

    width = max(ref_max_x - ref_min_x, 1e-9)
    height = max(ref_max_y - ref_min_y, 1e-9)

    distances = {
        "west": abs(obj_max_x - ref_min_x),
        "east": abs(obj_min_x - ref_max_x),
        "south": abs(obj_max_y - ref_min_y),
        "north": abs(obj_min_y - ref_max_y),
    }

    # Prefer the boundary that is closest in normalized space.
    normalized = {
        name: dist / (width if name in {"west", "east"} else height)
        for name, dist in distances.items()
    }
    best_role, _best_dist = min(normalized.items(), key=lambda item: item[1])
    return best_role
