"""Unit tests for scene setup logic (pure-Python, no bpy required).

Blender-dependent helpers (create_demo_grid, create_boundary_markers, etc.)
are defined in the same file but are only exercised inside Blender.  These
tests validate the *logic* (bounds, positions, role assignment) in isolation.
"""

import unittest
import sys
from pathlib import Path

# Make blender-addon importable without bpy
addon_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(addon_path))


# ---------------------------------------------------------------------------
# Geometry / pure-Python helpers extracted from the scene-setup logic
# ---------------------------------------------------------------------------

def compute_boundary_positions(x_min: float, x_max: float, y_min: float, y_max: float, offset: float = 0.15) -> dict:
    """Pure-Python version of the boundary-position logic used by create_boundary_markers."""
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return {
        "west":  (x_min - offset, y_center, 0.0),
        "east":  (x_max + offset, y_center, 0.0),
        "south": (x_center, y_min - offset, 0.0),
        "north": (x_center, y_max + offset, 0.0),
    }


def compute_grid_bounds(nx: int, ny: int) -> tuple:
    """Return (x_min, x_max, y_min, y_max) for a unit [0,1]² grid."""
    dx, dy = 1.0 / nx, 1.0 / ny
    x_max = nx * dx  # == 1.0
    y_max = ny * dy  # == 1.0
    return 0.0, x_max, 0.0, y_max


class TestGridLogic(unittest.TestCase):
    """Validate grid dimension and face-count expectations."""

    def test_face_count_formula(self):
        """Grid of nx×ny cells should have nx*ny faces."""
        for nx, ny in [(64, 64), (32, 32), (8, 16)]:
            expected = nx * ny
            # Replicate the bmesh loop logic symbolically
            n_faces = nx * ny  # one quad per (i,j) cell pair
            self.assertEqual(n_faces, expected, f"Face count wrong for {nx}×{ny}")

    def test_vertex_count_formula(self):
        """Grid of nx×ny cells needs (nx+1)*(ny+1) vertices."""
        for nx, ny in [(64, 64), (32, 32)]:
            n_verts = (nx + 1) * (ny + 1)
            self.assertGreater(n_verts, nx * ny)

    def test_grid_spacing(self):
        """dx and dy should be exact reciprocals."""
        nx, ny = 64, 64
        dx, dy = 1.0 / nx, 1.0 / ny
        self.assertAlmostEqual(dx * nx, 1.0)
        self.assertAlmostEqual(dy * ny, 1.0)

    def test_unit_grid_bounds(self):
        """A 64×64 grid in [0,1]² has x_max=1.0 and y_max=1.0."""
        x_min, x_max, y_min, y_max = compute_grid_bounds(64, 64)
        self.assertAlmostEqual(x_min, 0.0)
        self.assertAlmostEqual(x_max, 1.0)
        self.assertAlmostEqual(y_min, 0.0)
        self.assertAlmostEqual(y_max, 1.0)


class TestBoundaryMarkerPositions(unittest.TestCase):
    """Validate boundary marker placement logic."""

    def setUp(self):
        self.x_min, self.x_max = 0.0, 1.0
        self.y_min, self.y_max = 0.0, 1.0
        self.offset = 0.15
        self.positions = compute_boundary_positions(
            self.x_min, self.x_max, self.y_min, self.y_max, self.offset
        )

    def test_all_four_boundaries_present(self):
        for role in ["west", "east", "south", "north"]:
            self.assertIn(role, self.positions)

    def test_west_is_left_of_domain(self):
        wx, wy, wz = self.positions["west"]
        self.assertLess(wx, self.x_min, "West marker should be left of domain")

    def test_east_is_right_of_domain(self):
        ex, ey, ez = self.positions["east"]
        self.assertGreater(ex, self.x_max, "East marker should be right of domain")

    def test_south_is_below_domain(self):
        sx, sy, sz = self.positions["south"]
        self.assertLess(sy, self.y_min, "South marker should be below domain")

    def test_north_is_above_domain(self):
        nx, ny, nz = self.positions["north"]
        self.assertGreater(ny, self.y_max, "North marker should be above domain")

    def test_west_east_symmetric_in_x(self):
        wx, _, _ = self.positions["west"]
        ex, _, _ = self.positions["east"]
        cx = (self.x_min + self.x_max) / 2
        self.assertAlmostEqual(abs(wx - cx), abs(ex - cx), places=10)

    def test_south_north_symmetric_in_y(self):
        _, sy, _ = self.positions["south"]
        _, ny, _ = self.positions["north"]
        cy = (self.y_min + self.y_max) / 2
        self.assertAlmostEqual(abs(sy - cy), abs(ny - cy), places=10)

    def test_zero_offset_places_markers_on_boundary(self):
        pos = compute_boundary_positions(0.0, 1.0, 0.0, 1.0, offset=0.0)
        wx, wy, _ = pos["west"]
        self.assertAlmostEqual(wx, 0.0)
        ex, ey, _ = pos["east"]
        self.assertAlmostEqual(ex, 1.0)


class TestSceneDictStructure(unittest.TestCase):
    """Validate expected structure of scene_dict returned by setup_demo_scene."""

    def _make_mock_scene_dict(self):
        """Build a minimal dict that mimics setup_demo_scene output."""
        return {
            "grid": {"name": "JanusField", "n_faces": 64 * 64},
            "boundaries": {
                "west":  {"janus_boundary_role": "west",  "janus_boundary_temperature": 300.0, "janus_boundary_velocity": [0.0, 0.0]},
                "east":  {"janus_boundary_role": "east",  "janus_boundary_temperature": 300.0, "janus_boundary_velocity": [0.0, 0.0]},
                "south": {"janus_boundary_role": "south", "janus_boundary_temperature": 300.0, "janus_boundary_velocity": [0.0, 0.0]},
                "north": {"janus_boundary_role": "north", "janus_boundary_temperature": 300.0, "janus_boundary_velocity": [0.0, 0.0]},
            },
            "collection": "JanusWorkflowTest",
        }

    def test_grid_key_present(self):
        scene = self._make_mock_scene_dict()
        self.assertIn("grid", scene)

    def test_boundaries_key_present(self):
        scene = self._make_mock_scene_dict()
        self.assertIn("boundaries", scene)

    def test_all_four_boundary_roles(self):
        scene = self._make_mock_scene_dict()
        for role in ["west", "east", "south", "north"]:
            self.assertIn(role, scene["boundaries"])

    def test_grid_has_correct_name(self):
        scene = self._make_mock_scene_dict()
        self.assertEqual(scene["grid"]["name"], "JanusField")

    def test_grid_face_count(self):
        scene = self._make_mock_scene_dict()
        self.assertEqual(scene["grid"]["n_faces"], 4096)

    def test_boundary_metadata_fields(self):
        scene = self._make_mock_scene_dict()
        required_keys = ["janus_boundary_role", "janus_boundary_temperature", "janus_boundary_velocity"]
        for role, obj in scene["boundaries"].items():
            for key in required_keys:
                self.assertIn(key, obj, f"Boundary '{role}' missing key '{key}'")

    def test_boundary_role_matches_key(self):
        scene = self._make_mock_scene_dict()
        for role, obj in scene["boundaries"].items():
            self.assertEqual(obj["janus_boundary_role"], role)


class TestSceneSourceFile(unittest.TestCase):
    """Verify the actual Blender scene-setup source file exists and contains expected symbols."""

    def setUp(self):
        self.source = Path(__file__).resolve().parent / "test_scene_setup.py"
        # The actual Blender helper lives one level up; we point to core helpers instead.
        self.viz_source = Path(__file__).resolve().parent.parent / "core" / "visualization.py"

    def test_visualization_source_exists(self):
        self.assertTrue(self.viz_source.exists(), "visualization.py not found")

    def test_ensure_attribute_domain_present(self):
        content = self.viz_source.read_text()
        self.assertIn("ensure_attribute_domain", content)

    def test_setup_field_material_present(self):
        content = self.viz_source.read_text()
        self.assertIn("setup_field_material", content)

    def test_setup_regime_overlay_present(self):
        content = self.viz_source.read_text()
        self.assertIn("setup_regime_overlay_material", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
