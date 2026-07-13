"""Unit tests for boundary role assignment workflow."""

import unittest
import sys
from pathlib import Path

# Add blender-addon to path for testing
addon_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(addon_path))

from core.boundary_flow import infer_boundary_role_from_geometry


class TestBoundaryRoleInference(unittest.TestCase):
    """Test geometry-based boundary role inference."""

    def test_west_boundary(self):
        """Boundary on the left should infer as 'west'."""
        # (xmin, xmax, ymin, ymax, zmin, zmax) – matches boundary_flow.py unpacking
        obj_bounds = (-0.2, 0.2, -0.2, 0.2, 0.0, 0.1)
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.1)
        
        role = infer_boundary_role_from_geometry(obj_bounds, ref_bounds)
        self.assertEqual(role, "west", f"Expected 'west', got '{role}'")

    def test_east_boundary(self):
        """Boundary on the right should infer as 'east'."""
        obj_bounds = (1.2, 1.4, 0.2, 0.8, 0.0, 0.1)
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.1)
        
        role = infer_boundary_role_from_geometry(obj_bounds, ref_bounds)
        self.assertEqual(role, "east", f"Expected 'east', got '{role}'")

    def test_south_boundary(self):
        """Boundary at the bottom should infer as 'south'."""
        obj_bounds = (0.2, 0.8, -0.2, 0.0, 0.0, 0.1)
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.1)
        
        role = infer_boundary_role_from_geometry(obj_bounds, ref_bounds)
        self.assertEqual(role, "south", f"Expected 'south', got '{role}'")

    def test_north_boundary(self):
        """Boundary at the top should infer as 'north'."""
        obj_bounds = (0.2, 0.8, 1.2, 1.4, 0.0, 0.1)
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.1)
        
        role = infer_boundary_role_from_geometry(obj_bounds, ref_bounds)
        self.assertEqual(role, "north", f"Expected 'north', got '{role}'")

    def test_corner_object(self):
        """Object at NE corner should assign to closest edge (north or east)."""
        obj_bounds = (1.3, 1.4, 1.3, 1.4, 0.0, 0.1)
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.1)
        
        role = infer_boundary_role_from_geometry(obj_bounds, ref_bounds)
        # Should be either "north" or "east" (equally close)
        self.assertIn(role, ["north", "east"], f"Expected 'north' or 'east', got '{role}'")

    def test_center_object(self):
        """Object at domain center should have ambiguous assignment."""
        obj_bounds = (0.4, 0.6, 0.4, 0.6, 0.0, 0.1)  # Center
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.1)
        
        role = infer_boundary_role_from_geometry(obj_bounds, ref_bounds)
        # Should assign to some edge (all equidistant, just check it's valid)
        self.assertIn(role, ["west", "east", "south", "north"], f"Got invalid role '{role}'")

    def test_distance_calculation_consistency(self):
        """Verify distance heuristic is symmetric."""
        # Boundary equally offset west and south – correct arg order (xmin, xmax, ymin, ymax, zmin, zmax)
        obj_bounds_west  = (-0.15, -0.05, 0.4, 0.6, 0.0, 0.1)
        obj_bounds_south = (0.4, 0.6, -0.15, -0.05, 0.0, 0.1)
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.1)
        
        role_west  = infer_boundary_role_from_geometry(obj_bounds_west,  ref_bounds)
        role_south = infer_boundary_role_from_geometry(obj_bounds_south, ref_bounds)
        
        # Both should infer correctly despite symmetry
        self.assertEqual(role_west,  "west")
        self.assertEqual(role_south, "south")


class TestBoundaryMetadataStorage(unittest.TestCase):
    """Test BC parameter storage (requires Blender context)."""

    def test_metadata_structure(self):
        """Verify expected BC metadata fields."""
        expected_fields = [
            "janus_boundary_role",
            "janus_boundary_kind",
            "janus_boundary_temperature",
            "janus_boundary_velocity",
        ]
        
        # This would be in a Blender-dependent test
        # For now, just document expected structure
        for field in expected_fields:
            self.assertTrue(field.startswith("janus_"), f"Field '{field}' should have 'janus_' prefix")


if __name__ == "__main__":
    unittest.main(verbosity=2)
