import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "blender-addon" / "core" / "boundary_flow.py"

spec = importlib.util.spec_from_file_location("janus_boundary_flow", MODULE_PATH)
boundary_flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary_flow)


class BoundaryFlowTests(unittest.TestCase):
    def test_infer_boundary_role_from_geometry_prefers_left_edge(self):
        obj_bounds = (-0.05, 0.05, -0.2, 0.2, 0.0, 0.0)
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
        self.assertEqual(boundary_flow.infer_boundary_role_from_geometry(obj_bounds, ref_bounds), "west")

    def test_infer_boundary_role_from_geometry_prefers_top_edge(self):
        obj_bounds = (0.4, 0.6, 0.95, 1.05, 0.0, 0.0)
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
        self.assertEqual(boundary_flow.infer_boundary_role_from_geometry(obj_bounds, ref_bounds), "north")


if __name__ == "__main__":
    unittest.main()
