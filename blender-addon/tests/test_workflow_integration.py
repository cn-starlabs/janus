"""Integration tests for the complete Janus Blender workflow (pure-Python).

These tests orchestrate all workflow stages end-to-end using only pure-Python
components.  Blender-context stages (material nodes, mesh attribute I/O) are
validated structurally (source inspection) or via the lightweight mock objects
defined here.

Run with:
    python3 -m unittest blender-addon/tests/test_workflow_integration.py -v
"""

import json
import math
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Any

addon_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(addon_path))

from core.boundary_flow import infer_boundary_role_from_geometry


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

VALID_ROLES = {"west", "east", "south", "north"}
VALID_FIELDS = ["rho", "temperature", "kn_loc", "mom_x", "mom_y", "energy"]
VALID_SCHEMES = ["euler", "rk2", "rk4"]
SCHEME_ENUM_MAP = {0: "euler", 1: "rk2", 2: "rk4"}


# ---------------------------------------------------------------------------
# Pure-Python helpers that mirror Blender addon logic
# ---------------------------------------------------------------------------

def build_mock_grid_info(nx: int = 64, ny: int = 64) -> Dict[str, Any]:
    """Replicate grid info extracted by the setup operator."""
    dx = 1.0 / nx
    dy = 1.0 / ny
    return {
        "nx": nx, "ny": ny,
        "dx": dx, "dy": dy,
        "origin_x": 0.0, "origin_y": 0.0,
        "n_faces": nx * ny,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
    }


def build_mock_boundary_tags(roles: List[str]) -> List[Dict[str, Any]]:
    """Replicate boundary tag dicts stored by assign_boundary operator."""
    return [
        {
            "role": role,
            "kind": "DiffuseWall",
            "temperature": 300.0,
            "velocity": [0.0, 0.0],
        }
        for role in roles
    ]


def build_case_payload(
    grid: Dict[str, Any],
    boundary_tags: List[Dict[str, Any]],
    scheme_index: int = 2,  # 2 → rk4
) -> Dict[str, Any]:
    """Replicate _build_case_payload from operators/simulate.py."""
    scheme_name = SCHEME_ENUM_MAP.get(scheme_index, "rk4")
    bcs: Dict[str, Any] = {}
    for tag in boundary_tags:
        role = tag["role"]
        bcs[role] = {
            "DiffuseWall": {
                "temperature": tag["temperature"],
                "wall_velocity": tag["velocity"],
            }
        }
    # Fill missing BCs with Periodic
    for role in VALID_ROLES:
        if role not in bcs:
            bcs[role] = "Periodic"

    return {
        "config": {
            "grid": {
                "nx": grid["nx"], "ny": grid["ny"],
                "dx": grid["dx"], "dy": grid["dy"],
                "origin_x": grid["origin_x"], "origin_y": grid["origin_y"],
            },
            "bcs": bcs,
            "gas": {
                "r_gas": 208.13,
                "molar_mass": 0.039948,
                "vhs_omega": 0.81,
                "mu_ref": 2.117e-5,
                "t_ref": 273.15,
                "prandtl": 2.0 / 3.0,
            },
        },
        "initial": {"rho": 1.0, "temperature": 300.0, "velocity": [0.0, 0.0]},
        "velocity_grid": {"v_max": 1800.0, "n_per_dim": 25},
        "time_scheme": {"scheme": scheme_name},
        "scene": {"boundary_tags": boundary_tags},
    }


def simulate_status_updates(n_steps: int, dt: float) -> List[Dict[str, Any]]:
    """Simulate what the timer callback writes to JanusSceneProperties."""
    updates = []
    for i in range(n_steps):
        updates.append({
            "sim_running": True,
            "sim_current_step": i + 1,
            "sim_current_time": (i + 1) * dt,
            "sim_status": f"Running (rho)",
        })
    return updates


# ---------------------------------------------------------------------------
# Stage 1 + 2: Scene Setup & Boundary Assignment
# ---------------------------------------------------------------------------

class TestStage1And2_SceneAndBoundaries(unittest.TestCase):
    """Stages 1 & 2: grid creation + boundary role assignment."""

    def setUp(self):
        self.grid = build_mock_grid_info(64, 64)
        self.ref_bounds = (
            self.grid["x_min"], self.grid["x_max"],
            self.grid["y_min"], self.grid["y_max"],
            0.0, 0.1,
        )

    def test_grid_has_correct_face_count(self):
        self.assertEqual(self.grid["n_faces"], 64 * 64)

    def test_grid_dx_dy_reciprocal(self):
        self.assertAlmostEqual(self.grid["dx"] * self.grid["nx"], 1.0)
        self.assertAlmostEqual(self.grid["dy"] * self.grid["ny"], 1.0)

    def test_west_inference(self):
        west_bounds = (-0.2, 0.0, 0.4, 0.6, 0.0, 0.1)  # (xmin,xmax,ymin,ymax,zmin,zmax)
        role = infer_boundary_role_from_geometry(west_bounds, self.ref_bounds)
        self.assertEqual(role, "west")

    def test_east_inference(self):
        east_bounds = (1.0, 1.2, 0.4, 0.6, 0.0, 0.1)
        role = infer_boundary_role_from_geometry(east_bounds, self.ref_bounds)
        self.assertEqual(role, "east")

    def test_south_inference(self):
        south_bounds = (0.4, 0.6, -0.2, 0.0, 0.0, 0.1)
        role = infer_boundary_role_from_geometry(south_bounds, self.ref_bounds)
        self.assertEqual(role, "south")

    def test_north_inference(self):
        north_bounds = (0.4, 0.6, 1.0, 1.2, 0.0, 0.1)
        role = infer_boundary_role_from_geometry(north_bounds, self.ref_bounds)
        self.assertEqual(role, "north")

    def test_all_four_roles_inferrable(self):
        boundary_objects = [
            ((-0.2, 0.0, 0.4, 0.6, 0.0, 0.1), "west"),
            ((1.0, 1.2, 0.4, 0.6, 0.0, 0.1), "east"),
            ((0.4, 0.6, -0.2, 0.0, 0.0, 0.1), "south"),
            ((0.4, 0.6, 1.0, 1.2, 0.0, 0.1), "north"),
        ]
        for bounds, expected in boundary_objects:
            role = infer_boundary_role_from_geometry(bounds, self.ref_bounds)
            self.assertEqual(role, expected, f"Expected '{expected}', got '{role}'")

    def test_boundary_tags_have_required_fields(self):
        tags = build_mock_boundary_tags(list(VALID_ROLES))
        required = {"role", "kind", "temperature", "velocity"}
        for tag in tags:
            self.assertTrue(required.issubset(tag.keys()), f"Missing keys in tag: {tag}")

    def test_boundary_roles_are_valid(self):
        tags = build_mock_boundary_tags(list(VALID_ROLES))
        for tag in tags:
            self.assertIn(tag["role"], VALID_ROLES)


# ---------------------------------------------------------------------------
# Stage 4: Case Payload Generation
# ---------------------------------------------------------------------------

class TestStage4_CasePayload(unittest.TestCase):
    """Stage 4: payload generation with time_scheme."""

    def setUp(self):
        grid = build_mock_grid_info(64, 64)
        tags = build_mock_boundary_tags(["west", "east", "south", "north"])
        self.payload = build_case_payload(grid, tags, scheme_index=2)

    def test_time_scheme_present(self):
        self.assertIn("time_scheme", self.payload)

    def test_scheme_is_rk4(self):
        self.assertEqual(self.payload["time_scheme"]["scheme"], "rk4")

    def test_config_grid_present(self):
        self.assertIn("grid", self.payload["config"])
        g = self.payload["config"]["grid"]
        self.assertEqual(g["nx"], 64)
        self.assertEqual(g["ny"], 64)
        self.assertAlmostEqual(g["dx"], 1.0 / 64)

    def test_bcs_cover_all_edges(self):
        bcs = self.payload["config"]["bcs"]
        for role in VALID_ROLES:
            self.assertIn(role, bcs)

    def test_gas_properties_present(self):
        gas = self.payload["config"]["gas"]
        for key in ["r_gas", "molar_mass", "vhs_omega", "mu_ref", "t_ref", "prandtl"]:
            self.assertIn(key, gas)

    def test_initial_conditions_present(self):
        init = self.payload["initial"]
        self.assertIn("rho", init)
        self.assertIn("temperature", init)
        self.assertIn("velocity", init)
        self.assertGreater(init["rho"], 0)
        self.assertGreater(init["temperature"], 0)

    def test_velocity_grid_present(self):
        vg = self.payload["velocity_grid"]
        self.assertIn("v_max", vg)
        self.assertIn("n_per_dim", vg)
        self.assertGreater(vg["v_max"], 0)
        self.assertGreater(vg["n_per_dim"], 0)

    def test_scene_boundary_tags_present(self):
        self.assertIn("boundary_tags", self.payload["scene"])
        self.assertEqual(len(self.payload["scene"]["boundary_tags"]), 4)

    def test_json_serializable(self):
        try:
            json_str = json.dumps(self.payload)
        except (TypeError, ValueError) as e:
            self.fail(f"Payload is not JSON-serializable: {e}")
        self.assertIn('"rk4"', json_str)

    def test_json_roundtrip(self):
        json_str = json.dumps(self.payload)
        restored = json.loads(json_str)
        self.assertEqual(restored["time_scheme"]["scheme"], "rk4")
        self.assertEqual(restored["config"]["grid"]["nx"], 64)

    def test_euler_scheme_payload(self):
        grid = build_mock_grid_info()
        tags = build_mock_boundary_tags(["west", "east"])
        p = build_case_payload(grid, tags, scheme_index=0)
        self.assertEqual(p["time_scheme"]["scheme"], "euler")

    def test_rk2_scheme_payload(self):
        grid = build_mock_grid_info()
        tags = build_mock_boundary_tags(["west"])
        p = build_case_payload(grid, tags, scheme_index=1)
        self.assertEqual(p["time_scheme"]["scheme"], "rk2")

    def test_missing_bc_defaults_to_periodic(self):
        """BCs not in boundary_tags should default to Periodic."""
        grid = build_mock_grid_info()
        tags = build_mock_boundary_tags(["west"])  # only west provided
        p = build_case_payload(grid, tags, scheme_index=2)
        bcs = p["config"]["bcs"]
        self.assertEqual(bcs.get("east"), "Periodic")
        self.assertEqual(bcs.get("south"), "Periodic")
        self.assertEqual(bcs.get("north"), "Periodic")
        self.assertNotEqual(bcs.get("west"), "Periodic")


# ---------------------------------------------------------------------------
# Stage 6: Live Status Updates
# ---------------------------------------------------------------------------

class TestStage6_LiveStatusUpdates(unittest.TestCase):
    """Stage 6: sim_status / sim_current_time / sim_current_step updates."""

    def setUp(self):
        self.dt = 1.5e-7  # typical kinetic timestep
        self.n_steps = 10
        self.updates = simulate_status_updates(self.n_steps, self.dt)

    def test_correct_number_of_updates(self):
        self.assertEqual(len(self.updates), self.n_steps)

    def test_step_increments_by_one(self):
        for i, update in enumerate(self.updates):
            self.assertEqual(update["sim_current_step"], i + 1)

    def test_time_accumulates_correctly(self):
        for i, update in enumerate(self.updates):
            expected_time = (i + 1) * self.dt
            self.assertAlmostEqual(
                update["sim_current_time"], expected_time,
                places=20,
                msg=f"Step {i+1}: time mismatch"
            )

    def test_sim_running_is_true(self):
        for update in self.updates:
            self.assertTrue(update["sim_running"])

    def test_status_string_non_empty(self):
        for update in self.updates:
            self.assertIsInstance(update["sim_status"], str)
            self.assertGreater(len(update["sim_status"]), 0)

    def test_final_time_matches_n_steps_times_dt(self):
        final = self.updates[-1]
        self.assertAlmostEqual(final["sim_current_time"], self.n_steps * self.dt, places=20)

    def test_final_step_count(self):
        self.assertEqual(self.updates[-1]["sim_current_step"], self.n_steps)

    def test_time_precision_large_step_count(self):
        """Accumulated float error over many steps should stay below 1 ns."""
        n = 10_000
        dt = 1.5e-7
        updates = simulate_status_updates(n, dt)
        final_time = updates[-1]["sim_current_time"]
        expected = n * dt
        self.assertAlmostEqual(final_time, expected, delta=1e-9)

    def test_negative_dt_produces_decreasing_time(self):
        """Sanity check: negative dt inverts time accumulation."""
        updates = simulate_status_updates(5, -1e-7)
        for i, u in enumerate(updates):
            self.assertAlmostEqual(u["sim_current_time"], (i + 1) * -1e-7)


# ---------------------------------------------------------------------------
# Stage 7: Visualization requirements
# ---------------------------------------------------------------------------

class TestStage7_VisualizationRequirements(unittest.TestCase):
    """Stage 7: field data population and material binding requirements."""

    def test_valid_fields_list(self):
        expected = {"rho", "temperature", "kn_loc", "mom_x", "mom_y", "energy"}
        self.assertEqual(set(VALID_FIELDS), expected)

    def test_kn_regime_field_name(self):
        """kn_regime is the enum value for overlay mode, not an attribute."""
        # It should NOT be in the per-cell attribute list
        self.assertNotIn("kn_regime", VALID_FIELDS)

    def test_kn_loc_in_attributes_for_regime_overlay(self):
        """kn_loc must be present for regime overlay to work."""
        self.assertIn("kn_loc", VALID_FIELDS)

    def test_field_data_positive_density(self):
        """Density field values must be positive."""
        mock_rho = [1.0 + 0.01 * math.sin(i * 0.1) for i in range(64 * 64)]
        self.assertTrue(all(r > 0 for r in mock_rho), "All rho values must be positive")

    def test_field_data_length_matches_grid(self):
        """Field array length must equal nx*ny."""
        nx, ny = 64, 64
        mock_field = [1.0] * (nx * ny)
        self.assertEqual(len(mock_field), nx * ny)

    def test_nan_detection_in_field(self):
        """NaN in field data should be detectable."""
        field = [1.0, 2.0, float("nan"), 4.0]
        has_nan = any(math.isnan(v) for v in field)
        self.assertTrue(has_nan)

    def test_inf_detection_in_field(self):
        """Inf in field data should be detectable."""
        field = [1.0, float("inf"), 3.0]
        has_inf = any(math.isinf(v) for v in field)
        self.assertTrue(has_inf)

    def test_field_value_normalization(self):
        """Values should normalize to [0,1] for color mapping."""
        field = [0.5, 1.0, 1.5, 2.0, 2.5]
        f_min, f_max = min(field), max(field)
        normalized = [(v - f_min) / (f_max - f_min) for v in field]
        self.assertAlmostEqual(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[-1], 1.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in normalized))


# ---------------------------------------------------------------------------
# End-to-end pipeline integration
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration(unittest.TestCase):
    """Run all stages in sequence and verify data flows correctly."""

    def test_stage1_to_stage4_pipeline(self):
        """Grid + boundaries → payload is valid and JSON-serializable."""
        # Stage 1: grid
        grid = build_mock_grid_info(32, 32)
        self.assertEqual(grid["n_faces"], 1024)

        # Stage 2: boundary tags
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.1)
        tagged = []
        placements = [
            ((-0.2, 0.0, 0.4, 0.6, 0.0, 0.1), "west"),
            ((1.0, 1.2, 0.4, 0.6, 0.0, 0.1), "east"),
            ((0.4, 0.6, -0.2, 0.0, 0.0, 0.1), "south"),
            ((0.4, 0.6, 1.0, 1.2, 0.0, 0.1), "north"),
        ]
        for bounds, expected_role in placements:
            inferred = infer_boundary_role_from_geometry(bounds, ref_bounds)
            self.assertEqual(inferred, expected_role)
            tagged.append({
                "role": inferred,
                "kind": "DiffuseWall",
                "temperature": 300.0,
                "velocity": [0.0, 0.0],
            })

        # Stage 4: payload
        payload = build_case_payload(grid, tagged, scheme_index=2)
        self.assertEqual(payload["time_scheme"]["scheme"], "rk4")
        self.assertEqual(payload["config"]["grid"]["nx"], 32)
        self.assertEqual(len(payload["scene"]["boundary_tags"]), 4)

        # Verify JSON serializable
        json_str = json.dumps(payload)
        restored = json.loads(json_str)
        self.assertEqual(restored["time_scheme"]["scheme"], "rk4")

    def test_stage6_status_over_many_steps(self):
        """100 status updates accumulate time correctly."""
        dt = 2.0e-8
        updates = simulate_status_updates(100, dt)
        self.assertEqual(updates[-1]["sim_current_step"], 100)
        self.assertAlmostEqual(updates[-1]["sim_current_time"], 100 * dt, places=20)

    def test_stage7_field_length_and_validity(self):
        """Mock field of correct size with valid values."""
        nx, ny = 64, 64
        n = nx * ny
        rho = [1.225 + 0.001 * math.sin(i) for i in range(n)]
        self.assertEqual(len(rho), n)
        self.assertTrue(all(r > 0 for r in rho))
        self.assertFalse(any(math.isnan(r) for r in rho))
        self.assertFalse(any(math.isinf(r) for r in rho))

    def test_scheme_mapping_completeness(self):
        """All three scheme indices map to valid names."""
        for idx in range(3):
            name = SCHEME_ENUM_MAP[idx]
            self.assertIn(name, VALID_SCHEMES)

    def test_boundary_inference_stable_under_offset_variation(self):
        """Role inference is stable across different offsets (0.05 to 0.5)."""
        ref_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 0.1)
        for offset in [0.05, 0.1, 0.15, 0.3, 0.5]:
            west_bounds = (-offset, 0.0, 0.4, 0.6, 0.0, 0.1)
            role = infer_boundary_role_from_geometry(west_bounds, ref_bounds)
            self.assertEqual(role, "west", f"West inference failed at offset={offset}")


class TestSourceFileCompleteness(unittest.TestCase):
    """Verify that all required source files exist and are non-empty."""

    BASE = Path(__file__).resolve().parent.parent

    def _check(self, rel_path: str, *symbols: str):
        p = self.BASE / rel_path
        self.assertTrue(p.exists(), f"Missing: {rel_path}")
        content = p.read_text()
        for sym in symbols:
            self.assertIn(sym, content, f"Symbol '{sym}' not found in {rel_path}")

    def test_properties_module(self):
        self._check("properties.py", "sim_status", "sim_running", "sim_current_time", "sim_current_step")

    def test_visualization_operators(self):
        self._check(
            "operators/visualization.py",
            "JANUS_OT_setup_visualization",
            "JANUS_OT_update_field_material",
        )

    def test_visualization_core(self):
        self._check(
            "core/visualization.py",
            "setup_field_material",
            "setup_regime_overlay_material",
            "ensure_attribute_domain",
        )

    def test_boundary_flow_core(self):
        self._check("core/boundary_flow.py", "infer_boundary_role_from_geometry")

    def test_jvtk_reader_core(self):
        self._check("core/jvtk_reader.py", "JvtkReader", "JvtkHeader", "MAGIC")

    def test_operators_boundaries(self):
        self._check("operators/boundaries.py", "self.report")

    def test_operators_simulate(self):
        self._check("operators/simulate.py", "janus")


if __name__ == "__main__":
    unittest.main(verbosity=2)
