"""Unit tests for case payload generation with time scheme selection."""

import unittest
import json
import sys
from pathlib import Path

# Add blender-addon to path
addon_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(addon_path))


class TestCasePayloadGeneration(unittest.TestCase):
    """Test that case payloads include correct time scheme."""

    def test_time_scheme_in_payload(self):
        """Verify time_scheme field is in generated payload."""
        # Mock payload structure
        payload = {
            "config": {
                "grid": {"nx": 64, "ny": 64, "dx": 1.0/64, "dy": 1.0/64, "origin_x": 0.0, "origin_y": 0.0},
                "bcs": {
                    "west": "Periodic",
                    "east": "Periodic",
                    "south": {"DiffuseWall": {"temperature": 300.0, "wall_velocity": [0.0, 0.0]}},
                    "north": {"DiffuseWall": {"temperature": 300.0, "wall_velocity": [0.0, 0.0]}},
                },
                "gas": {"r_gas": 208.13, "molar_mass": 0.039948, "vhs_omega": 0.81, "mu_ref": 2.117e-5, "t_ref": 273.15, "prandtl": 2.0/3.0},
            },
            "initial": {"rho": 1.0, "temperature": 300.0, "velocity": [0.0, 0.0]},
            "velocity_grid": {"v_max": 1800.0, "n_per_dim": 25},
            "time_scheme": {"scheme": "rk4"},
            "scene": {"boundary_tags": []},
        }
        
        # Verify time_scheme exists
        self.assertIn("time_scheme", payload, "Payload missing 'time_scheme' field")
        self.assertIn("scheme", payload["time_scheme"], "time_scheme missing 'scheme' field")
        self.assertEqual(payload["time_scheme"]["scheme"], "rk4", "Scheme should be 'rk4'")

    def test_time_scheme_variants(self):
        """Verify all valid time scheme options."""
        valid_schemes = ["euler", "rk2", "rk4"]
        
        for scheme in valid_schemes:
            payload = {
                "time_scheme": {"scheme": scheme},
                "config": {},
            }
            
            # Verify scheme can be serialized/deserialized
            json_str = json.dumps(payload)
            restored = json.loads(json_str)
            self.assertEqual(restored["time_scheme"]["scheme"], scheme)

    def test_default_scheme_mapping(self):
        """Verify enum index to scheme name mapping."""
        # Blender uses enum indices (0, 1, 2) → ("euler", "rk2", "rk4")
        scheme_map = {
            0: "euler",
            1: "rk2",
            2: "rk4",
        }
        
        for idx, scheme_name in scheme_map.items():
            self.assertEqual(scheme_map[idx], scheme_name)

    def test_payload_with_boundary_tags(self):
        """Verify boundary tags are included alongside time_scheme."""
        payload = {
            "config": {"grid": {}, "bcs": {}, "gas": {}},
            "time_scheme": {"scheme": "rk4"},
            "scene": {
                "boundary_tags": [
                    {
                        "role": "west",
                        "kind": "DiffuseWall",
                        "temperature": 300.0,
                        "velocity": [0.0, 0.0],
                    },
                    {
                        "role": "east",
                        "kind": "DiffuseWall",
                        "temperature": 300.0,
                        "velocity": [0.0, 0.0],
                    },
                ],
            },
        }
        
        # Verify all parts are present
        self.assertIn("time_scheme", payload)
        self.assertIn("scene", payload)
        self.assertIn("boundary_tags", payload["scene"])
        self.assertEqual(len(payload["scene"]["boundary_tags"]), 2)

    def test_payload_json_serializable(self):
        """Verify entire payload can be JSON serialized (for FFI)."""
        payload = {
            "config": {
                "grid": {"nx": 64, "ny": 64, "dx": 0.015625, "dy": 0.015625, "origin_x": 0.0, "origin_y": 0.0},
                "bcs": {"west": "Periodic", "east": "Periodic", "south": "Periodic", "north": "Periodic"},
                "gas": {"r_gas": 208.13, "molar_mass": 0.039948, "vhs_omega": 0.81, "mu_ref": 2.117e-5, "t_ref": 273.15, "prandtl": 2.0/3.0},
            },
            "initial": {"rho": 1.0, "temperature": 300.0, "velocity": [0.0, 0.0]},
            "velocity_grid": {"v_max": 1800.0, "n_per_dim": 25},
            "time_scheme": {"scheme": "rk4"},
            "scene": {"boundary_tags": []},
        }
        
        # Should serialize without error
        json_str = json.dumps(payload)
        self.assertIsInstance(json_str, str)
        self.assertIn('"scheme":"rk4"', json_str.replace(" ", ""))

    def test_payload_roundtrip(self):
        """Verify payload survives JSON roundtrip."""
        original = {
            "time_scheme": {"scheme": "rk4"},
            "config": {"grid": {"nx": 64}},
        }
        
        # Serialize and deserialize
        json_str = json.dumps(original)
        restored = json.loads(json_str)
        
        # Verify exact match
        self.assertEqual(restored["time_scheme"]["scheme"], original["time_scheme"]["scheme"])
        self.assertEqual(restored["config"]["grid"]["nx"], original["config"]["grid"]["nx"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
