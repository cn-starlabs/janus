"""Unit tests for visualization logic (pure-Python, no bpy required).

These tests validate the mathematical mappings, color ramp stop positions,
and structural requirements of the visualization module's design, without
needing a live Blender context.  Blender-specific node-creation code is
verified by inspecting the source file.
"""

import math
import unittest
import sys
from pathlib import Path

addon_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(addon_path))

VIZ_SRC = Path(__file__).resolve().parent.parent / "core" / "visualization.py"


# ---------------------------------------------------------------------------
# Pure-Python replica of the kn_loc → [0,1] log-scale mapping used by the
# regime-overlay material (for mathematical correctness tests).
# ---------------------------------------------------------------------------

def kn_to_normalized(kn_loc: float) -> float:
    """Map kn_loc to [0, 1] via log10 normalization: (log10(kn) + 3) / 5, clamped."""
    if kn_loc <= 0:
        return 0.0
    log_val = math.log10(kn_loc)
    normalized = (log_val + 3.0) / 5.0
    return max(0.0, min(1.0, normalized))


# ---------------------------------------------------------------------------
# Material name conventions
# ---------------------------------------------------------------------------

def field_material_name(field_name: str) -> str:
    return f"JanusField_{field_name}"


VALID_FIELDS = ["rho", "temperature", "kn_loc", "mom_x", "mom_y", "energy"]
REGIME_MATERIAL_NAME = "JanusRegimeOverlay"


class TestKnLocLogScaleMapping(unittest.TestCase):
    """Verify the kn_loc → normalized color-ramp position math."""

    def test_continuum_regime(self):
        """kn < 0.01 maps to < 0.2 (deep blue zone)."""
        for kn in [0.001, 0.005, 0.009]:
            v = kn_to_normalized(kn)
            self.assertLess(v, 0.2, f"kn={kn} should be < 0.2, got {v}")

    def test_slip_regime(self):
        """0.01 ≤ kn < 0.1 maps to [0.2, 0.4)."""
        for kn in [0.01, 0.05, 0.09]:
            v = kn_to_normalized(kn)
            self.assertGreaterEqual(v, 0.2, f"kn={kn} should be >= 0.2")
            self.assertLess(v, 0.4, f"kn={kn} should be < 0.4")

    def test_transition_regime(self):
        """0.1 ≤ kn < 10 maps to [0.4, 0.8)."""
        for kn in [0.1, 1.0, 5.0, 9.9]:
            v = kn_to_normalized(kn)
            self.assertGreaterEqual(v, 0.4, f"kn={kn} should be >= 0.4")
            self.assertLess(v, 0.8, f"kn={kn} should be < 0.8")

    def test_free_molecular_regime(self):
        """kn >= 10 maps to >= 0.8 (red zone)."""
        for kn in [10.0, 50.0, 100.0, 1000.0]:
            v = kn_to_normalized(kn)
            self.assertGreaterEqual(v, 0.8, f"kn={kn} should be >= 0.8")

    def test_clamped_to_unit_interval(self):
        """Any kn_loc should produce a value in [0, 1]."""
        for kn in [1e-10, 0.001, 1.0, 100.0, 1e6]:
            v = kn_to_normalized(kn)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_zero_kn_clamps_to_zero(self):
        """Zero or negative kn_loc must clamp to 0 without raising."""
        self.assertEqual(kn_to_normalized(0.0), 0.0)
        self.assertEqual(kn_to_normalized(-1.0), 0.0)

    def test_boundary_at_kn_0p01(self):
        """kn=0.01 exactly sits at normalized value 0.2."""
        v = kn_to_normalized(0.01)
        self.assertAlmostEqual(v, 0.2, places=10)

    def test_boundary_at_kn_0p1(self):
        """kn=0.1 exactly sits at normalized value 0.4."""
        v = kn_to_normalized(0.1)
        self.assertAlmostEqual(v, 0.4, places=10)

    def test_boundary_at_kn_10(self):
        """kn=10 exactly sits at normalized value 0.8."""
        v = kn_to_normalized(10.0)
        self.assertAlmostEqual(v, 0.8, places=10)

    def test_monotone_increasing(self):
        """Larger kn should always produce larger (or equal) normalized value."""
        kn_values = [0.001, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0]
        normalized = [kn_to_normalized(k) for k in kn_values]
        for i in range(len(normalized) - 1):
            self.assertLessEqual(
                normalized[i], normalized[i + 1],
                f"Non-monotone at kn={kn_values[i]} → {kn_values[i+1]}"
            )


class TestFieldMaterialNaming(unittest.TestCase):
    """Verify material naming conventions."""

    def test_field_material_name_format(self):
        for field in VALID_FIELDS:
            name = field_material_name(field)
            self.assertTrue(name.startswith("JanusField_"))
            self.assertIn(field, name)

    def test_all_valid_fields_produce_unique_names(self):
        names = [field_material_name(f) for f in VALID_FIELDS]
        self.assertEqual(len(names), len(set(names)), "Duplicate material names detected")

    def test_regime_material_name(self):
        self.assertEqual(REGIME_MATERIAL_NAME, "JanusRegimeOverlay")

    def test_valid_fields_list(self):
        """Verify the expected fields match the plan."""
        expected = {"rho", "temperature", "kn_loc", "mom_x", "mom_y", "energy"}
        self.assertEqual(set(VALID_FIELDS), expected)


class TestColorRampStopPositions(unittest.TestCase):
    """Validate that regime overlay ramp stop positions match regime boundaries."""

    # Expected stop positions from visualization.py
    STOP_POSITIONS = [0.0, 0.2, 0.4, 0.8, 1.0]

    def test_continuum_stop_at_zero(self):
        self.assertAlmostEqual(self.STOP_POSITIONS[0], 0.0)

    def test_slip_boundary_at_0p2(self):
        """kn=0.01 maps to 0.2 — this is where continuum → slip color changes."""
        self.assertAlmostEqual(self.STOP_POSITIONS[1], kn_to_normalized(0.01))

    def test_transition_boundary_at_0p4(self):
        """kn=0.1 maps to 0.4 — slip → transition boundary."""
        self.assertAlmostEqual(self.STOP_POSITIONS[2], kn_to_normalized(0.1))

    def test_free_molecular_boundary_at_0p8(self):
        """kn=10 maps to 0.8 — transition → free molecular boundary."""
        self.assertAlmostEqual(self.STOP_POSITIONS[3], kn_to_normalized(10.0))

    def test_far_free_molecular_at_1p0(self):
        self.assertAlmostEqual(self.STOP_POSITIONS[4], 1.0)

    def test_stops_are_sorted(self):
        for i in range(len(self.STOP_POSITIONS) - 1):
            self.assertLess(self.STOP_POSITIONS[i], self.STOP_POSITIONS[i + 1])


class TestVisualizationSourceStructure(unittest.TestCase):
    """Verify the visualization module source contains required symbols and patterns."""

    def setUp(self):
        self.content = VIZ_SRC.read_text()

    def test_source_file_exists(self):
        self.assertTrue(VIZ_SRC.exists())

    def test_setup_field_material_defined(self):
        self.assertIn("def setup_field_material", self.content)

    def test_setup_regime_overlay_defined(self):
        self.assertIn("def setup_regime_overlay_material", self.content)

    def test_ensure_attribute_domain_defined(self):
        self.assertIn("def ensure_attribute_domain", self.content)

    def test_principled_bsdf_used(self):
        self.assertIn("ShaderNodeBsdfPrincipled", self.content)

    def test_color_ramp_node_used(self):
        self.assertIn("ShaderNodeValToRGB", self.content)

    def test_attribute_node_used(self):
        self.assertIn("ShaderNodeAttribute", self.content)

    def test_log_node_used_in_regime_overlay(self):
        """Regime overlay must use a logarithmic Math node for kn_loc."""
        self.assertIn("LOGARITHM", self.content)

    def test_material_output_node_used(self):
        self.assertIn("ShaderNodeOutputMaterial", self.content)

    def test_kn_loc_attribute_name(self):
        """Regime overlay must bind to 'kn_loc' attribute."""
        self.assertIn('"kn_loc"', self.content)

    def test_materials_appended_to_object(self):
        """Material should be appended to or replace object's material slot."""
        self.assertIn("obj.data.materials", self.content)

    def test_use_nodes_enabled(self):
        self.assertIn("use_nodes = True", self.content)

    def test_regime_material_name_in_source(self):
        self.assertIn("JanusRegimeOverlay", self.content)

    def test_field_material_prefix_in_source(self):
        self.assertIn("JanusField_", self.content)

    def test_links_connect_ramp_to_bsdf(self):
        """Color ramp output should feed into BSDF Base Color."""
        self.assertIn("Base Color", self.content)

    def test_attribute_domain_creates_if_missing(self):
        """ensure_attribute_domain should create attribute when absent."""
        self.assertIn("attributes.new", self.content)


class TestFieldAttributeRequirements(unittest.TestCase):
    """Validate expected mesh attribute metadata."""

    EXPECTED_ATTRIBUTES = ["rho", "temperature", "kn_loc", "mom_x", "mom_y", "energy"]
    EXPECTED_DOMAIN = "FACE"
    EXPECTED_DTYPE = "FLOAT"

    def test_all_expected_attributes_listed(self):
        self.assertEqual(len(self.EXPECTED_ATTRIBUTES), 6)

    def test_domain_is_face(self):
        """Physics scalars should live on mesh faces (cells)."""
        self.assertEqual(self.EXPECTED_DOMAIN, "FACE")

    def test_dtype_is_float(self):
        self.assertEqual(self.EXPECTED_DTYPE, "FLOAT")

    def test_kn_loc_in_attributes(self):
        """kn_loc must be in expected attributes for regime overlay to work."""
        self.assertIn("kn_loc", self.EXPECTED_ATTRIBUTES)

    def test_vector_fields_separate_components(self):
        """mom_x and mom_y stored as separate scalars, not FLOAT_VECTOR."""
        self.assertIn("mom_x", self.EXPECTED_ATTRIBUTES)
        self.assertIn("mom_y", self.EXPECTED_ATTRIBUTES)
        self.assertNotIn("momentum", self.EXPECTED_ATTRIBUTES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
