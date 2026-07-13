import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "blender-addon" / "core" / "timeline_utils.py"

spec = importlib.util.spec_from_file_location("janus_timeline_utils", MODULE_PATH)
timeline_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timeline_utils)


class TimelineUtilsTests(unittest.TestCase):
    def test_frame_index_from_blender_frame(self):
        self.assertEqual(timeline_utils.frame_index_from_blender_frame(5, 2), 3)

    def test_resolve_manifest_frame_path_prefers_data_root(self):
        manifest = {"frames": [{"file": "frame_0000.jvtk"}]}
        resolved = timeline_utils.resolve_manifest_frame_path(manifest, "/tmp/manifest.json", "/tmp/data", 0)
        self.assertEqual(resolved, "/tmp/data/frame_0000.jvtk")


if __name__ == "__main__":
    unittest.main()
