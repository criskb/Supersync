import json
import unittest

from supersync.comfyui_nodes import FaceRigApplyDeltasNode, FaceRigBuildNeutralNode


class ComfyUiFaceRigNodeTests(unittest.TestCase):
    def test_build_neutral_node_outputs_json_payload(self):
        node = FaceRigBuildNeutralNode()
        result = node.build('{"left_eye": [192, 216], "right_eye": [448, 216]}', 640, 1080)
        payload = json.loads(result[0])

        self.assertEqual(payload["identity_mode"], "reference")
        self.assertAlmostEqual(payload["landmarks"]["left_eye"][0], 0.3, places=6)

    def test_apply_deltas_node_outputs_frame_payload(self):
        build = FaceRigBuildNeutralNode()
        neutral = build.build("{}", 0, 0)[0]
        apply_node = FaceRigApplyDeltasNode()

        frames_json = json.dumps([
            {"frame_index": 0, "jaw_open": 0.0},
            {"frame_index": 1, "jaw_open": 1.0, "smoothing": 0.8},
        ])
        result = apply_node.retarget(neutral, frames_json)
        frames = json.loads(result[0])

        self.assertEqual(len(frames), 2)
        self.assertIn("landmarks_2d", frames[0])
        self.assertLess(frames[1]["motion"]["jaw_open"], 0.5)


if __name__ == "__main__":
    unittest.main()
