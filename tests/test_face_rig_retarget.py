import unittest

from supersync.nodes.face_rig_retarget import FaceRigRetarget


class FaceRigRetargetTests(unittest.TestCase):
    def test_reference_rig_preserves_identity_geometry(self):
        node = FaceRigRetarget()
        reference = {
            "left_eye": (0.30, 0.36),
            "right_eye": (0.70, 0.36),
            "nose": (0.50, 0.50),
            "mouth_left": (0.40, 0.65),
            "mouth_right": (0.60, 0.65),
            "upper_lip": (0.50, 0.63),
            "lower_lip": (0.50, 0.67),
            "jaw": (0.50, 0.82),
        }

        rig = node.build_neutral_rig(reference)

        self.assertEqual(rig["identity_mode"], "reference")
        eye_spacing = rig["landmarks"]["right_eye"][0] - rig["landmarks"]["left_eye"][0]
        self.assertAlmostEqual(eye_spacing, 0.4, places=6)

    def test_canonical_mode_is_available_without_reference(self):
        node = FaceRigRetarget()

        rig = node.build_neutral_rig()

        self.assertEqual(rig["identity_mode"], "canonical")
        self.assertIn("jaw", rig["landmarks"])

    def test_apply_deltas_outputs_clamped_landmarks(self):
        node = FaceRigRetarget()
        rig = node.build_neutral_rig()
        frames = [{"frame_index": 0, "jaw_open": 1.0, "lip_open": 1.0, "lip_wide": 1.0, "blink_l": 1.0, "blink_r": 1.0}]

        out = node.apply_deltas(rig, frames)

        self.assertEqual(out[0]["identity_mode"], "canonical")
        self.assertEqual(len(out[0]["landmarks_2d"]), len(rig["landmark_order"]))
        for x, y in out[0]["landmarks_2d"]:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)


if __name__ == "__main__":
    unittest.main()
