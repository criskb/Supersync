import unittest

from supersync.nodes.lip_refine_face import LipRefineFace


class LipRefineFaceTests(unittest.TestCase):
    def test_refinement_aligns_to_base_frame_count(self):
        base_frames = [{"frame_index": i, "jaw_open": 0.5} for i in range(8)]
        audio_frames = [{"frame_index": i, "articulation": 0.3} for i in range(8)]

        out = LipRefineFace().refine(base_frames, audio_frames)

        self.assertEqual(len(out), len(base_frames))
        self.assertEqual([f["frame_index"] for f in out], list(range(8)))

    def test_lip_contact_boosts_on_low_articulation_and_plosive(self):
        base_frames = [
            {"frame_index": 0, "jaw_open": 0.8, "mouth_press_l": 0.1},
            {"frame_index": 1, "jaw_open": 0.8, "mouth_press_l": 0.1},
        ]
        high_contact_audio = [
            {"frame_index": 0, "articulation": 0.1, "plosive": 1.0},
            {"frame_index": 1, "articulation": 0.1, "plosive": 1.0},
        ]
        neutral_audio = [
            {"frame_index": 0, "articulation": 1.0, "plosive": 0.0},
            {"frame_index": 1, "articulation": 1.0, "plosive": 0.0},
        ]

        refiner = LipRefineFace()
        high_contact = refiner.refine(base_frames, high_contact_audio)
        neutral = refiner.refine(base_frames, neutral_audio)

        self.assertGreater(high_contact[0]["lip_contact"], neutral[0]["lip_contact"])
        self.assertLess(high_contact[0]["refined_jaw_open"], neutral[0]["refined_jaw_open"])

    def test_quality_mode_uses_larger_mask_feather_than_realtime(self):
        base_frames = [{"frame_index": 0, "jaw_open": 0.2, "gaze_yaw": 0.5, "gaze_pitch": -0.4}]
        audio_frames = [{"frame_index": 0, "articulation": 0.4, "energy": 0.8}]

        refiner = LipRefineFace()
        rt = refiner.refine(base_frames, audio_frames, mode="realtime")[0]
        quality = refiner.refine(base_frames, audio_frames, mode="quality")[0]

        self.assertEqual(rt["realtime_mode"], 1)
        self.assertEqual(rt["quality_mode"], 0)
        self.assertEqual(quality["realtime_mode"], 0)
        self.assertEqual(quality["quality_mode"], 1)
        self.assertGreater(quality["composite_mask_feather_px"], rt["composite_mask_feather_px"])


if __name__ == "__main__":
    unittest.main()
