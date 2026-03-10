import unittest

from supersync.nodes.combine_pose_data import CombinePoseData, CombinePoseDataConfig


class CombinePoseDataTests(unittest.TestCase):
    def test_merges_priority_with_bounded_head_offsets(self):
        combiner = CombinePoseData()
        body = [{"frame_index": 0, "head_yaw": 0.2, "confidence": 0.9}]
        face = [{"frame_index": 0, "head_yaw_offset": 1.0, "confidence": 0.8}]
        facial = [{"frame_index": 0, "jaw_open": 0.4, "expr_confidence": 0.8}]

        out = combiner.combine(body, face, facial)

        # Offset is bounded by config head_yaw bound (0.35)
        self.assertAlmostEqual(out[0]["head_yaw"], 0.55, places=6)
        self.assertAlmostEqual(out[0]["jaw_open"], 0.4, places=6)

    def test_temporal_filters_and_max_delta_limit_frame_pops(self):
        cfg = CombinePoseDataConfig(max_delta_per_frame=0.1, eyes_event_threshold=0.1)
        combiner = CombinePoseData(cfg)
        body = [{"frame_index": 0}, {"frame_index": 1}]
        face = [{"frame_index": 0}, {"frame_index": 1}]
        facial = [
            {"frame_index": 0, "blink_l": 0.0, "jaw_open": 0.0, "expr_confidence": 1.0},
            {"frame_index": 1, "blink_l": 1.0, "jaw_open": 1.0, "expr_confidence": 1.0},
        ]

        out = combiner.combine(body, face, facial)

        self.assertAlmostEqual(out[1]["blink_l"], 0.1, places=6)
        self.assertAlmostEqual(out[1]["jaw_open"], 0.1, places=6)

    def test_emits_debug_diagnostics_on_divergence(self):
        combiner = CombinePoseData(CombinePoseDataConfig(divergence_threshold=0.2))
        body = [{"frame_index": 0, "head_pitch": 0.0, "confidence": 1.0}]
        face = [{"frame_index": 0, "head_pitch": 1.0, "head_pitch_offset": 0.3, "confidence": 1.0}]
        facial = [{"frame_index": 0}]

        combiner.combine(body, face, facial)

        self.assertTrue(any(d["type"] == "body_face_divergence" for d in combiner.last_diagnostics))
        self.assertTrue(any(d["type"] == "head_offset_divergence" for d in combiner.last_diagnostics))


if __name__ == "__main__":
    unittest.main()
