import unittest

from supersync.nodes.eye_motion_synth import EyeMotionSynth, EyeMotionSynthConfig


class EyeMotionSynthTests(unittest.TestCase):
    def test_output_aligns_frame_indices(self):
        frames = [{"frame_index": i, "articulation": 0.2} for i in range(120)]
        synth = EyeMotionSynth(seed=7)

        output = synth.synthesize(frames)

        self.assertEqual(len(output), len(frames))
        self.assertEqual([f["frame_index"] for f in output], [f["frame_index"] for f in frames])

    def test_optional_squint(self):
        frames = [{"frame_index": i, "articulation": 0.9} for i in range(20)]
        synth = EyeMotionSynth(EyeMotionSynthConfig(include_squint=True), seed=11)

        output = synth.synthesize(frames)

        self.assertTrue(all("squint" in frame for frame in output))

    def test_phrase_boundary_increases_blink_rate_over_suppressed_speech(self):
        cfg = EyeMotionSynthConfig(mean_blink_interval_s=0.7, blink_interval_jitter_s=0.1)
        speech_frames = [
            {"frame_index": i, "articulation": 0.95, "pause": 0.0, "prosody_valley": 0.0}
            for i in range(360)
        ]
        boundary_frames = [
            {"frame_index": i, "articulation": 0.95, "pause": 0.35, "prosody_valley": -0.4}
            for i in range(360)
        ]

        speech_out = EyeMotionSynth(cfg, seed=23).synthesize(speech_frames)
        boundary_out = EyeMotionSynth(cfg, seed=23).synthesize(boundary_frames)

        speech_blink_energy = sum(f["blink_l"] for f in speech_out)
        boundary_blink_energy = sum(f["blink_l"] for f in boundary_out)
        self.assertGreater(boundary_blink_energy, speech_blink_energy)

    def test_expression_payload_fields_are_present_and_clamped(self):
        frames = [
            {
                "frame_index": i,
                "articulation": 0.4,
                "energy": 1.3,
                "pitch_slope": 1.5,
                "phrase_boundary": 1.2,
                "smile": 0.8,
            }
            for i in range(30)
        ]
        output = EyeMotionSynth(seed=3).synthesize(frames)
        keys = {
            "brow_inner_up",
            "brow_outer_up_l",
            "brow_outer_up_r",
            "eyelid_open_l",
            "eyelid_open_r",
            "cheek_raise_l",
            "cheek_raise_r",
            "nasolabial_tension",
            "expr_confidence",
        }
        self.assertTrue(all(keys.issubset(set(frame.keys())) for frame in output))
        for frame in output:
            for key in keys:
                self.assertGreaterEqual(frame[key], 0.0)
                self.assertLessEqual(frame[key], 1.0)

    def test_smile_increases_cheek_raise_and_reduces_eyelid_open(self):
        neutral = [
            {"frame_index": i, "energy": 0.4, "pitch_slope": 0.0, "phrase_boundary": 0.0, "smile": 0.0}
            for i in range(120)
        ]
        smiling = [
            {"frame_index": i, "energy": 0.4, "pitch_slope": 0.0, "phrase_boundary": 0.0, "smile": 1.0}
            for i in range(120)
        ]

        neutral_out = EyeMotionSynth(seed=5).synthesize(neutral)
        smiling_out = EyeMotionSynth(seed=5).synthesize(smiling)

        neutral_cheek = sum(f["cheek_raise_l"] + f["cheek_raise_r"] for f in neutral_out)
        smiling_cheek = sum(f["cheek_raise_l"] + f["cheek_raise_r"] for f in smiling_out)
        neutral_eyelid = sum(f["eyelid_open_l"] + f["eyelid_open_r"] for f in neutral_out)
        smiling_eyelid = sum(f["eyelid_open_l"] + f["eyelid_open_r"] for f in smiling_out)

        self.assertGreater(smiling_cheek, neutral_cheek)
        self.assertLess(smiling_eyelid, neutral_eyelid)


if __name__ == "__main__":
    unittest.main()
