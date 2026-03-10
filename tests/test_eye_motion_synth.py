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


if __name__ == "__main__":
    unittest.main()
