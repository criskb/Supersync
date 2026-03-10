import json
import unittest
from pathlib import Path

from supersync.nodes.evaluation_utility import NodeEvaluationUtility, RegressionThresholds
from tools.run_sample_metrics import _build_clip


class NodeMetricRegressionTests(unittest.TestCase):
    def test_sample_clip_metrics_stay_within_thresholds(self):
        thresholds_raw = json.loads(Path("tests/fixtures/node_metric_thresholds.json").read_text(encoding="utf-8"))
        thresholds = RegressionThresholds(
            max_abs_lip_audio_sync_lag_frames=int(thresholds_raw["max_abs_lip_audio_sync_lag_frames"]),
            min_lip_audio_sync_correlation=float(thresholds_raw["min_lip_audio_sync_correlation"]),
            blink_rate_per_minute_range=tuple(thresholds_raw["blink_rate_per_minute_range"]),
            max_landmark_velocity_outliers=int(thresholds_raw["max_landmark_velocity_outliers"]),
            max_landmark_acceleration_outliers=int(thresholds_raw["max_landmark_acceleration_outliers"]),
            max_pose_jitter_score=float(thresholds_raw["max_pose_jitter_score"]),
        )

        utility = NodeEvaluationUtility(fps=int(thresholds_raw["fps"]))
        clips = [
            _build_clip("clip_alpha", phase=0.2, seed=11),
            _build_clip("clip_beta", phase=1.1, seed=27),
            _build_clip("clip_gamma", phase=2.0, seed=51),
        ]
        metrics = [utility.evaluate_clip(*clip) for clip in clips]

        failures = utility.check_thresholds(metrics, thresholds)
        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
