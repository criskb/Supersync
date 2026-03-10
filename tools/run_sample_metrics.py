from __future__ import annotations

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supersync.nodes.combine_pose_data import CombinePoseData
from supersync.nodes.eye_motion_synth import EyeMotionSynth
from supersync.nodes.evaluation_utility import NodeEvaluationUtility, RegressionThresholds
from supersync.nodes.lip_refine_face import LipRefineFace


FPS = 60


def _generate_audio_clip(frame_count: int, phase: float) -> list[dict[str, float | int]]:
    frames: list[dict[str, float | int]] = []
    for i in range(frame_count):
        t = i / FPS
        articulation = 0.45 + 0.3 * math.sin(2.0 * math.pi * 1.9 * t + phase)
        articulation += 0.08 * math.sin(2.0 * math.pi * 5.1 * t + 0.3 * phase)
        articulation = max(0.0, min(1.0, articulation))

        energy = 0.4 + 0.35 * (0.5 + 0.5 * math.sin(2.0 * math.pi * 0.6 * t + phase * 0.7))
        pitch_slope = math.sin(2.0 * math.pi * 0.8 * t + 0.5 * phase)
        pause = 0.35 if (i % 150) > 140 else 0.05
        prosody_valley = -0.45 if (i % 170) > 164 else -0.1
        phrase_boundary = 1.0 if pause > 0.25 or prosody_valley < -0.35 else 0.0

        frames.append(
            {
                "frame_index": i,
                "articulation": articulation,
                "energy": energy,
                "pitch_slope": pitch_slope,
                "pause": pause,
                "prosody_valley": prosody_valley,
                "phrase_boundary": phrase_boundary,
                "plosive": max(0.0, min(1.0, 0.3 + 0.5 * math.sin(2 * math.pi * 1.2 * t + phase))),
                "smile": max(0.0, min(1.0, 0.3 + 0.25 * math.sin(2 * math.pi * 0.35 * t + 0.2))),
            }
        )
    return frames


def _build_clip(clip_id: str, phase: float, seed: int) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    audio = _generate_audio_clip(frame_count=420, phase=phase)
    eye = EyeMotionSynth(seed=seed).synthesize(audio)

    lag_frames = 2
    base = []
    for i, frame in enumerate(audio):
        lag_src = audio[max(0, i - lag_frames)]
        articulation = float(lag_src["articulation"])
        jaw_open = max(0.0, min(1.0, 0.1 + 0.85 * articulation))
        eye_frame = eye[i]
        base.append(
            {
                "frame_index": i,
                "jaw_open": jaw_open,
                "mouth_press": max(0.0, 0.45 - 0.4 * articulation),
                "blink_l": eye_frame["blink_l"],
                "blink_r": eye_frame["blink_r"],
                "gaze_yaw": eye_frame["gaze_yaw"],
                "gaze_pitch": eye_frame["gaze_pitch"],
            }
        )

    refined = LipRefineFace().refine(base, audio)

    body = []
    face = []
    facial = []
    for i, eye_frame in enumerate(eye):
        t = i / FPS
        body.append(
            {
                "frame_index": i,
                "confidence": 0.9,
                "head_yaw": 0.15 * math.sin(2 * math.pi * 0.5 * t + phase),
                "head_pitch": 0.12 * math.sin(2 * math.pi * 0.42 * t + phase),
                "head_roll": 0.08 * math.sin(2 * math.pi * 0.36 * t + phase),
                "neck_yaw": 0.1 * math.sin(2 * math.pi * 0.4 * t),
                "neck_pitch": 0.07 * math.sin(2 * math.pi * 0.44 * t),
                "neck_roll": 0.06 * math.sin(2 * math.pi * 0.31 * t),
            }
        )
        face.append({"frame_index": i, "confidence": 0.7})
        facial.append(
            {
                "frame_index": i,
                "expr_confidence": eye_frame["expr_confidence"],
                "jaw_open": refined[i]["refined_jaw_open"],
                "blink_l": eye_frame["blink_l"],
                "blink_r": eye_frame["blink_r"],
                "eyelid_open_l": eye_frame["eyelid_open_l"],
                "eyelid_open_r": eye_frame["eyelid_open_r"],
                "brow_inner_up": eye_frame["brow_inner_up"],
                "brow_outer_up_l": eye_frame["brow_outer_up_l"],
                "brow_outer_up_r": eye_frame["brow_outer_up_r"],
            }
        )

    pose = CombinePoseData().combine(body, face, facial)
    return clip_id, audio, refined, eye, pose


def main() -> None:
    utility = NodeEvaluationUtility(fps=FPS)
    thresholds = RegressionThresholds()

    clips = [
        _build_clip("clip_alpha", phase=0.2, seed=11),
        _build_clip("clip_beta", phase=1.1, seed=27),
        _build_clip("clip_gamma", phase=2.0, seed=51),
    ]
    metrics = [utility.evaluate_clip(*clip) for clip in clips]

    output_dir = Path("tests/artifacts")
    utility.write_artifacts(output_dir, metrics, thresholds)

    thresholds_payload = {
        "fps": FPS,
        **thresholds.__dict__,
    }
    Path("tests/fixtures/node_metric_thresholds.json").write_text(
        json.dumps(thresholds_payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
