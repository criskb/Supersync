from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FaceRigRetargetConfig:
    """Configuration for neutral rig selection and motion retargeting."""

    landmark_order: tuple[str, ...] = (
        "left_eye",
        "right_eye",
        "nose",
        "mouth_left",
        "mouth_right",
        "upper_lip",
        "lower_lip",
        "jaw",
    )
    canonical_landmarks: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "left_eye": (0.38, 0.36),
            "right_eye": (0.62, 0.36),
            "nose": (0.50, 0.48),
            "mouth_left": (0.43, 0.62),
            "mouth_right": (0.57, 0.62),
            "upper_lip": (0.50, 0.60),
            "lower_lip": (0.50, 0.64),
            "jaw": (0.50, 0.78),
        }
    )
    jaw_open_scale: float = 0.11
    lip_open_scale: float = 0.06
    lip_wide_scale: float = 0.05
    blink_scale: float = 0.02


class FaceRigRetarget:
    """Apply audio-driven facial motion deltas to a neutral face rig.

    The node intentionally separates identity geometry (from a reference frame)
    from motion controls (jaw/lips/blinks). If no reference rig is supplied, a
    canonical template is used for fast prototyping.
    """

    def __init__(self, config: FaceRigRetargetConfig | None = None):
        self.config = config or FaceRigRetargetConfig()

    def build_neutral_rig(self, reference_landmarks: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
        base = reference_landmarks or self.config.canonical_landmarks
        points = {name: tuple(base[name]) for name in self.config.landmark_order if name in base}
        mode = "reference" if reference_landmarks else "canonical"
        return {
            "identity_mode": mode,
            "landmarks": points,
            "landmark_order": list(self.config.landmark_order),
        }

    def apply_deltas(self, neutral_rig: dict[str, Any], motion_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        landmarks = neutral_rig.get("landmarks", {})
        order = list(neutral_rig.get("landmark_order", self.config.landmark_order))
        identity_mode = str(neutral_rig.get("identity_mode", "canonical"))

        out: list[dict[str, Any]] = []
        for i, frame in enumerate(motion_frames):
            jaw_open = self._clamp(float(frame.get("jaw_open", 0.0)), 0.0, 1.0)
            lip_open = self._clamp(float(frame.get("lip_open", 0.0)), 0.0, 1.0)
            lip_wide = self._clamp(float(frame.get("lip_wide", 0.0)), -1.0, 1.0)
            blink = 0.5 * self._clamp(float(frame.get("blink_l", 0.0)), 0.0, 1.0) + 0.5 * self._clamp(float(frame.get("blink_r", 0.0)), 0.0, 1.0)

            posed = {name: list(landmarks.get(name, (0.5, 0.5))) for name in order}

            # Mouth and jaw retargeting.
            posed["jaw"][1] += jaw_open * self.config.jaw_open_scale
            posed["upper_lip"][1] -= lip_open * self.config.lip_open_scale * 0.5
            posed["lower_lip"][1] += lip_open * self.config.lip_open_scale
            posed["mouth_left"][0] -= lip_wide * self.config.lip_wide_scale
            posed["mouth_right"][0] += lip_wide * self.config.lip_wide_scale

            # Blink closes eyelids by moving eyes slightly downward.
            posed["left_eye"][1] += blink * self.config.blink_scale
            posed["right_eye"][1] += blink * self.config.blink_scale

            out.append(
                {
                    "frame_index": int(frame.get("frame_index", i)),
                    "identity_mode": identity_mode,
                    "landmarks": {k: (self._clamp(v[0], 0.0, 1.0), self._clamp(v[1], 0.0, 1.0)) for k, v in posed.items()},
                    "landmarks_2d": [
                        [
                            self._clamp(posed[name][0], 0.0, 1.0),
                            self._clamp(posed[name][1], 0.0, 1.0),
                        ]
                        for name in order
                    ],
                }
            )

        return out

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
