from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from .nodes.face_rig_retarget import FaceRigRetarget


class FaceRigBuildNeutralNode:
    CATEGORY = "Supersync/Face"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("neutral_rig_json",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "reference_landmarks_json": ("STRING", {"default": "{}", "multiline": True}),
                "image_width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, reference_landmarks_json: str, image_width: int, image_height: int) -> bool | str:
        if image_width < 0 or image_height < 0:
            return "image_width and image_height must be non-negative"
        try:
            parsed = json.loads(reference_landmarks_json or "{}")
        except JSONDecodeError:
            return "reference_landmarks_json must be valid JSON"
        if not isinstance(parsed, dict):
            return "reference_landmarks_json must decode to an object"
        return True

    def build(self, reference_landmarks_json: str, image_width: int, image_height: int) -> tuple[str]:
        reference = _normalize_landmark_payload(_safe_json_load(reference_landmarks_json, {}), default={})
        image_size = (image_width, image_height) if image_width > 0 and image_height > 0 else None
        payload = FaceRigRetarget().build_neutral_rig(reference_landmarks=reference, image_size=image_size)
        return (json.dumps(payload),)


class FaceRigApplyDeltasNode:
    CATEGORY = "Supersync/Face"
    FUNCTION = "retarget"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("retargeted_frames_json",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "neutral_rig_json": ("STRING", {"default": "{}", "multiline": True}),
                "motion_frames_json": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, neutral_rig_json: str, motion_frames_json: str) -> bool | str:
        try:
            neutral = json.loads(neutral_rig_json or "{}")
        except JSONDecodeError:
            return "neutral_rig_json must be valid JSON"
        try:
            frames = json.loads(motion_frames_json or "[]")
        except JSONDecodeError:
            return "motion_frames_json must be valid JSON"
        if not isinstance(neutral, dict):
            return "neutral_rig_json must decode to an object"
        if not isinstance(frames, list):
            return "motion_frames_json must decode to an array"
        return True

    def retarget(self, neutral_rig_json: str, motion_frames_json: str) -> tuple[str]:
        neutral_raw = _safe_json_load(neutral_rig_json, default={})
        frames_raw = _safe_json_load(motion_frames_json, default=[])

        neutral = neutral_raw if isinstance(neutral_raw, dict) else {}
        frames = [f for f in frames_raw if isinstance(f, dict)] if isinstance(frames_raw, list) else []

        payload = FaceRigRetarget().apply_deltas(neutral_rig=neutral, motion_frames=frames)
        return (json.dumps(payload),)


def _safe_json_load(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw or "")
    except JSONDecodeError:
        return default


def _normalize_landmark_payload(payload: Any, default: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    if not isinstance(payload, dict):
        return dict(default)

    out: dict[str, tuple[float, float]] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                out[key] = (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                continue
    return out


NODE_CLASS_MAPPINGS = {
    "FaceRigBuildNeutral": FaceRigBuildNeutralNode,
    "FaceRigApplyDeltas": FaceRigApplyDeltasNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceRigBuildNeutral": "Face Rig - Build Neutral",
    "FaceRigApplyDeltas": "Face Rig - Apply Deltas",
}
