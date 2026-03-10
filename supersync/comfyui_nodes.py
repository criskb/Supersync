from __future__ import annotations

import json
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
                "reference_landmarks_json": ("STRING", {"default": "{}"}),
                "image_width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            }
        }

    def build(self, reference_landmarks_json: str, image_width: int, image_height: int) -> tuple[str]:
        ref = json.loads(reference_landmarks_json or "{}")
        image_size = (image_width, image_height) if image_width > 0 and image_height > 0 else None
        payload = FaceRigRetarget().build_neutral_rig(reference_landmarks=ref, image_size=image_size)
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
                "neutral_rig_json": ("STRING", {"default": "{}"}),
                "motion_frames_json": ("STRING", {"default": "[]"}),
            }
        }

    def retarget(self, neutral_rig_json: str, motion_frames_json: str) -> tuple[str]:
        neutral = json.loads(neutral_rig_json or "{}")
        frames = json.loads(motion_frames_json or "[]")
        payload = FaceRigRetarget().apply_deltas(neutral_rig=neutral, motion_frames=frames)
        return (json.dumps(payload),)


NODE_CLASS_MAPPINGS = {
    "FaceRigBuildNeutral": FaceRigBuildNeutralNode,
    "FaceRigApplyDeltas": FaceRigApplyDeltasNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceRigBuildNeutral": "Face Rig - Build Neutral",
    "FaceRigApplyDeltas": "Face Rig - Apply Deltas",
}
