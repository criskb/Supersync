from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any


@dataclass
class EyeMotionSynthConfig:
    """Configuration for EyeMotionSynth probabilistic state machine."""

    target_fps: int = 60
    mean_blink_interval_s: float = 4.2
    blink_interval_jitter_s: float = 1.6
    blink_duration_s: float = 0.16
    blink_peak_hold_s: float = 0.03
    articulation_suppression_threshold: float = 0.65
    articulation_suppression_factor: float = 0.2
    phrase_boundary_boost: float = 2.5
    pause_boundary_threshold: float = 0.25
    prosody_valley_threshold: float = -0.35
    micro_saccade_interval_s: tuple[float, float] = (0.08, 0.28)
    micro_saccade_sigma: float = 0.02
    gaze_drift_sigma: float = 0.004
    gaze_clamp: tuple[float, float] = (-1.0, 1.0)
    include_squint: bool = False


class EyeMotionSynth:
    """
    Synthesizes per-frame eye motion.

    Output schema per frame:
      - frame_index
      - blink_l, blink_r
      - gaze_yaw, gaze_pitch
      - squint (optional)

    Frames are emitted one-to-one against the provided Audio2FaceSync frames so
    frame indices align naturally for CombinePoseData merging.
    """

    def __init__(self, config: EyeMotionSynthConfig | None = None, seed: int | None = None):
        self.config = config or EyeMotionSynthConfig()
        self._rng = random.Random(seed)

    def synthesize(self, audio2face_frames: list[dict[str, Any]]) -> list[dict[str, float | int]]:
        if not audio2face_frames:
            return []

        cfg = self.config
        blink_close_frames = max(1, int(round((cfg.blink_duration_s - cfg.blink_peak_hold_s) * 0.5 * cfg.target_fps)))
        blink_hold_frames = max(0, int(round(cfg.blink_peak_hold_s * cfg.target_fps)))
        blink_open_frames = blink_close_frames

        gaze_yaw = 0.0
        gaze_pitch = 0.0
        microsaccade_countdown = self._sample_microsaccade_countdown()

        next_blink_in_frames = self._sample_next_blink_interval()
        blink_phase = "idle"
        phase_frame = 0

        output: list[dict[str, float | int]] = []

        for i, src in enumerate(audio2face_frames):
            frame_index = int(src.get("frame_index", i))
            articulation = float(src.get("articulation", 0.0))
            pause = float(src.get("pause", 0.0))
            prosody_valley = float(src.get("prosody_valley", 0.0))

            # Baseline blink trigger from randomized inter-blink timing.
            if blink_phase == "idle":
                next_blink_in_frames -= 1
                if next_blink_in_frames <= 0:
                    trigger_prob = self._blink_trigger_probability(
                        articulation=articulation,
                        pause=pause,
                        prosody_valley=prosody_valley,
                    )
                    if self._rng.random() < trigger_prob:
                        blink_phase = "closing"
                        phase_frame = 0
                    else:
                        # Retry soon if blink got suppressed.
                        next_blink_in_frames = max(1, int(0.2 * cfg.target_fps))

            blink_value = 0.0
            if blink_phase == "closing":
                phase_frame += 1
                blink_value = min(1.0, phase_frame / blink_close_frames)
                if phase_frame >= blink_close_frames:
                    blink_phase = "hold" if blink_hold_frames > 0 else "opening"
                    phase_frame = 0
                    blink_value = 1.0
            elif blink_phase == "hold":
                phase_frame += 1
                blink_value = 1.0
                if phase_frame >= blink_hold_frames:
                    blink_phase = "opening"
                    phase_frame = 0
            elif blink_phase == "opening":
                phase_frame += 1
                blink_value = max(0.0, 1.0 - (phase_frame / blink_open_frames))
                if phase_frame >= blink_open_frames:
                    blink_phase = "idle"
                    phase_frame = 0
                    blink_value = 0.0
                    next_blink_in_frames = self._sample_next_blink_interval()

            # Micro-saccades and fixation drift.
            microsaccade_countdown -= 1
            if microsaccade_countdown <= 0:
                gaze_yaw += self._rng.gauss(0.0, cfg.micro_saccade_sigma)
                gaze_pitch += self._rng.gauss(0.0, cfg.micro_saccade_sigma)
                microsaccade_countdown = self._sample_microsaccade_countdown()
            else:
                gaze_yaw += self._rng.gauss(0.0, cfg.gaze_drift_sigma)
                gaze_pitch += self._rng.gauss(0.0, cfg.gaze_drift_sigma)

            gaze_yaw = max(cfg.gaze_clamp[0], min(cfg.gaze_clamp[1], gaze_yaw))
            gaze_pitch = max(cfg.gaze_clamp[0], min(cfg.gaze_clamp[1], gaze_pitch))

            out: dict[str, float | int] = {
                "frame_index": frame_index,
                "blink_l": blink_value,
                "blink_r": blink_value,
                "gaze_yaw": gaze_yaw,
                "gaze_pitch": gaze_pitch,
            }
            if cfg.include_squint:
                squint = min(1.0, max(0.0, 0.08 + 0.2 * blink_value + 0.12 * max(0.0, articulation - 0.5)))
                out["squint"] = squint

            output.append(out)

        return output

    def _sample_next_blink_interval(self) -> int:
        cfg = self.config
        base = cfg.mean_blink_interval_s + self._rng.uniform(-cfg.blink_interval_jitter_s, cfg.blink_interval_jitter_s)
        base = max(0.5, base)
        return int(round(base * cfg.target_fps))

    def _sample_microsaccade_countdown(self) -> int:
        low, high = self.config.micro_saccade_interval_s
        return max(1, int(round(self._rng.uniform(low, high) * self.config.target_fps)))

    def _blink_trigger_probability(self, articulation: float, pause: float, prosody_valley: float) -> float:
        cfg = self.config
        # Start from a unit hazard at due-time and then modulate.
        prob = 1.0

        if articulation >= cfg.articulation_suppression_threshold:
            prob *= cfg.articulation_suppression_factor

        at_phrase_boundary = pause >= cfg.pause_boundary_threshold or prosody_valley <= cfg.prosody_valley_threshold
        if at_phrase_boundary:
            prob *= cfg.phrase_boundary_boost

        return min(1.0, max(0.0, prob))
