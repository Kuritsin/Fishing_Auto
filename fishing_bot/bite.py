from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BiteSignal:
    detected: bool
    reason: str = "tracking"
    drop: float = 0.0
    velocity: float = 0.0


class BiteSignalDetector:
    """Classify a short downward dive independently of colour and resolution."""

    def __init__(
        self,
        bobber_height: int,
        drop_ratio: float = 0.12,
        velocity_ratio: float = 0.55,
        confirmations: int = 2,
        warmup_seconds: float = 1.0,
    ) -> None:
        self.height = max(1, bobber_height)
        self.drop_ratio = drop_ratio
        self.velocity_ratio = velocity_ratio
        self.required_confirmations = confirmations
        self.warmup_seconds = warmup_seconds
        self.history: deque[tuple[float, int, int]] = deque(maxlen=24)
        self.confirmations = 0
        self.last_drop = self.last_velocity = 0.0
        self.pending_dive = False
        self.started_at: float | None = None

    def update(
        self,
        timestamp: float,
        x: int | None,
        y: int | None,
    ) -> BiteSignal:
        if x is None or y is None:
            warmed_up = (
                self.started_at is not None
                and timestamp - self.started_at >= self.warmup_seconds
            )

            if (
                warmed_up
                and self.pending_dive
                and self.last_drop >= self._minimum_drop()
            ):
                return BiteSignal(
                    True,
                    "submerged_after_drop",
                    self.last_drop,
                    self.last_velocity,
                )

            self.confirmations = 0
            return BiteSignal(False, "temporarily_lost")

        if self.started_at is None:
            self.started_at = timestamp

        self.history.append((timestamp, x, y))

        if len(self.history) < 8:
            return BiteSignal(False, "building_baseline")

        if timestamp - self.started_at < self.warmup_seconds:
            return BiteSignal(False, "building_baseline")

        points = list(self.history)

        # Keep the newest three frames out of the baseline so that a fast dive
        # cannot drag its own reference level down.
        baseline_points = points[:-3]

        baseline_y = float(
            np.median([point[2] for point in baseline_points])
        )
        baseline_x = float(
            np.median([point[1] for point in baseline_points])
        )

        baseline_deviations = np.abs(
            np.asarray(
                [point[2] for point in baseline_points],
                dtype=float,
            )
            - baseline_y
        )

        # Water and matching noise differ greatly between locations.
        # MAD keeps the threshold above the movement observed during this cast
        # without allowing one old outlier to dominate it.
        jitter = float(np.median(baseline_deviations))

        self.last_drop = y - baseline_y

        elapsed = max(
            0.001,
            points[-1][0] - points[-3][0],
        )
        self.last_velocity = (
            points[-1][2] - points[-3][2]
        ) / elapsed

        horizontal = abs(x - baseline_x)

        minimum_drop = max(
            self._minimum_drop(),
            jitter * 4.0 + 2.0,
        )

        downward = (
            self.last_drop >= minimum_drop
            and self.last_velocity >= self._minimum_velocity()
            and self.last_drop > horizontal * 0.55
        )

        self.pending_dive = downward
        self.confirmations = (
            self.confirmations + 1
            if downward
            else 0
        )

        # A real bite can be visible for only one captured frame before the
        # bobber disappears. Accept one frame only when it is substantially
        # stronger than the ordinary two-frame condition.
        strong_dive = (
            self.last_drop
            >= max(
                minimum_drop * 1.8,
                self.height * 0.32,
            )
            and self.last_velocity
            >= self._minimum_velocity() * 1.6
            and self.last_drop > horizontal * 0.8
        )

        if self.confirmations >= self.required_confirmations:
            return BiteSignal(
                True,
                "rapid_downward_motion",
                self.last_drop,
                self.last_velocity,
            )

        if strong_dive:
            return BiteSignal(
                True,
                "strong_downward_motion",
                self.last_drop,
                self.last_velocity,
            )

        return BiteSignal(
            False,
            "tracking",
            self.last_drop,
            self.last_velocity,
        )

    def _minimum_drop(self) -> float:
        return max(
            3.0,
            self.height * self.drop_ratio,
        )

    def _minimum_velocity(self) -> float:
        return max(
            18.0,
            self.height * self.velocity_ratio,
        )