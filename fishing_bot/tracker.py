from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from collections.abc import Callable

import numpy as np

from config import Region, Settings
from .capture import ScreenCapture
from .vision import Detection, match


@dataclass(frozen=True)
class BiteResult:
    detected: bool
    x: int
    y: int
    reason: str
    drop: float = 0.0
    velocity: float = 0.0


class BobberTracker:
    def __init__(self, capture: ScreenCapture, settings: Settings) -> None:
        self.capture, self.settings = capture, settings

    def wait(self, initial: Detection, template: np.ndarray, client: Region, timeout: float,
             stop: threading.Event, safe: Callable[[], bool]) -> BiteResult:
        width, height = initial.size
        radius = max(36, max(width, height) * 2)
        history: deque[tuple[float, int, int]] = deque(maxlen=12)
        last, missing, confirmations = initial, 0, 0
        deadline = time.monotonic() + timeout
        delay = 1 / self.settings.tracker_fps

        while time.monotonic() < deadline and not stop.is_set() and safe():
            left, top = max(client.left, last.x - radius), max(client.top, last.y - radius)
            right = min(client.left + client.width, last.x + radius)
            bottom = min(client.top + client.height, last.y + radius)
            local = Region(left, top, right - left, bottom - top)
            current = match(self.capture.grab(local), template, local,
                            self.settings.match_confidence, (0.9, 1.0, 1.1))
            now = time.monotonic()
            if current.found:
                last, missing = current, 0
                history.append((now, current.x, current.y))
            else:
                missing += 1

            if len(history) >= 6:
                recent = list(history)
                baseline_points = recent[:-2]
                baseline_y = float(np.median([point[2] for point in baseline_points]))
                baseline_x = float(np.median([point[1] for point in baseline_points]))
                drop = last.y - baseline_y
                elapsed = max(0.001, recent[-1][0] - recent[-3][0])
                velocity = (recent[-1][2] - recent[-3][2]) / elapsed
                horizontal = abs(last.x - baseline_x)
                min_drop = max(3.0, height * self.settings.bite_drop_height_ratio)
                min_velocity = max(18.0, height * self.settings.bite_velocity_height_ratio)
                downward = drop >= min_drop and velocity >= min_velocity and drop > horizontal * 0.55
                confirmations = confirmations + 1 if downward else 0
                if confirmations >= self.settings.bite_confirmation_frames:
                    return BiteResult(True, last.x, last.y, "rapid_downward_motion", drop, velocity)
                if missing >= 2 and drop >= min_drop:
                    return BiteResult(True, last.x, last.y, "submerged_after_drop", drop, velocity)

            if missing >= 5:
                return BiteResult(False, last.x, last.y, "tracker_lost")
            if stop.wait(delay):
                break
        return BiteResult(False, last.x, last.y,
                          "timeout" if time.monotonic() >= deadline else "interrupted")
