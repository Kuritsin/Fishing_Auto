from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from collections.abc import Callable

from config import Region, Settings
from .bite import BiteSignalDetector
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
        detector = BiteSignalDetector(
            height,
            self.settings.bite_drop_height_ratio,
            self.settings.bite_velocity_height_ratio,
            self.settings.bite_confirmation_frames,
        )
        last, missing = initial, 0
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
                signal = detector.update(now, current.x, current.y)
            else:
                missing += 1
                signal = detector.update(now, None, None)

            if signal.detected:
                return BiteResult(True, last.x, last.y, signal.reason,
                                  signal.drop, signal.velocity)

            if missing >= 5:
                return BiteResult(False, last.x, last.y, "tracker_lost")
            if stop.wait(delay):
                break
        return BiteResult(False, last.x, last.y,
                          "timeout" if time.monotonic() >= deadline else "interrupted")