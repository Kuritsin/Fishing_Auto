from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from collections.abc import Callable

from config import Region, Settings
from . import vision
from .bite import BiteSignalDetector
from .capture import ScreenCapture


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

    def wait(self, initial: vision.Detection, templates: list[vision.TemplateAsset],
             client: Region, timeout: float,
             stop: threading.Event, safe: Callable[[], bool]) -> BiteResult:
        width, height = initial.size
        radius = max(36, max(width, height) * 2)
        detector = BiteSignalDetector(
            height,
            self.settings.bite_drop_height_ratio,
            self.settings.bite_velocity_height_ratio,
            self.settings.bite_confirmation_frames,
            self.settings.bite_warmup_seconds,
        )
        last, missing = initial, 0
        lost_since: float | None = None
        if 0 <= initial.template_index < len(templates):
            preferred_templates = [templates[initial.template_index]]
        else:
            preferred_templates = templates
        deadline = time.monotonic() + timeout
        delay = 1 / self.settings.tracker_fps

        maximum_step = max(15.0, height * 1.0)
        while time.monotonic() < deadline and not stop.is_set() and safe():
            left, top = max(client.left, last.x - radius), max(client.top, last.y - radius)
            right = min(client.left + client.width, last.x + radius)
            bottom = min(client.top + client.height, last.y + radius)
            local = Region(left, top, right - left, bottom - top)
            frame = self.capture.grab(local)
            current = vision.match_templates(
                frame, preferred_templates, local, self.settings.tracker_match_confidence,
                (0.9, 1.0, 1.1),
                masked_threshold=self.settings.masked_tracker_confidence,
            )
            if not current.found and missing >= 2 and len(templates) > 1:
                current = vision.match_templates(
                    frame, templates, local, self.settings.tracker_match_confidence,
                    (0.9, 1.0, 1.1),
                    masked_threshold=self.settings.masked_tracker_confidence,
                )
            now = time.monotonic()
            continuous = current.found and (
                (current.x - last.x) ** 2 + (current.y - last.y) ** 2
            ) ** 0.5 <= maximum_step
            if continuous:
                last, missing = current, 0
                lost_since = None
                signal = detector.update(now, current.x, current.y)
            else:
                missing += 1
                if lost_since is None:
                    lost_since = now
                signal = detector.update(now, None, None)

            if signal.detected:
                return BiteResult(True, last.x, last.y, signal.reason,
                                  signal.drop, signal.velocity)

            if lost_since is not None and now - lost_since >= self.settings.tracker_lost_seconds:
                return BiteResult(False, last.x, last.y, "tracker_lost")
            if stop.wait(delay):
                break
        return BiteResult(False, last.x, last.y,
                          "timeout" if time.monotonic() >= deadline else "interrupted")