from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import threading
import time

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
    def __init__(
        self,
        capture: ScreenCapture,
        settings: Settings,
    ) -> None:
        self.capture = capture
        self.settings = settings
        self.log = logging.getLogger(__name__)

    def wait(
        self,
        initial: vision.Detection,
        templates: list[vision.TemplateAsset],
        client: Region,
        timeout: float,
        stop: threading.Event,
        safe: Callable[[], bool],
    ) -> BiteResult:
        width, height = initial.size
        radius = max(
            36,
            max(width, height) * 2,
        )

        detector = BiteSignalDetector(
            height,
            self.settings.bite_drop_height_ratio,
            self.settings.bite_velocity_height_ratio,
            self.settings.bite_confirmation_frames,
            self.settings.bite_warmup_seconds,
        )

        last = initial
        missing = 0
        lost_since: float | None = None

        if 0 <= initial.template_index < len(templates):
            preferred_templates = [
                templates[initial.template_index]
            ]
        else:
            preferred_templates = templates

        deadline = time.monotonic() + timeout
        delay = 1 / self.settings.tracker_fps

        maximum_step = max(
            15.0,
            height * 1.0,
        )
        maximum_dive_step = max(
            18.0,
            height * 1.35,
        )

        # Keep the click on the last trustworthy surface position.  During a
        # bite the visible template may collapse onto the line, a reflection,
        # or another object far below the bobber.  Following that match made
        # the cursor visibly run down the screen and click the water.
        click_x, click_y = initial.x, initial.y

        while (
            time.monotonic() < deadline
            and not stop.is_set()
            and safe()
        ):
            left = max(
                client.left,
                last.x - radius,
            )
            top = max(
                client.top,
                last.y - radius,
            )
            right = min(
                client.left + client.width,
                last.x + radius,
            )
            bottom = min(
                client.top + client.height,
                last.y + radius,
            )

            local = Region(
                left,
                top,
                right - left,
                bottom - top,
            )

            frame = self.capture.grab(local)

            current = vision.match_templates(
                frame,
                preferred_templates,
                local,
                self.settings.tracker_match_confidence,
                (0.9, 1.0, 1.1),
                masked_threshold=(
                    self.settings.masked_tracker_confidence
                ),
            )

            if (
                not current.found
                and missing >= 2
                and len(templates) > 1
            ):
                current = vision.match_templates(
                    frame,
                    templates,
                    local,
                    self.settings.tracker_match_confidence,
                    (0.9, 1.0, 1.1),
                    masked_threshold=(
                        self.settings.masked_tracker_confidence
                    ),
                )

            now = time.monotonic()

            dx = (
                current.x - last.x
                if current.found
                else 0
            )
            dy = (
                current.y - last.y
                if current.found
                else 0
            )

            ordinary_step = (
                dx * dx + dy * dy
            ) ** 0.5 <= maximum_step

            # A bite is precisely the exceptional fast downward movement we
            # are looking for. The old circular step gate discarded stronger
            # dives before BiteSignalDetector could inspect them. Keep a
            # wider, directional gate while still rejecting sideways jumps.
            plausible_dive = (
                dy >= max(3.0, height * self.settings.bite_drop_height_ratio)
                and dy <= maximum_dive_step
                and abs(dx) <= max(
                    10.0,
                    width * 0.60,
                )
                and dy >= abs(dx) * 0.80
            )

            continuous = (
                current.found
                and (
                    ordinary_step
                    or plausible_dive
                )
            )

            if continuous:
                last = current
                missing = 0
                lost_since = None

                signal = detector.update(
                    now,
                    current.x,
                    current.y,
                )

                # Only an ordinary, non-bite observation is allowed to move
                # the eventual click point.  A directional dive is evidence
                # for the detector, not a new cursor target.
                if ordinary_step and not plausible_dive and not signal.detected:
                    click_x, click_y = current.x, current.y
            else:
                missing += 1

                if lost_since is None:
                    lost_since = now

                signal = detector.update(
                    now,
                    None,
                    None,
                )

            self.log.debug(
                "TRACK x=%s y=%s confidence=%.3f dx=%.1f dy=%.1f "
                "ordinary=%s dive=%s continuous=%s signal=%s drop=%.1f velocity=%.1f",
                current.x if current.found else None,
                current.y if current.found else None,
                current.confidence,
                dx,
                dy,
                ordinary_step,
                plausible_dive,
                continuous,
                signal.reason,
                signal.drop,
                signal.velocity,
            )

            if signal.detected:
                return BiteResult(
                    True,
                    click_x,
                    click_y,
                    signal.reason,
                    signal.drop,
                    signal.velocity,
                )

            if (
                lost_since is not None
                and now - lost_since
                >= self.settings.tracker_lost_seconds
            ):
                return BiteResult(
                    False,
                    last.x,
                    last.y,
                    "tracker_lost",
                )

            if stop.wait(delay):
                break

        return BiteResult(
            False,
            last.x,
            last.y,
            (
                "timeout"
                if time.monotonic() >= deadline
                else "interrupted"
            ),
        )
