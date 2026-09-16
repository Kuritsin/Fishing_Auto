from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable


def cursor_path(start: tuple[int, int], end: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    """Build a short eased curve which always finishes on the exact target."""
    start_x, start_y = start
    delta_x, delta_y = end[0] - start_x, end[1] - start_y
    distance = math.hypot(delta_x, delta_y)
    if distance == 0:
        return [end]
    bend = min(18.0, distance * 0.035)
    perpendicular_x, perpendicular_y = -delta_y / distance, delta_x / distance
    direction = -1.0 if (start_x + start_y + end[0] + end[1]) % 2 else 1.0
    control_1 = (start_x + delta_x * 0.28 + perpendicular_x * bend * direction,
                 start_y + delta_y * 0.28 + perpendicular_y * bend * direction)
    control_2 = (start_x + delta_x * 0.74 - perpendicular_x * bend * direction * 0.35,
                 start_y + delta_y * 0.74 - perpendicular_y * bend * direction * 0.35)
    points = []
    for step in range(1, steps + 1):
        raw = step / steps
        t = raw * raw * (3.0 - 2.0 * raw)
        inverse = 1.0 - t
        x = (inverse ** 3 * start_x + 3 * inverse ** 2 * t * control_1[0] +
             3 * inverse * t ** 2 * control_2[0] + t ** 3 * end[0])
        y = (inverse ** 3 * start_y + 3 * inverse ** 2 * t * control_1[1] +
             3 * inverse * t ** 2 * control_2[1] + t ** 3 * end[1])
        points.append((round(x), round(y)))
    points[-1] = end
    return points


class SafeInput:
    def __init__(self, allowed: Callable[[], bool], dry_run: bool) -> None:
        self.allowed, self.dry_run = allowed, dry_run
        self.keyboard = self.mouse = None
        self.log = logging.getLogger(__name__)

    def _controllers(self):
        if self.keyboard is None:
            from pynput import keyboard, mouse
            self.keyboard, self.mouse = keyboard.Controller(), mouse.Controller()
        return self.keyboard, self.mouse

    def press(self, key: str) -> bool:
        if not self.allowed():
            return False
        if self.dry_run:
            self.log.info("DRY RUN: press %s", key)
            return True
        keyboard, _ = self._controllers()
        keyboard.press(key); time.sleep(0.05); keyboard.release(key)
        return True

    def move(self, x: int, y: int) -> bool:
        if not self.allowed():
            return False
        if self.dry_run:
            self.log.info("DRY RUN: move to (%d, %d)", x, y)
            return True
        _, mouse = self._controllers()
        start_x, start_y = mouse.position
        distance = math.hypot(x - start_x, y - start_y)
        duration = min(0.40, max(0.08, distance / 2500.0))
        steps = min(32, max(8, round(duration * 90)))
        for point in cursor_path((start_x, start_y), (x, y), steps):
            if not self.allowed():
                return False
            mouse.position = point
            time.sleep(duration / steps)
        return True

    def right_click(self) -> bool:
        if not self.allowed():
            return False
        if self.dry_run:
            self.log.info("DRY RUN: right click")
            return True
        from pynput.keyboard import Key
        from pynput.mouse import Button
        keyboard, mouse = self._controllers()
        for modifier in (Key.shift, Key.ctrl, Key.alt):
            keyboard.release(modifier)
        if not self.allowed():
            return False
        mouse.click(Button.right)
        return True

    def release_modifiers(self) -> None:
        if self.keyboard is None:
            return
        from pynput.keyboard import Key
        for modifier in (Key.shift, Key.ctrl, Key.alt):
            self.keyboard.release(modifier)