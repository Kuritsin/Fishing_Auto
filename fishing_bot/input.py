from __future__ import annotations

import logging
import time
from collections.abc import Callable


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
        for step in range(1, 13):
            if not self.allowed():
                return False
            fraction = step / 12
            mouse.position = (round(start_x + (x - start_x) * fraction),
                              round(start_y + (y - start_y) * fraction))
            time.sleep(0.01)
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
