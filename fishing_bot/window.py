from __future__ import annotations

import ctypes
import os

from config import Region


class WowWindow:
    def __init__(self, allowed_titles: tuple[str, ...]) -> None:
        self.allowed_titles = tuple(title.lower() for title in allowed_titles)

    def _handle(self) -> int:
        if os.name != "nt":
            return 0
        return int(ctypes.windll.user32.GetForegroundWindow())

    def title(self) -> str:
        handle = self._handle()
        if not handle:
            return ""
        length = ctypes.windll.user32.GetWindowTextLengthW(handle)
        value = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(handle, value, length + 1)
        return value.value

    def active(self) -> bool:
        title = self.title().lower()
        return bool(title) and any(allowed in title for allowed in self.allowed_titles)

    def client_region(self) -> Region | None:
        handle = self._handle()
        if not handle or not self.active():
            return None

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        rectangle, origin = RECT(), POINT(0, 0)
        user32 = ctypes.windll.user32
        if not user32.GetClientRect(handle, ctypes.byref(rectangle)):
            return None
        if not user32.ClientToScreen(handle, ctypes.byref(origin)):
            return None
        width, height = rectangle.right - rectangle.left, rectangle.bottom - rectangle.top
        return Region(origin.x, origin.y, width, height) if width > 0 and height > 0 else None


def enable_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
