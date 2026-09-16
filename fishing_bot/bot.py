from __future__ import annotations

import logging
from pathlib import Path
import threading
import time

import cv2

from config import Profile, SETTINGS
from .capture import ScreenCapture
from .input import SafeInput
from .tracker import BobberTracker
from .vision import Detection, loot_window_appeared, match_templates
from .window import WowWindow


class FishingBot:
    def __init__(self, profile: Profile, window: WowWindow, capture: ScreenCapture,
                 dry_run: bool = False) -> None:
        self.profile, self.window, self.capture = profile, window, capture
        self.stop_event, self.paused = threading.Event(), threading.Event()
        self.input = SafeInput(self.safe, dry_run)
        self.tracker = BobberTracker(capture, SETTINGS)
        self.log = logging.getLogger(__name__)
        root = Path(__file__).resolve().parents[1]
        self.templates = [image for name in profile.templates()
                          if (image := cv2.imread(str(root / name))) is not None]
        if not self.templates:
            raise RuntimeError("Шаблон поплавка не найден; повторите настройку")

    def safe(self) -> bool:
        return not self.stop_event.is_set() and not self.paused.is_set() and self.window.active()

    def toggle_pause(self) -> None:
        if self.paused.is_set():
            self.paused.clear(); self.log.info("Продолжение")
        else:
            self.paused.set(); self.input.release_modifiers(); self.log.info("Пауза")

    def stop(self) -> None:
        self.stop_event.set(); self.input.release_modifiers()

    def _find(self, client, deadline: float) -> Detection:
        search = client.inset(0.08, 0.18, 0.16)
        best = Detection(False)
        previous = Detection(False)
        confirmations = 0
        while time.monotonic() < deadline and self.safe():
            found = match_templates(self.capture.grab(search), self.templates, search,
                                    SETTINGS.match_confidence)
            if found.confidence > best.confidence:
                best = found
            if found.found and previous.found and abs(found.x - previous.x) < found.size[0] and abs(found.y - previous.y) < found.size[1]:
                confirmations += 1
            else:
                confirmations = 1 if found.found else 0
            previous = found
            required = 1 if found.confidence >= SETTINGS.strong_match_confidence else 2
            if confirmations >= required:
                return found
            self.stop_event.wait(0.06)
        return Detection(False, confidence=best.confidence)

    def attempt(self) -> None:
        client = self.window.client_region()
        if not client or not self.safe():
            return
        started = time.monotonic(); deadline = started + SETTINGS.attempt_timeout
        self.log.info("Заброс")
        if not self.input.press(self.profile.cast_key):
            return
        if self.stop_event.wait(SETTINGS.cast_settle_seconds):
            return
        found = self._find(client, min(deadline, time.monotonic() + SETTINGS.find_timeout))
        if not found.found:
            self.log.warning("Поплавок не найден; best=%.3f", found.confidence)
            return
        self.log.info("Поплавок найден за %.2f с: (%d, %d), confidence=%.3f",
                      time.monotonic() - started, found.x, found.y, found.confidence)
        if not self.input.move(found.x, found.y):
            return
        # Track with the template whose dimensions best match the accepted result.
        template = min(self.templates, key=lambda item: abs(item.shape[1] - found.size[0]) +
                       abs(item.shape[0] - found.size[1]))
        result = self.tracker.wait(found, template, client,
                                   max(0.0, deadline - time.monotonic()),
                                   self.stop_event, self.safe)
        if not result.detected:
            self.log.warning("Поклёвка не обнаружена: %s", result.reason)
            return
        self.log.info("Поклёвка: %s drop=%.1f velocity=%.1f", result.reason, result.drop, result.velocity)
        before_loot = self.capture.grab(client)
        if not self.input.move(result.x, result.y) or not self.input.right_click():
            return
        self.stop_event.wait(SETTINGS.post_loot_delay)
        if self.safe() and loot_window_appeared(before_loot, self.capture.grab(client)):
            self.log.warning(
                "Окно добычи осталось открытым. Проверьте, что Auto Loot включён; "
                "повторный клик намеренно не выполняется"
            )

    def run(self, once: bool = False) -> None:
        self.log.info("Бот запущен")
        while not self.stop_event.is_set():
            if self.paused.is_set() or not self.window.active():
                self.stop_event.wait(SETTINGS.inactive_delay); continue
            try:
                self.attempt()
            except Exception:
                self.log.exception("Ошибка попытки; бот продолжит работу")
            if once:
                break
            self.stop_event.wait(SETTINGS.retry_delay)
