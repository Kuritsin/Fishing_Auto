
from __future__ import annotations

import logging
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from config import Profile, SETTINGS
from . import vision
from .capture import ScreenCapture
from .input import SafeInput
from .tracker import BobberTracker
from .window import WowWindow


REQUIRED_VISION_API_VERSION = 3


def validate_vision_api() -> None:
    actual = getattr(vision, "VISION_API_VERSION", 0)
    if actual != REQUIRED_VISION_API_VERSION:
        raise RuntimeError(
            "Файлы проекта имеют разные версии (bot.py/vision.py). "
            "Обновите весь репозиторий целиком, затем удалите каталоги __pycache__."
        )


class FishingBot:
    def __init__(self, profile: Profile, window: WowWindow, capture: ScreenCapture,
                 dry_run: bool = False, debug: bool = False) -> None:
        validate_vision_api()
        self.profile, self.window, self.capture = profile, window, capture
        self.stop_event, self.paused = threading.Event(), threading.Event()
        self.input = SafeInput(self.safe, dry_run)
        self.tracker = BobberTracker(capture, SETTINGS)
        self.log = logging.getLogger(__name__)
        root = Path(__file__).resolve().parents[1]
        self.debug, self.debug_dir = debug, root / "debug"
        masks = profile.template_mask_files or []
        anchors = profile.template_anchors or []
        self.templates: list[vision.TemplateAsset] = []
        for index, name in enumerate(profile.templates()):
            image = cv2.imread(str(root / name))
            if image is None:
                continue
            mask = (cv2.imread(str(root / masks[index]), cv2.IMREAD_GRAYSCALE)
                    if index < len(masks) else None)
            anchor = tuple(anchors[index]) if index < len(anchors) else None
            if not vision.usable_template(image, mask):
                self.log.warning("Шаблон %s пропущен: изображение или маска слишком малы", name)
                continue
            self.templates.append(vision.TemplateAsset(image, mask, anchor))
        if not self.templates:
            raise RuntimeError("Шаблон поплавка не найден; повторите настройку")
        if not masks or not anchors:
            self.log.warning("Профиль создан старой версией; выполните python main.py --setup")
        if self.debug:
            self.debug_dir.mkdir(exist_ok=True)

    def safe(self) -> bool:
        return not self.stop_event.is_set() and not self.paused.is_set() and self.window.active()

    def toggle_pause(self) -> None:
        if self.paused.is_set():
            self.paused.clear(); self.log.info("Продолжение")
        else:
            self.paused.set(); self.input.release_modifiers(); self.log.info("Пауза")

    def stop(self) -> None:
        self.stop_event.set(); self.input.release_modifiers()

    def _save_find_debug(self, frame: np.ndarray, novelty: np.ndarray,
                         found: vision.Detection,
                         candidates: list[vision.Detection] | None = None) -> None:
        if not self.debug:
            return
        preview = frame.copy()
        candidates = candidates or ([found] if found.box[2] else [])
        report = []
        for rank, candidate in enumerate(candidates[:3], 1):
            left, top, width, height = candidate.box
            if not width or not height:
                continue
            search_left, search_top = left - self._debug_search.left, top - self._debug_search.top
            colour = ((0, 220, 0) if candidate is found and found.found else
                      (0, 165, 255) if rank == 1 else (255, 160, 0))
            cv2.rectangle(preview, (search_left, search_top),
                          (search_left + width, search_top + height), colour, 2)
            cv2.circle(preview, (candidate.x - self._debug_search.left,
                                 candidate.y - self._debug_search.top), 6, colour, 2)
            label = (f"#{rank} total={candidate.confidence:.3f} "
                     f"shape={candidate.structural_score:.3f} "
                     f"color={candidate.color_score:.2f} t={candidate.template_index + 1}")
            cv2.putText(preview, label,
                        (max(0, search_left), max(18, search_top - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
            report.append({
                "rank": rank, "accepted": bool(candidate is found and found.found),
                "total": candidate.confidence, "structural": candidate.structural_score,
                "color": candidate.color_score, "novelty": candidate.novelty_score,
                "template": candidate.template_index + 1, "box": candidate.box,
                "anchor": [candidate.x, candidate.y],
            })
        cv2.imwrite(str(self.debug_dir / "latest_find.png"), preview)
        cv2.imwrite(str(self.debug_dir / "latest_novelty.png"), novelty)
        (self.debug_dir / "latest_find.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    def _find(self, search, background: list[np.ndarray], deadline: float) -> vision.Detection:
        self._debug_search = search
        best = vision.Detection(False)
        previous = vision.Detection(False)
        confirmations = 0
        last_frame = background[-1]
        last_novelty = np.zeros(last_frame.shape[:2], np.uint8)
        last_candidates: list[vision.Detection] = []
        while time.monotonic() < deadline and self.safe():
            last_frame = self.capture.grab(search)
            last_novelty = vision.adaptive_novelty(background, last_frame)
            last_candidates = vision.match_template_candidates(
                last_frame, self.templates, search, SETTINGS.match_confidence, (1.0,),
                last_novelty, SETTINGS.minimum_novelty_pixels,
                SETTINGS.masked_match_confidence, SETTINGS.finder_candidates,
                SETTINGS.finder_color_weight,
            )
            found = last_candidates[0] if last_candidates else vision.Detection(False)
            if found.confidence > best.confidence:
                best = found
            if found.found and previous.found and abs(found.x - previous.x) < found.size[0] and abs(found.y - previous.y) < found.size[1]:
                confirmations += 1
            else:
                confirmations = 1 if found.found else 0
            previous = found
            masked = (0 <= found.template_index < len(self.templates) and
                      self.templates[found.template_index].mask is not None)
            strong_threshold = (SETTINGS.strong_masked_match_confidence if masked else
                                SETTINGS.strong_match_confidence)
            required = 1 if found.confidence >= strong_threshold else 2
            if confirmations >= required:
                self._save_find_debug(last_frame, last_novelty, found, last_candidates)
                return found
            self.stop_event.wait(0.06)
        rejected = vision.Detection(False, best.x, best.y, best.confidence, best.size,
                                    best.template_index, best.box)
        self._save_find_debug(last_frame, last_novelty, rejected, last_candidates)
        return rejected

    def attempt(self) -> bool:
        client = self.window.client_region()
        if not client or not self.safe():
            return False
        started = time.monotonic(); deadline = started + SETTINGS.attempt_timeout
        search = client.inset(0.08, 0.18, 0.16)
        background = []
        for _ in range(8):
            background.append(self.capture.grab(search))
            if self.stop_event.wait(0.035):
                return False
        self.log.info("Заброс")
        if self.input.dry_run:
            self.log.info("DRY RUN: выполните заброс вручную в течение 3 секунд")
            if self.stop_event.wait(3.0):
                return False
        elif not self.input.press(self.profile.cast_key):
            return False
        if self.stop_event.wait(SETTINGS.cast_settle_seconds):
            return False
        found = self._find(search, background,
                           min(deadline, time.monotonic() + SETTINGS.find_timeout))
        if not found.found:
            self.log.warning("Заброс не подтверждён: подходящий новый поплавок не найден; best=%.3f",
                             found.confidence)
            return True
        self.log.info("Поплавок найден за %.2f с: (%d, %d), confidence=%.3f",
                      time.monotonic() - started, found.x, found.y, found.confidence)
        if not self.input.move(found.x, found.y):
            self.log.info("Наведение отменено: WoW больше не активно")
            return False
        # Finding the bobber can take several seconds.  The fishing window must
        # start after it was found; otherwise a slow finder silently shortens
        # the useful bite wait to about 18 seconds.
        result = self.tracker.wait(found, self.templates, client,
                                   SETTINGS.bite_wait_timeout,
                                   self.stop_event, self.safe)
        if not result.detected:
            self.log.warning("Поклёвка не обнаружена: %s", result.reason)
            return result.reason not in ("interrupted",)
        self.log.info("Поклёвка: %s drop=%.1f velocity=%.1f", result.reason, result.drop, result.velocity)
        before_loot = self.capture.grab(client)
        if self.debug:
            preview = before_loot.copy()
            cv2.circle(preview, (result.x - client.left, result.y - client.top),
                       8, (0, 0, 255), 2)
            cv2.imwrite(str(self.debug_dir / "latest_bite.png"), preview)
        if not self.input.move(result.x, result.y):
            self.log.info("Клик отменён: WoW больше не активно")
            return False
        if not self.input.right_click():
            self.log.info("Клик отменён проверкой безопасности")
            return False
        self.log.info("ПКМ выполнен: (%d, %d)", result.x, result.y)
        self.stop_event.wait(SETTINGS.post_loot_delay)
        if self.safe() and vision.loot_window_appeared(before_loot, self.capture.grab(client)):
            self.log.warning(
                "Окно добычи осталось открытым. Проверьте, что Auto Loot включён; "
                "повторный клик намеренно не выполняется"
            )
        return True

    def run(self, once: bool = False) -> None:
        self.log.info("Бот запущен")
        waiting_logged = False
        while not self.stop_event.is_set():
            if self.paused.is_set() or not self.window.active():
                if not waiting_logged:
                    self.log.info("WoW не активно; бот ждёт и не выполняет ввод")
                    waiting_logged = True
                self.stop_event.wait(SETTINGS.inactive_delay); continue
            if waiting_logged:
                self.log.info("WoW снова активно; работа продолжена")
                waiting_logged = False
            try:
                completed = self.attempt()
            except Exception:
                self.log.exception("Ошибка попытки; бот продолжит работу")
                completed = True
            if once and completed:
                break
            self.stop_event.wait(SETTINGS.retry_delay)
