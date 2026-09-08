from __future__ import annotations

import time
from pathlib import Path

import cv2

from config import PROFILE_PATH, Profile, Region
from .capture import ScreenCapture
from .input import SafeInput
from .vision import candidate_boxes, crop_around, stable_difference
from .window import WowWindow


def _wait_for_wow(window: WowWindow, seconds: float = 30.0) -> Region:
    print("Переключитесь в WoW. Ожидание активного окна...")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        region = window.client_region()
        if region:
            return region
        time.sleep(0.25)
    raise RuntimeError("WoW не стало активным за 30 секунд")


def _select(frame, boxes: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    preview = frame.copy()
    for x, y, width, height in boxes[:8]:
        cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 220, 255), 2)
    point: list[tuple[int, int]] = []

    def clicked(event, x, y, _flags, _value):
        if event == cv2.EVENT_LBUTTONDOWN:
            point[:] = [(x, y)]

    title = "Click the bobber, then press Enter"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, clicked)
    while not point:
        cv2.imshow(title, preview)
        if cv2.waitKey(30) & 0xFF == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Настройка отменена")
    cv2.circle(preview, point[0], 8, (0, 255, 0), 2)
    cv2.imshow(title, preview); cv2.waitKey(350); cv2.destroyAllWindows()
    return point[0]


def run_setup(window: WowWindow, capture: ScreenCapture) -> Profile:
    print("=== Первоначальная настройка ===")
    print("Включите Auto Loot, выключите Click to Move и не двигайте камерой во время теста.")
    cast_key = input("Клавиша Fishing [0]: ").strip().lower() or "0"
    client = _wait_for_wow(window)
    search = client.inset(0.08, 0.18, 0.16)
    before = [capture.grab(search) for _ in range(6)]
    sender = SafeInput(window.active, dry_run=False)
    if not sender.press(cast_key):
        raise RuntimeError("Ввод заблокирован: WoW не активно")
    time.sleep(1.25)
    after = []
    for _ in range(10):
        after.append(capture.grab(search)); time.sleep(0.06)
    mask = stable_difference(before, after)
    frame = after[-1]
    x, y = _select(frame, candidate_boxes(mask))
    radius = max(18, round(min(client.width, client.height) * 0.028))
    template = crop_around(frame, x, y, radius)
    data = Path(__file__).resolve().parents[1] / "data"
    data.mkdir(exist_ok=True)
    template_path = data / "bobber.png"
    if not cv2.imwrite(str(template_path), template):
        raise RuntimeError("Не удалось сохранить шаблон поплавка")
    profile = Profile(cast_key=cast_key, template_width_ratio=template.shape[1] / client.width,
                      template_height_ratio=template.shape[0] / client.height)
    profile.save(PROFILE_PATH)
    print("Поплавок сохранён автоматически. Настройка завершена.")
    return profile
