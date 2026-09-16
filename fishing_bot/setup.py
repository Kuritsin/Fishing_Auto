from __future__ import annotations

import time
from pathlib import Path
import shutil

import cv2

from config import PROFILE_PATH, Profile, Region
from .capture import ScreenCapture
from .input import SafeInput
from .vision import candidate_boxes, extract_object_template, stable_difference
from .window import WowWindow


def _wait_for_wow(window: WowWindow, seconds: float | None = None) -> Region:
    region = window.client_region()
    if region:
        return region
    print("Переключитесь в WoW. Ожидание активного окна...")
    deadline = None if seconds is None else time.monotonic() + seconds
    while deadline is None or time.monotonic() < deadline:
        region = window.client_region()
        if region:
            return region
        time.sleep(0.25)
    raise RuntimeError("WoW не стало активным за отведённое время")


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


def run_setup(window: WowWindow, capture: ScreenCapture, dry_run: bool = False) -> Profile:
    print("=== Первоначальная настройка ===")
    print("Включите Auto Loot, выключите Click to Move и не двигайте камерой во время теста.")
    cast_key = input("Клавиша Fishing [0]: ").strip().lower() or "0"
    client = _wait_for_wow(window)
    sender = SafeInput(window.active, dry_run=dry_run)
    radius = max(18, round(min(client.width, client.height) * 0.028))
    data = Path(__file__).resolve().parents[1] / "data"
    data.mkdir(exist_ok=True)
    pending = data / ".setup_pending"
    shutil.rmtree(pending, ignore_errors=True)
    pending.mkdir()
    template_files: list[str] = []
    mask_files: list[str] = []
    anchors: list[list[int]] = []
    shapes: list[tuple[int, int]] = []
    try:
        for attempt in range(1, 4):
            # Selecting the previous bobber activates the OpenCV preview window.
            # Wait instead of aborting and refresh coordinates in case WoW moved.
            client = _wait_for_wow(window)
            search = client.inset(0.08, 0.18, 0.16)
            print(f"Тестовый заброс {attempt}/3: не двигайте камерой")
            before = [capture.grab(search) for _ in range(8)]
            if dry_run:
                print("DRY RUN: выполните заброс вручную в течение следующих 3 секунд")
                time.sleep(3.0)
            elif not sender.press(cast_key):
                client = _wait_for_wow(window)
                search = client.inset(0.08, 0.18, 0.16)
                if not sender.press(cast_key):
                    raise RuntimeError("Не удалось выполнить тестовый заброс")
            time.sleep(1.25)
            after = []
            for _ in range(12):
                after.append(capture.grab(search)); time.sleep(0.06)
            mask = stable_difference(before, after)
            frame = after[-1]
            x, y = _select(frame, candidate_boxes(mask))
            template, object_mask, anchor = extract_object_template(frame, mask, x, y, radius)
            relative = f"data/bobber_{attempt}.png"
            mask_relative = f"data/bobber_{attempt}_mask.png"
            template_path = pending / f"bobber_{attempt}.png"
            mask_path = pending / f"bobber_{attempt}_mask.png"
            if not cv2.imwrite(str(template_path), template) or not cv2.imwrite(
                    str(mask_path), object_mask):
                raise RuntimeError("Не удалось сохранить шаблон поплавка")
            template_files.append(relative)
            mask_files.append(mask_relative)
            anchors.append([anchor[0], anchor[1]])
            shapes.append(template.shape[:2])
        for staged in pending.iterdir():
            staged.replace(data / staged.name)
    finally:
        shutil.rmtree(pending, ignore_errors=True)
    median_h = sorted(shape[0] for shape in shapes)[len(shapes) // 2]
    median_w = sorted(shape[1] for shape in shapes)[len(shapes) // 2]
    profile = Profile(cast_key=cast_key, template_file=template_files[0],
                      template_files=template_files,
                      template_mask_files=mask_files,
                      template_anchors=anchors,
                      template_width_ratio=median_w / client.width,
                      template_height_ratio=median_h / client.height)
    profile.save(PROFILE_PATH)
    print("Три вида поплавка сохранены автоматически. Настройка завершена.")
    return profile
