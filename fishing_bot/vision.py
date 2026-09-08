from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import Region


@dataclass(frozen=True)
class Detection:
    found: bool
    x: int = 0
    y: int = 0
    confidence: float = 0.0
    size: tuple[int, int] = (0, 0)


def prepare(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.GaussianBlur(gray, (3, 3), 0)


def match(frame: np.ndarray, template: np.ndarray, region: Region,
          threshold: float, scales: tuple[float, ...] = (0.85, 1.0, 1.15)) -> Detection:
    source, needle = prepare(frame), prepare(template)
    best = Detection(False)
    for scale in scales:
        width, height = round(needle.shape[1] * scale), round(needle.shape[0] * scale)
        if min(width, height) < 4 or width > source.shape[1] or height > source.shape[0]:
            continue
        resized = cv2.resize(needle, (width, height), interpolation=cv2.INTER_AREA)
        response = cv2.matchTemplate(source, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, point = cv2.minMaxLoc(response)
        if score > best.confidence:
            best = Detection(score >= threshold, region.left + point[0] + width // 2,
                             region.top + point[1] + height // 2, float(score), (width, height))
    return best


def stable_difference(before: list[np.ndarray], after: list[np.ndarray]) -> np.ndarray:
    background = np.median(np.stack(before), axis=0).astype(np.uint8)
    foreground = np.median(np.stack(after), axis=0).astype(np.uint8)
    difference = cv2.cvtColor(cv2.absdiff(background, foreground), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(difference, 22, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def candidate_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    image_area = mask.shape[0] * mask.shape[1]
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        maximum_area = max(500, image_area * 0.015)
        if 12 <= area <= maximum_area and width <= height * 4 and height <= width * 5:
            boxes.append((x, y, width, height))
    return sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)


def crop_around(frame: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    top, bottom = max(0, y - radius), min(frame.shape[0], y + radius + 1)
    left, right = max(0, x - radius), min(frame.shape[1], x + radius + 1)
    return frame[top:bottom, left:right].copy()
