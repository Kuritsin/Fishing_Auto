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


def extract_object_template(frame: np.ndarray, change_mask: np.ndarray, x: int, y: int,
                            radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a tight bobber crop and mask near a user-confirmed point.

    Thin line pixels and most moving water are deliberately excluded.  The mask
    is retained for diagnostics and future masked matching; the tight crop alone
    already makes ordinary template matching substantially less background-led.
    """
    height, width = frame.shape[:2]
    left, right = max(0, x - radius), min(width, x + radius + 1)
    top, bottom = max(0, y - radius), min(height, y + radius + 1)
    local = change_mask[top:bottom, left:right].copy()
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(local)
    selected: list[int] = []
    local_point = np.array((x - left, y - top), dtype=float)
    for label in range(1, count):
        bx, by, bw, bh, area = stats[label]
        distance = float(np.linalg.norm(centres[label] - local_point))
        # Reject thread-like components while retaining feathers and body.
        if area >= 8 and bw >= 3 and distance <= max(12.0, radius * 0.65):
            selected.append(label)
    if not selected:
        fallback = crop_around(frame, x, y, max(12, radius // 2))
        return fallback, np.full(fallback.shape[:2], 255, np.uint8)
    object_mask = np.isin(labels, selected).astype(np.uint8) * 255
    ys, xs = np.nonzero(object_mask)
    x0, x1 = max(0, int(xs.min()) - 4), min(local.shape[1], int(xs.max()) + 5)
    y0, y1 = max(0, int(ys.min()) - 4), min(local.shape[0], int(ys.max()) + 5)
    return frame[top + y0:top + y1, left + x0:left + x1].copy(), object_mask[y0:y1, x0:x1]


def match_templates(frame: np.ndarray, templates: list[np.ndarray], region: Region,
                    threshold: float, scales: tuple[float, ...] = (0.9, 1.0, 1.1)) -> Detection:
    best = Detection(False)
    for template in templates:
        found = match(frame, template, region, threshold, scales)
        if found.confidence > best.confidence:
            best = found
    return best


def loot_window_appeared(before: np.ndarray, after: np.ndarray) -> bool:
    """Detect a newly opened, persistent loot-sized panel in the upper-left."""
    if before.shape != after.shape or before.ndim != 3:
        return False
    height, width = after.shape[:2]
    # The classic loot window opens in this quadrant; ratios keep the check
    # independent of the user's resolution.
    before_roi = before[:round(height * 0.48), :round(width * 0.38)]
    after_roi = after[:round(height * 0.48), :round(width * 0.38)]
    changed = cv2.cvtColor(cv2.absdiff(before_roi, after_roi), cv2.COLOR_BGR2GRAY)
    mask = (changed >= 24).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = before_roi.shape[0] * before_roi.shape[1]
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        if area < roi_area * 0.035 or box_height < box_width * 0.7:
            continue
        panel = after_roi[y:y + box_height, x:x + box_width]
        dark_ratio = float(np.mean(cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) < 85))
        if dark_ratio >= 0.42:
            return True
    return False
