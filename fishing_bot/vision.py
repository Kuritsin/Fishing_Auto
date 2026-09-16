from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import Region


VISION_API_VERSION = 3


@dataclass(frozen=True)
class Detection:
    found: bool
    x: int = 0
    y: int = 0
    confidence: float = 0.0
    size: tuple[int, int] = (0, 0)
    template_index: int = -1
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    structural_score: float = 0.0
    color_score: float = 0.0
    novelty_score: float = 0.0


@dataclass(frozen=True)
class TemplateAsset:
    image: np.ndarray
    mask: np.ndarray | None = None
    anchor: tuple[int, int] | None = None

    def click_anchor(self) -> tuple[int, int]:
        return self.anchor or (self.image.shape[1] // 2, self.image.shape[0] // 2)


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


def adaptive_novelty(before: list[np.ndarray], frame: np.ndarray) -> np.ndarray:
    """Find changes which exceed the motion already present before casting."""
    gray_before = np.stack([prepare(item) for item in before]).astype(np.float32)
    background = np.median(gray_before, axis=0)
    variability = np.median(np.abs(gray_before - background), axis=0)
    current = prepare(frame).astype(np.float32)
    changed = np.abs(current - background) >= np.maximum(18.0, variability * 3.5 + 8.0)
    mask = changed.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


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


def template_from_roi(frame: np.ndarray, change_mask: np.ndarray,
                      roi: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray,
                                                               tuple[int, int]]:
    """Build a stable template from a user-drawn box around the complete bobber."""
    x, y, width, height = roi
    if width < 16 or height < 16:
        raise ValueError("Рамка поплавка слишком мала")
    crop = frame[y:y + height, x:x + width].copy()
    mask = change_mask[y:y + height, x:x + width].copy()
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    foreground = int(np.count_nonzero(mask))
    minimum = max(50, round(width * height * 0.08))
    # A tiny mask produces near-perfect matches on random highlights.  The
    # manually selected full crop is a safer fallback when differencing failed.
    if foreground < minimum:
        mask = np.full((height, width), 255, np.uint8)
    return crop, mask, (width // 2, height // 2)


def usable_template(image: np.ndarray, mask: np.ndarray | None) -> bool:
    if image.size == 0 or min(image.shape[:2]) < 16 or image.shape[0] * image.shape[1] < 400:
        return False
    if mask is None:
        return True
    minimum = max(50, round(mask.size * 0.08))
    return mask.shape == image.shape[:2] and np.count_nonzero(mask) >= minimum


def color_similarity(template: np.ndarray, candidate: np.ndarray,
                     mask: np.ndarray | None = None) -> float:
    """Compare chroma while remaining tolerant of global brightness changes."""
    if template.shape[:2] != candidate.shape[:2]:
        return 0.0
    selected = np.ones(template.shape[:2], dtype=bool) if mask is None else mask > 0
    if np.count_nonzero(selected) < 4:
        return 0.0
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    candidate_hsv = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV)

    def descriptor(hsv: np.ndarray) -> tuple[np.ndarray, float, float]:
        hue = hsv[..., 0][selected]
        saturation = hsv[..., 1][selected].astype(np.float32)
        weights = saturation / 255.0
        histogram, _ = np.histogram(hue, bins=12, range=(0, 180), weights=weights)
        total = float(histogram.sum())
        if total > 0:
            histogram = histogram.astype(np.float32) / total
        return histogram, float(np.mean(saturation >= 55)), float(np.mean(saturation) / 255.0)

    template_hist, template_fraction, template_mean = descriptor(template_hsv)
    candidate_hist, candidate_fraction, candidate_mean = descriptor(candidate_hsv)
    if template_fraction < 0.04 and template_mean < 0.12:
        return 1.0
    histogram_score = float(np.minimum(template_hist, candidate_hist).sum())
    fraction_score = max(0.0, 1.0 - abs(template_fraction - candidate_fraction) /
                         max(0.12, template_fraction))
    mean_score = max(0.0, 1.0 - abs(template_mean - candidate_mean) /
                     max(0.15, template_mean))
    return float(np.clip(histogram_score * 0.60 + fraction_score * 0.25 +
                         mean_score * 0.15, 0.0, 1.0))


def extract_object_template(frame: np.ndarray, change_mask: np.ndarray, x: int, y: int,
                            radius: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
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
    candidates: list[tuple[float, int]] = []
    local_point = np.array((x - left, y - top), dtype=float)
    for label in range(1, count):
        bx, by, bw, bh, area = stats[label]
        distance = float(np.linalg.norm(centres[label] - local_point))
        # Reject thread-like components while retaining feathers and body.
        if area >= 8 and bw >= 3 and distance <= max(12.0, radius * 0.65):
            candidates.append((distance, label))
    if not candidates:
        fallback = crop_around(frame, x, y, max(12, radius // 2))
        mask = np.full(fallback.shape[:2], 255, np.uint8)
        return fallback, mask, (fallback.shape[1] // 2, fallback.shape[0] // 2)
    point_x = int(np.clip(x - left, 0, labels.shape[1] - 1))
    point_y = int(np.clip(y - top, 0, labels.shape[0] - 1))
    point_label = int(labels[point_y, point_x])
    candidate_labels = {label for _, label in candidates}
    selected = point_label if point_label in candidate_labels else min(candidates)[1]
    object_mask = (labels == selected).astype(np.uint8) * 255
    ys, xs = np.nonzero(object_mask)
    x0, x1 = max(0, int(xs.min()) - 4), min(local.shape[1], int(xs.max()) + 5)
    y0, y1 = max(0, int(ys.min()) - 4), min(local.shape[0], int(ys.max()) + 5)
    crop = frame[top + y0:top + y1, left + x0:left + x1].copy()
    anchor = (int(np.clip(x - left - x0, 0, crop.shape[1] - 1)),
              int(np.clip(y - top - y0, 0, crop.shape[0] - 1)))
    return crop, object_mask[y0:y1, x0:x1], anchor


def _match_asset_candidates(frame: np.ndarray, asset: TemplateAsset, region: Region,
                            threshold: float, scales: tuple[float, ...],
                            novelty: np.ndarray | None, minimum_novelty_pixels: int,
                            maximum_candidates: int, color_weight: float) -> list[Detection]:
    source, needle = prepare(frame), prepare(asset.image)
    candidates: list[Detection] = []
    for scale in scales:
        width, height = round(needle.shape[1] * scale), round(needle.shape[0] * scale)
        if min(width, height) < 4 or width > source.shape[1] or height > source.shape[0]:
            continue
        resized = cv2.resize(needle, (width, height), interpolation=cv2.INTER_AREA)
        scaled_mask = None
        if asset.mask is not None:
            scaled_mask = cv2.resize(asset.mask, (width, height), interpolation=cv2.INTER_NEAREST)
            if np.count_nonzero(scaled_mask) < 4:
                scaled_mask = None
        if scaled_mask is None:
            response = cv2.matchTemplate(source, resized, cv2.TM_CCOEFF_NORMED)
        else:
            weights = (scaled_mask > 0).astype(np.float32)
            needle_float = resized.astype(np.float32)
            source_float = source.astype(np.float32)
            weight_sum = float(np.sum(weights))
            needle_mean = float(np.sum(needle_float * weights) / weight_sum)
            centred_needle = (needle_float - needle_mean) * weights
            needle_energy = float(np.sum(centred_needle ** 2))
            source_sum = cv2.matchTemplate(source_float, weights, cv2.TM_CCORR)
            source_square_sum = cv2.matchTemplate(source_float ** 2, weights,
                                                  cv2.TM_CCORR)
            source_energy = np.maximum(
                0.0, source_square_sum - source_sum ** 2 / weight_sum
            )
            numerator = cv2.matchTemplate(source_float, centred_needle, cv2.TM_CCORR)
            denominator = np.sqrt(source_energy * max(needle_energy, 1e-6))
            response = np.divide(numerator, denominator, out=np.full_like(numerator, -1.0),
                                 where=denominator > 1e-6)
            response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
        novelty_counts = None
        novelty_denominator = 1.0
        if novelty is not None:
            novelty_binary = (novelty > 0).astype(np.float32)
            novelty_kernel = ((scaled_mask > 0).astype(np.float32) if scaled_mask is not None
                              else np.ones((height, width), np.float32))
            novelty_counts = cv2.matchTemplate(novelty_binary, novelty_kernel, cv2.TM_CCORR)
            required_novelty = max(minimum_novelty_pixels,
                                   round(np.count_nonzero(novelty_kernel) * 0.08))
            response = np.where(novelty_counts >= required_novelty, response, -1.0)
            novelty_denominator = float(max(1, np.count_nonzero(novelty_kernel)))
        remaining = response.copy()
        for _ in range(maximum_candidates):
            _, structural_score, _, point = cv2.minMaxLoc(remaining)
            if structural_score <= 0:
                break
            px, py = point
            candidate_patch = frame[py:py + height, px:px + width]
            resized_colour = cv2.resize(asset.image, (width, height), interpolation=cv2.INTER_AREA)
            chroma_score = color_similarity(resized_colour, candidate_patch, scaled_mask)
            novelty_score = (float(novelty_counts[py, px]) / novelty_denominator
                             if novelty_counts is not None else 1.0)
            combined = float(structural_score)
            if color_weight > 0:
                combined *= ((1.0 - color_weight - 0.05) +
                             color_weight * chroma_score +
                             0.05 * min(1.0, novelty_score * 4.0))
            anchor_x, anchor_y = asset.click_anchor()
            screen_x = region.left + px + round(anchor_x * scale)
            screen_y = region.top + py + round(anchor_y * scale)
            candidates.append(Detection(
                combined >= threshold, screen_x, screen_y, combined, (width, height), -1,
                (region.left + px, region.top + py, width, height),
                float(structural_score), chroma_score, novelty_score,
            ))
            suppress_left, suppress_top = max(0, px - width // 2), max(0, py - height // 2)
            suppress_right = min(remaining.shape[1], px + width // 2 + 1)
            suppress_bottom = min(remaining.shape[0], py + height // 2 + 1)
            remaining[suppress_top:suppress_bottom, suppress_left:suppress_right] = -1.0
    return candidates


def match_template_candidates(
        frame: np.ndarray, templates: list[np.ndarray | TemplateAsset], region: Region,
        threshold: float, scales: tuple[float, ...] = (0.9, 1.0, 1.1),
        novelty: np.ndarray | None = None, minimum_novelty_pixels: int = 1,
        masked_threshold: float | None = None, maximum_candidates: int = 3,
        color_weight: float = 0.0) -> list[Detection]:
    ranked: list[Detection] = []
    for index, template in enumerate(templates):
        asset = template if isinstance(template, TemplateAsset) else TemplateAsset(template)
        effective_threshold = (masked_threshold if asset.mask is not None and
                               masked_threshold is not None else threshold)
        for found in _match_asset_candidates(
                frame, asset, region, effective_threshold, scales, novelty,
                minimum_novelty_pixels, maximum_candidates, color_weight):
            ranked.append(Detection(
                found.found, found.x, found.y, found.confidence, found.size, index,
                found.box, found.structural_score, found.color_score, found.novelty_score,
            ))
    ranked.sort(key=lambda item: item.confidence, reverse=True)
    distinct: list[Detection] = []
    for candidate in ranked:
        if any(abs(candidate.x - saved.x) < max(6, candidate.size[0] // 2) and
               abs(candidate.y - saved.y) < max(6, candidate.size[1] // 2)
               for saved in distinct):
            continue
        distinct.append(candidate)
        if len(distinct) >= maximum_candidates:
            break
    return distinct


def match_templates(frame: np.ndarray, templates: list[np.ndarray | TemplateAsset], region: Region,
                    threshold: float, scales: tuple[float, ...] = (0.9, 1.0, 1.1),
                    novelty: np.ndarray | None = None, minimum_novelty_pixels: int = 1,
                    masked_threshold: float | None = None, maximum_candidates: int = 1,
                    color_weight: float = 0.0) -> Detection:
    candidates = match_template_candidates(
        frame, templates, region, threshold, scales, novelty, minimum_novelty_pixels,
        masked_threshold, maximum_candidates, color_weight,
    )
    return candidates[0] if candidates else Detection(False)


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