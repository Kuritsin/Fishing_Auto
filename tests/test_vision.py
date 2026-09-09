import cv2
import numpy as np

from config import Region
from fishing_bot.vision import (candidate_boxes, extract_object_template,
                                loot_window_appeared, match, match_templates,
                                stable_difference)


def test_template_match_returns_screen_coordinates():
    template = np.random.default_rng(5).integers(0, 255, (20, 14, 3), dtype=np.uint8)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[40:60, 50:64] = template
    result = match(frame, template, Region(300, 200, 120, 100), 0.8, (1.0,))
    assert result.found
    assert (result.x, result.y) == (357, 250)


def test_stable_difference_proposes_new_object():
    before = [np.zeros((80, 100, 3), dtype=np.uint8) for _ in range(5)]
    after = [frame.copy() for frame in before]
    for frame in after:
        cv2.rectangle(frame, (40, 30), (51, 47), (220, 220, 220), -1)
    boxes = candidate_boxes(stable_difference(before, after))
    assert boxes
    x, y, width, height = boxes[0]
    assert x <= 40 < x + width and y <= 30 < y + height


def test_extract_object_template_ignores_distant_line_and_background():
    frame = np.full((100, 100, 3), 90, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(frame, (42, 47), (59, 65), (20, 20, 220), -1)
    cv2.rectangle(mask, (42, 47), (59, 65), 255, -1)
    cv2.line(frame, (50, 5), (50, 35), (230, 230, 230), 1)
    cv2.line(mask, (50, 5), (50, 35), 255, 1)
    crop, object_mask = extract_object_template(frame, mask, 50, 55, 45)
    assert crop.shape[0] < 40
    assert crop.shape[1] < 40
    assert object_mask.shape == crop.shape[:2]


def test_match_templates_selects_the_matching_view():
    rng = np.random.default_rng(7)
    wrong = rng.integers(0, 255, (18, 18, 3), dtype=np.uint8)
    correct = rng.integers(0, 255, (18, 18, 3), dtype=np.uint8)
    frame = np.zeros((80, 90, 3), dtype=np.uint8)
    frame[30:48, 41:59] = correct
    result = match_templates(frame, [wrong, correct], Region(100, 200, 90, 80),
                             0.8, (1.0,))
    assert result.found
    assert (result.x, result.y) == (150, 239)


def test_detects_new_dark_loot_panel():
    before = np.full((600, 800, 3), 120, dtype=np.uint8)
    after = before.copy()
    cv2.rectangle(after, (15, 40), (220, 340), (24, 24, 24), -1)
    cv2.rectangle(after, (15, 40), (220, 340), (80, 110, 150), 5)
    assert loot_window_appeared(before, after)
    assert not loot_window_appeared(before, before.copy())
