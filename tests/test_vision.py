import cv2
import numpy as np

from config import Region
from fishing_bot.vision import candidate_boxes, loot_window_appeared, match, stable_difference


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


def test_detects_new_dark_loot_panel():
    before = np.full((600, 800, 3), 120, dtype=np.uint8)
    after = before.copy()
    cv2.rectangle(after, (15, 40), (220, 340), (24, 24, 24), -1)
    cv2.rectangle(after, (15, 40), (220, 340), (80, 110, 150), 5)
    assert loot_window_appeared(before, after)
    assert not loot_window_appeared(before, before.copy())