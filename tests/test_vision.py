import cv2
import numpy as np

from config import Region
from fishing_bot.vision import (TemplateAsset, adaptive_novelty, candidate_boxes,
                                color_similarity,
                                extract_object_template, loot_window_appeared,
                                match, match_template_candidates, match_templates, stable_difference,
                                template_from_roi, usable_template)


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
    crop, object_mask, anchor = extract_object_template(frame, mask, 50, 55, 45)
    assert crop.shape[0] < 40
    assert crop.shape[1] < 40
    assert object_mask.shape == crop.shape[:2]
    assert 0 <= anchor[0] < crop.shape[1]
    assert 0 <= anchor[1] < crop.shape[0]


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
    assert result.template_index == 1


def test_static_template_is_rejected_without_post_cast_novelty():
    template = np.random.default_rng(8).integers(20, 240, (16, 18, 3), dtype=np.uint8)
    frame = np.zeros((70, 80, 3), dtype=np.uint8)
    frame[25:41, 30:48] = template
    novelty = np.zeros(frame.shape[:2], dtype=np.uint8)
    result = match_templates(frame, [template], Region(0, 0, 80, 70), 0.8,
                             (1.0,), novelty, 5)
    assert not result.found


def test_masked_template_uses_saved_anchor_and_novelty():
    rng = np.random.default_rng(9)
    template = rng.integers(30, 230, (16, 18, 3), dtype=np.uint8)
    mask = np.zeros((16, 18), dtype=np.uint8)
    mask[3:14, 4:16] = 255
    frame = np.zeros((70, 80, 3), dtype=np.uint8)
    frame[25:41, 30:48] = template
    novelty = np.zeros(frame.shape[:2], dtype=np.uint8)
    novelty[28:39, 34:46] = 255
    asset = TemplateAsset(template, mask, (6, 10))
    result = match_templates(frame, [asset], Region(100, 200, 80, 70), 0.8,
                             (1.0,), novelty, 5, 0.8)
    assert result.found
    assert (result.x, result.y) == (136, 235)


def test_masked_template_rejects_motion_only_outside_foreground():
    rng = np.random.default_rng(10)
    template = rng.integers(30, 230, (20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[7:14, 7:14] = 255
    frame = np.zeros((60, 70, 3), dtype=np.uint8)
    frame[20:40, 25:45] = template
    novelty = np.zeros(frame.shape[:2], dtype=np.uint8)
    novelty[20:25, 25:45] = 255
    result = match_templates(
        frame, [TemplateAsset(template, mask, (10, 10))], Region(0, 0, 70, 60),
        0.8, (1.0,), novelty, 4, 0.8,
    )
    assert not result.found


def test_adaptive_novelty_ignores_existing_static_object():
    before = [np.zeros((60, 70, 3), dtype=np.uint8) for _ in range(6)]
    for frame in before:
        frame[10:20, 12:24] = 180
    current = before[-1].copy()
    current[35:47, 42:54] = 220
    novelty = adaptive_novelty(before, current)
    assert not np.any(novelty[10:20, 12:24])
    assert np.any(novelty[35:47, 42:54])


def test_roi_template_falls_back_from_a_dangerously_tiny_mask():
    frame = np.full((80, 90, 3), 100, dtype=np.uint8)
    frame[25:55, 30:70] = (30, 80, 210)
    change = np.zeros(frame.shape[:2], dtype=np.uint8)
    change[38:40, 48:50] = 255
    crop, mask, anchor = template_from_roi(frame, change, (30, 25, 40, 30))
    assert crop.shape[:2] == (30, 40)
    assert np.all(mask == 255)
    assert anchor == (20, 15)
    assert usable_template(crop, mask)


def test_tiny_template_or_mask_is_not_usable():
    assert not usable_template(np.zeros((5, 7, 3), dtype=np.uint8), None)
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    tiny_mask = np.zeros((30, 40), dtype=np.uint8)
    tiny_mask[10:12, 10:12] = 255
    assert not usable_template(image, tiny_mask)


def test_color_similarity_rejects_gray_glare_with_same_luminance():
    template = np.zeros((24, 30, 3), dtype=np.uint8)
    cv2.rectangle(template, (3, 7), (14, 17), (20, 20, 220), -1)
    cv2.rectangle(template, (15, 5), (25, 15), (180, 60, 20), -1)
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    glare = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = np.full(template.shape[:2], 255, dtype=np.uint8)
    assert color_similarity(template, template, mask) > 0.99
    assert color_similarity(template, glare, mask) < 0.35


def test_color_ranking_prefers_bobber_over_structurally_similar_glare():
    template = np.zeros((24, 30, 3), dtype=np.uint8)
    cv2.rectangle(template, (3, 7), (14, 17), (20, 20, 220), -1)
    cv2.rectangle(template, (15, 5), (25, 15), (180, 60, 20), -1)
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    glare = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    frame = np.zeros((70, 120, 3), dtype=np.uint8)
    frame[22:46, 8:38] = glare
    frame[22:46, 72:102] = template
    novelty = np.zeros(frame.shape[:2], dtype=np.uint8)
    novelty[22:46, 8:38] = 255
    novelty[22:46, 72:102] = 255
    asset = TemplateAsset(template, np.full((24, 30), 255, np.uint8), (15, 12))
    candidates = match_template_candidates(
        frame, [asset], Region(0, 0, 120, 70), 0.72, (1.0,), novelty, 10,
        0.72, 3, 0.28,
    )
    assert candidates[0].found
    assert (candidates[0].x, candidates[0].y) == (87, 34)
    assert candidates[0].color_score > candidates[1].color_score


def test_gray_glare_alone_does_not_confirm_a_cast():
    template = np.zeros((24, 30, 3), dtype=np.uint8)
    cv2.rectangle(template, (3, 7), (14, 17), (20, 20, 220), -1)
    cv2.rectangle(template, (15, 5), (25, 15), (180, 60, 20), -1)
    glare = cv2.cvtColor(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[20:44, 25:55] = glare
    novelty = np.zeros(frame.shape[:2], dtype=np.uint8)
    novelty[20:44, 25:55] = 255
    asset = TemplateAsset(template, np.full((24, 30), 255, np.uint8), (15, 12))
    result = match_templates(frame, [asset], Region(0, 0, 80, 60), 0.72,
                             (1.0,), novelty, 10, 0.72, 3, 0.28)
    assert not result.found


def test_detects_new_dark_loot_panel():
    before = np.full((600, 800, 3), 120, dtype=np.uint8)
    after = before.copy()
    cv2.rectangle(after, (15, 40), (220, 340), (24, 24, 24), -1)
    cv2.rectangle(after, (15, 40), (220, 340), (80, 110, 150), 5)
    assert loot_window_appeared(before, after)
    assert not loot_window_appeared(before, before.copy())