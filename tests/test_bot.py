import threading
import time

import numpy as np

from config import Region
from fishing_bot import vision
from fishing_bot.bot import FishingBot, remaining_cast_time


def test_finder_time_is_part_of_the_cast_lifetime():
    assert remaining_cast_time(100.0, 103.5, 22.0) == 18.5


def test_shorter_expansion_cast_can_finish_tracking_early():
    assert remaining_cast_time(100.0, 123.0, 22.0) == 0.0


def test_finder_accepts_a_stable_weak_masked_candidate(monkeypatch):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    candidate = vision.Detection(
        False, 80, 60, 0.69, (30, 40), 0, (65, 40, 30, 40),
        0.80, 0.65, 0.40,
    )

    class Capture:
        def grab(self, _region):
            return frame

    bot = FishingBot.__new__(FishingBot)
    bot.capture = Capture()
    bot.templates = [vision.TemplateAsset(
        np.zeros((40, 30, 3), dtype=np.uint8),
        np.full((40, 30), 255, dtype=np.uint8),
    )]
    bot.stop_event = threading.Event()
    bot.debug = False
    bot.safe = lambda: True
    monkeypatch.setattr(vision, "adaptive_novelty", lambda *_args: np.zeros((120, 160), np.uint8))
    monkeypatch.setattr(vision, "match_template_candidates", lambda *_args, **_kwargs: [candidate])

    found = bot._find(Region(0, 0, 160, 120), [frame], time.monotonic() + 0.5)

    assert found.found
    assert found.confidence == 0.69
