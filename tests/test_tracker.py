import threading

import numpy as np

from config import Region, Settings
from fishing_bot.tracker import BobberTracker
from fishing_bot.vision import Detection


class BlankCapture:
    def grab(self, region):
        return np.zeros((region.height, region.width, 3), dtype=np.uint8)


def test_tracker_tolerates_short_match_loss(monkeypatch):
    initial = Detection(True, 100, 100, 0.8, (30, 40), 0)
    missing = Detection(False, confidence=0.4)
    recovered = Detection(True, 101, 100, 0.7, (30, 40), 0)
    sequence = [missing] * 5 + [recovered] * 100

    def fake_match(*_args, **_kwargs):
        return sequence.pop(0) if sequence else recovered

    monkeypatch.setattr("fishing_bot.tracker.vision.match_templates", fake_match)
    settings = Settings(tracker_fps=50, tracker_lost_seconds=0.2,
                        bite_warmup_seconds=2.0)
    result = BobberTracker(BlankCapture(), settings).wait(
        initial, [np.zeros((40, 30, 3), dtype=np.uint8)],
        Region(0, 0, 300, 300), 0.18, threading.Event(), lambda: True,
    )
    assert result.reason == "timeout"


def test_tracker_rejects_an_implausible_single_frame_jump(monkeypatch):
    initial = Detection(True, 100, 100, 0.8, (30, 40), 0)
    jumped = Detection(True, 150, 180, 0.9, (30, 40), 0)
    stable = Detection(True, 100, 100, 0.8, (30, 40), 0)
    sequence = [jumped, stable] + [stable] * 100

    def fake_match(*_args, **_kwargs):
        return sequence.pop(0) if sequence else stable

    monkeypatch.setattr("fishing_bot.tracker.vision.match_templates", fake_match)
    settings = Settings(tracker_fps=50, tracker_lost_seconds=0.2,
                        bite_warmup_seconds=2.0)
    result = BobberTracker(BlankCapture(), settings).wait(
        initial, [np.zeros((40, 30, 3), dtype=np.uint8)],
        Region(0, 0, 300, 300), 0.12, threading.Event(), lambda: True,
    )
    assert not result.detected
    assert result.reason == "timeout"
