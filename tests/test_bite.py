from fishing_bot.bite import BiteSignalDetector


def feed(detector, positions, fps=20):
    return [detector.update(index / fps, x, y)
            for index, (x, y) in enumerate(positions)]


def test_horizontal_drift_and_small_bobbing_are_not_a_bite():
    normal = [(100 + index, 200 + (index % 3 - 1)) for index in range(30)]
    assert not any(result.detected for result in feed(BiteSignalDetector(50), normal))


def test_fast_downward_dive_is_a_bite():
    positions = [(100, 200)] * 12 + [(100, 203), (100, 210), (100, 216)]
    results = feed(BiteSignalDetector(50), positions)
    assert any(result.detected and result.reason == "rapid_downward_motion" for result in results)


def test_disappearance_after_started_drop_is_a_bite():
    detector = BiteSignalDetector(50, confirmations=3)
    feed(detector, [(100, 200)] * 12 + [(100, 204), (100, 211)])
    result = detector.update(0.75, None, None)
    assert result.detected
    assert result.reason == "submerged_after_drop"
