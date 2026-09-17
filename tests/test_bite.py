from fishing_bot.bite import BiteSignalDetector


def feed(detector, positions, fps=20):
    return [
        detector.update(index / fps, x, y)
        for index, (x, y) in enumerate(positions)
    ]


def test_horizontal_drift_and_small_bobbing_are_not_a_bite():
    normal = [
        (
            100 + index,
            200 + (index % 3 - 1),
        )
        for index in range(30)
    ]

    assert not any(
        result.detected
        for result in feed(
            BiteSignalDetector(50),
            normal,
        )
    )


def test_fast_downward_dive_is_a_bite():
    positions = (
        [(100, 200)] * 24
        + [
            (100, 203),
            (100, 210),
            (100, 216),
        ]
    )

    results = feed(
        BiteSignalDetector(50),
        positions,
    )

    assert any(
        result.detected
        and result.reason == "rapid_downward_motion"
        for result in results
    )


def test_disappearance_after_started_drop_is_a_bite():
    detector = BiteSignalDetector(
        50,
        confirmations=3,
    )

    feed(
        detector,
        [(100, 200)] * 24
        + [
            (100, 204),
            (100, 211),
        ],
    )

    result = detector.update(
        1.30,
        None,
        None,
    )

    assert result.detected
    assert result.reason == "submerged_after_drop"


def test_large_early_jump_is_not_a_bite_before_baseline_is_ready():
    positions = (
        [(100, 200)] * 8
        + [
            (100, 260),
            (100, 270),
            (100, 280),
        ]
    )

    assert not any(
        result.detected
        for result in feed(
            BiteSignalDetector(50),
            positions,
        )
    )


def test_single_strong_dive_is_not_lost_between_video_frames():
    detector = BiteSignalDetector(42)

    results = feed(
        detector,
        [(100, 200)] * 24
        + [
            (103, 228),
        ],
    )

    assert results[-1].detected
    assert results[-1].reason == "strong_downward_motion"


def test_noisy_water_raises_adaptive_drop_threshold():
    detector = BiteSignalDetector(42)

    noisy = [
        (
            100,
            200 + offset,
        )
        for offset in (
            0,
            3,
            -3,
            2,
            -2,
        )
        * 6
    ]

    results = feed(
        detector,
        noisy
        + [
            (100, 206),
            (100, 207),
        ],
    )

    assert not any(
        result.detected
        for result in results
    )