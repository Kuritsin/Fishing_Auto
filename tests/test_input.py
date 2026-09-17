import math

from fishing_bot.input import cursor_path


def test_cursor_path_is_curved_but_finishes_exactly():
    points = cursor_path((10, 20), (310, 180), 20)
    assert len(points) == 20
    assert points[-1] == (310, 180)
    # A curved path has at least one point away from the straight segment.
    start, end = (10, 20), (310, 180)
    deviations = []
    for x, y in points[:-1]:
        numerator = abs((end[1] - start[1]) * x - (end[0] - start[0]) * y +
                        end[0] * start[1] - end[1] * start[0])
        deviations.append(numerator / math.hypot(end[1] - start[1], end[0] - start[0]))
    assert max(deviations) > 0.5
    assert max(deviations) <= 20


def test_cursor_path_handles_zero_distance():
    assert cursor_path((50, 60), (50, 60), 10) == [(50, 60)]