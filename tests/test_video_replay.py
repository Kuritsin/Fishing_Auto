from pathlib import Path

import pytest

from tools.replay_video import replay


CASES = (
    ("bobber_vid.mp4", 4.0, 12.0, 858, 622, (7.7, 8.4)),
    ("Bobber_vid_2.mp4", 7.0, 13.0, 890, 542, (8.5, 9.3)),
    ("Bobber_vid_2.mp4", 20.0, 30.0, 1035, 575, (27.8, 28.8)),
)


@pytest.mark.parametrize("name,start,end,x,y,expected", CASES)
def test_real_video_bite(name, start, end, x, y, expected, tmp_path):
    video = Path(name)
    if not video.exists():
        pytest.skip("private regression video is not present")
    events = replay(video, start, end, x, y, tmp_path / f"{name}.csv")
    assert len(events) == 1
    assert expected[0] <= events[0] <= expected[1]


def test_moving_water_without_bite_does_not_trigger(tmp_path):
    video = Path("bobber_vid_3.mp4")
    if not video.exists():
        pytest.skip("private moving-water regression video is not present")
    events = replay(video, 3.5, 6.5, 1210, 410, tmp_path / "moving-water.csv")
    assert events == []
