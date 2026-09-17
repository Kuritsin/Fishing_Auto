from fishing_bot.bot import remaining_cast_time


def test_finder_time_is_part_of_the_cast_lifetime():
    assert remaining_cast_time(100.0, 103.5, 22.0) == 18.5


def test_shorter_expansion_cast_can_finish_tracking_early():
    assert remaining_cast_time(100.0, 123.0, 22.0) == 0.0
