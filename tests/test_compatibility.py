import pytest

from fishing_bot import bot


def test_mixed_vision_version_fails_before_the_first_cast(monkeypatch):
    monkeypatch.setattr(bot.vision, "VISION_API_VERSION", 1)
    with pytest.raises(RuntimeError, match="разные версии"):
        bot.validate_vision_api()


def test_current_vision_version_is_accepted():
    bot.validate_vision_api()
