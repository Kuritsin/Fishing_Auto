from config import Region
from fishing_bot import setup


class WindowSequence:
    def __init__(self):
        self.calls = 0

    def client_region(self):
        self.calls += 1
        return Region(10, 20, 800, 600) if self.calls >= 3 else None


def test_wait_for_wow_survives_temporary_focus_loss(monkeypatch):
    window = WindowSequence()
    monkeypatch.setattr(setup.time, "sleep", lambda _seconds: None)
    assert setup._wait_for_wow(window) == Region(10, 20, 800, 600)
    assert window.calls == 3