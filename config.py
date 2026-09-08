from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "profile.json"


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        return asdict(self)

    def inset(self, x_fraction: float, top_fraction: float, bottom_fraction: float) -> "Region":
        x = round(self.width * x_fraction)
        top = round(self.height * top_fraction)
        bottom = round(self.height * bottom_fraction)
        return Region(self.left + x, self.top + top, self.width - 2 * x, self.height - top - bottom)


@dataclass
class Profile:
    cast_key: str = "0"
    pause_key: str = "f8"
    stop_key: str = "f10"
    template_file: str = "data/bobber.png"
    template_width_ratio: float = 0.0
    template_height_ratio: float = 0.0
    schema_version: int = 1

    def save(self, path: Path = PROFILE_PATH) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path = PROFILE_PATH) -> "Profile | None":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            profile = cls(**data)
            if profile.schema_version != 1 or not profile.cast_key.strip():
                return None
            return profile
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None


@dataclass(frozen=True)
class Settings:
    window_titles: tuple[str, ...] = ("world of warcraft",)
    cast_settle_seconds: float = 1.25
    find_timeout: float = 2.5
    attempt_timeout: float = 22.0
    tracker_fps: float = 24.0
    match_confidence: float = 0.62
    strong_match_confidence: float = 0.78
    bite_drop_height_ratio: float = 0.12
    bite_velocity_height_ratio: float = 0.55
    bite_confirmation_frames: int = 2
    inactive_delay: float = 0.4
    retry_delay: float = 0.7
    post_loot_delay: float = 0.8


SETTINGS = Settings()
