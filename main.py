from __future__ import annotations

import argparse
import logging

from config import Profile, SETTINGS
from fishing_bot.bot import FishingBot
from fishing_bot.capture import ScreenCapture
from fishing_bot.setup import run_setup
from fishing_bot.window import WowWindow, enable_dpi_awareness


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Простой визуальный WoW fishing bot")
    parser.add_argument("--setup", action="store_true", help="повторить настройку")
    parser.add_argument("--dry-run", action="store_true", help="запретить клавиатуру и мышь")
    parser.add_argument("--once", action="store_true", help="выполнить одну попытку")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    enable_dpi_awareness()
    window, capture = WowWindow(SETTINGS.window_titles), ScreenCapture()
    try:
        profile = Profile.load()
        if args.setup or profile is None:
            profile = run_setup(window, capture)
        bot = FishingBot(profile, window, capture, args.dry_run)
        from pynput import keyboard
        hotkeys = keyboard.GlobalHotKeys({f"<{profile.pause_key}>": bot.toggle_pause,
                                          f"<{profile.stop_key}>": bot.stop})
        hotkeys.start()
        try:
            bot.run(args.once)
        except KeyboardInterrupt:
            bot.stop()
        finally:
            hotkeys.stop(); hotkeys.join(timeout=1)
        return 0
    finally:
        capture.close()


if __name__ == "__main__":
    raise SystemExit(main())
