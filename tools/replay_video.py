from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from fishing_bot.bite import BiteSignalDetector
from config import Region
from fishing_bot.vision import match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline bobber trajectory replay")
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--x", type=int, required=True, help="bobber centre on the start frame")
    parser.add_argument("--y", type=int, required=True, help="bobber centre on the start frame")
    parser.add_argument("--output", type=Path, default=Path("debug/replay.csv"))
    return parser.parse_args()


def replay(video: Path, start: float, end: float, x: int, y: int,
           output: Path) -> list[float]:
    stream = cv2.VideoCapture(str(video))
    if not stream.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    stream.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    ok, first = stream.read()
    if not ok:
        raise RuntimeError("Cannot read the template frame")
    radius_x, radius_y = 40, 35
    template = first[y - radius_y:y + radius_y, x - radius_x:x + radius_x]
    if template.size == 0:
        raise ValueError("Template coordinates are outside the video")
    detector = BiteSignalDetector(template.shape[0])
    events: list[float] = []
    rows = []
    last_x, last_y = x, y
    while True:
        ok, frame = stream.read()
        if not ok:
            break
        timestamp = stream.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if timestamp > end:
            break
        search_radius = 120
        left, top = max(0, last_x - search_radius), max(0, last_y - search_radius)
        right = min(frame.shape[1], last_x + search_radius)
        bottom = min(frame.shape[0], last_y + search_radius)
        region = Region(left, top, right - left, bottom - top)
        found = match(frame[top:bottom, left:right], template, region, 0.48, (1.0,))
        signal = detector.update(timestamp, found.x if found.found else None,
                                 found.y if found.found else None)
        if found.found:
            last_x, last_y = found.x, found.y
        if signal.detected and (not events or timestamp - events[-1] > 1):
            events.append(timestamp)
        rows.append((timestamp, found.x, found.y, found.confidence,
                     signal.drop, signal.velocity, signal.detected))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("seconds", "x", "y", "confidence", "drop", "velocity", "bite"))
        writer.writerows(rows)
    return events


def main() -> int:
    args = parse_args()
    events = replay(args.video, args.start, args.end, args.x, args.y, args.output)
    print("Bite events:", ", ".join(f"{event:.3f}s" for event in events) or "none")
    print("Trajectory:", args.output)
    return 0 if events else 1


if __name__ == "__main__":
    raise SystemExit(main())
