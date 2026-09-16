from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from config import Region


class ScreenCapture:
    def __init__(self) -> None:
        import mss

        self._capture: Any = mss.mss()

    def grab(self, region: Region) -> np.ndarray:
        image = np.asarray(self._capture.grab(region.as_mss()))
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._capture.close()
