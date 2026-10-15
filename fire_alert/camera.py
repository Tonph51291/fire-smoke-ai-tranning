from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class Camera:
    """Quan ly luong camera (Webcam hoac RTSP) voi co che ket noi lai tu dong."""

    def __init__(
        self,
        source: int | str = 0,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.source = int(source) if str(source).isdecimal() else str(source)
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.width = width
        self.height = height
        self._capture: Any = None

    def open(self) -> bool:
        """Mo camera/stream voi so lan thu lai da cau hinh."""
        import cv2

        for attempt in range(self.reconnect_attempts + 1):
            capture = cv2.VideoCapture(self.source)
            if capture.isOpened():
                if self.width is not None:
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                if self.height is not None:
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._capture = capture
                return True

            capture.release()
            if attempt < self.reconnect_attempts:
                logger.warning(
                    "Khong the mo camera/stream (lan thu %d/%d); thu lai sau %g giay.",
                    attempt + 1,
                    self.reconnect_attempts + 1,
                    self.reconnect_delay,
                )
                time.sleep(self.reconnect_delay)

        logger.error("Khong the mo camera/stream sau %d lan thu.", self.reconnect_attempts + 1)
        return False

    def read(self) -> tuple[bool, Any]:
        """Doc mot khung hinh tu camera. Tu dong ket noi lai neu mat frame."""
        if self._capture is None:
            return False, None

        success, frame = self._capture.read()
        if not success:
            logger.warning("Mat ket noi camera; dang thu ket noi lai...")
            self.release()
            if self.open():
                return self._capture.read()
            return False, None

        return True, frame

    def release(self) -> None:
        """Giai phong tai nguyen camera."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as error:
                logger.warning("Loi khi dong camera: %s", error)
            finally:
                self._capture = None

    def is_opened(self) -> bool:
        """Kiem tra xem camera co dang mo hay khong."""
        return self._capture is not None and self._capture.isOpened()

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
