from __future__ import annotations

from collections import deque
from typing import Any

from fire_alert.config import AlertState, Detection


class TemporalFilter:
    """Loc ket qua theo thoi gian tren sliding window (K tren N khung hinh)."""

    def __init__(
        self,
        window_size: int = 10,
        min_hits: int = 5,
        target_classes: tuple[str, ...] = ("fire", "smoke"),
    ) -> None:
        self.window_size = window_size
        self.min_hits = min_hits
        self.target_classes = target_classes
        self._history: deque[set[str]] = deque(maxlen=window_size)

    def update(self, detections: list[Detection]) -> tuple[AlertState, dict[str, int], set[str]]:
        """Cap nhat ket qua tu khung hinh moi va xac dinh trang thai he thong.

        Returns:
            (trang_thai, so_lan_xuat_hien_tung_lop, tap_cac_lop_da_xac_nhan_bao_dong)
        """
        current_frame_classes = {
            det.class_name for det in detections if det.class_name in self.target_classes
        }
        self._history.append(current_frame_classes)

        class_hits = {
            cls_name: sum(cls_name in frame_set for frame_set in self._history)
            for cls_name in self.target_classes
        }

        confirmed_classes = {
            cls_name for cls_name, hits in class_hits.items() if hits >= self.min_hits
        }

        if confirmed_classes:
            state = AlertState.ALARM
        elif any(hits > 0 for hits in class_hits.values()):
            state = AlertState.SUSPICIOUS
        else:
            state = AlertState.NORMAL

        return state, class_hits, confirmed_classes

    def get_max_confirmed_confidence(
        self, detections: list[Detection], confirmed_classes: set[str]
    ) -> float:
        """Lay do tin cay cao nhat trong cac lop da duoc xac nhan bao dong."""
        confidences = [
            det.confidence for det in detections if det.class_name in confirmed_classes
        ]
        return max(confidences, default=0.0)

    def reset(self) -> None:
        """Xoa lich su sliding window."""
        self._history.clear()


class MotionFilter:
    """Bo loc chuyen dong / chop sang giua cac khung hinh (tuy chon)."""

    def __init__(self, enabled: bool = False, flicker_min: float = 0.0) -> None:
        self.enabled = enabled
        self.flicker_min = flicker_min
        self._prev_gray: Any = None

    def filter(self, frame: Any, detections: list[Detection]) -> list[Detection]:
        """Loc cac vung nhan dien dua tren chuyen dong/bien dong cuong do sang."""
        if not self.enabled or not detections:
            return detections

        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return detections

        filtered: list[Detection] = []
        diff = cv2.absdiff(gray, self._prev_gray)

        h, w = gray.shape
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            roi_diff = diff[y1:y2, x1:x2]
            motion_score = float(roi_diff.mean())
            if motion_score >= self.flicker_min:
                filtered.append(det)

        self._prev_gray = gray
        return filtered

    def reset(self) -> None:
        """Xoa khung hinh truoc do."""
        self._prev_gray = None
