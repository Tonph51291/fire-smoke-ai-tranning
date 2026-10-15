from __future__ import annotations

from typing import Any
import time

from fire_alert.config import AlertState


class Display:
    """Ve cac thong tin canh bao, hop nhan dien va quan ly cua so OpenCV."""

    def __init__(
        self,
        window_name: str = "Fire and Smoke Detection - Tecnix AI",
        state_colors: dict[AlertState, tuple[int, int, int]] | None = None,
        window_size: int = 10,
    ) -> None:
        self.window_name = window_name
        self.window_size = window_size
        self.state_colors = state_colors or {
            AlertState.NORMAL: (0, 200, 0),
            AlertState.SUSPICIOUS: (0, 165, 255),
            AlertState.ALARM: (0, 0, 255),
        }

    def render(
        self,
        frame: Any,
        raw_result: Any,
        state: AlertState,
        detections_text: str,
        confirmed_classes: set[str],
        class_hits: dict[str, int],
        fps: float,
        sound_enabled: bool,
    ) -> Any:
        """Ve bounding box, banner canh bao, vien nhap nhay va bang thong tin."""
        import cv2

        # 1. Ve bounding box mac dinh tu mo hinh
        if raw_result is not None:
            display_frame = raw_result.plot(img=frame.copy())
        else:
            display_frame = frame.copy()

        now = time.monotonic()
        status_color = self.state_colors.get(state, (0, 200, 0))

        # 2. Ve thong tin trang thai va HUD tren man hinh
        max_hits = max(class_hits.values()) if class_hits else 0

        cv2.putText(
            display_frame,
            f"TRANG THAI: {state.value}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_frame,
            f"PHAT HIEN: {detections_text}",
            (20, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f} | XAC NHAN: {max_hits}/{self.window_size}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_frame,
            f"AM THANH: {'BAT' if sound_enabled else 'TAT'} (M)",
            (20, 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255) if sound_enabled else (160, 160, 160),
            2,
            cv2.LINE_AA,
        )

        # 3. Xu ly khi o trang thai BAO DONG
        if state == AlertState.ALARM:
            if confirmed_classes == {"fire"}:
                banner_text = "CANH BAO CHAY! PHAT HIEN LUA"
            elif confirmed_classes == {"smoke"}:
                banner_text = "CANH BAO! PHAT HIEN KHOI"
            else:
                banner_text = "CANH BAO! LUA VA KHOI"

            # Banner do tren dinh
            cv2.rectangle(
                display_frame,
                (0, 0),
                (display_frame.shape[1] - 1, 50),
                (0, 0, 220),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                display_frame,
                banner_text,
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            # Vien do nhap nhay canh bao chu ky 0.5s
            if now % 0.5 < 0.25:
                cv2.rectangle(
                    display_frame,
                    (4, 4),
                    (display_frame.shape[1] - 5, display_frame.shape[0] - 5),
                    (0, 0, 255),
                    thickness=6,
                )

        return display_frame

    def show(self, frame: Any) -> None:
        """Hien thi khung hinh len cua so OpenCV."""
        import cv2

        cv2.imshow(self.window_name, frame)

    def wait_key(self, delay: int = 1) -> int:
        """Cho phim nhan tu ban phim."""
        import cv2

        return cv2.waitKey(delay) & 0xFF

    def close(self) -> None:
        """Dong toan bo cua so OpenCV."""
        import cv2

        cv2.destroyAllWindows()
