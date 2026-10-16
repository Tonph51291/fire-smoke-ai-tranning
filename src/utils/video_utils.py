from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.detection_service import Detection


def format_timestamp(seconds: float, show_millis: bool = True) -> str:
    """Định dạng số giây video thành chuỗi hiển thị MM:SS hoặc MM:SS.S."""
    if seconds < 0:
        seconds = 0.0

    total_seconds = int(seconds)
    millis = int((seconds - total_seconds) * 10)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    hours = minutes // 60
    minutes = minutes % 60

    if hours > 0:
        time_str = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        time_str = f"{minutes:02d}:{secs:02d}"

    if show_millis:
        return f"{time_str}.{millis:01d}"
    return time_str


def resize_frame_to_fit(
    frame: np.ndarray, max_w: int, max_h: int
) -> tuple[np.ndarray, float]:
    """Thay đổi kích thước khung hình giữ nguyên tỷ lệ khung hình (aspect ratio)."""
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0 or max_w <= 0 or max_h <= 0:
        return frame, 1.0

    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    if scale < 1.0:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    return resized, scale


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    class_colors_bgr: dict[str, tuple[int, int, int]] | None = None,
    display_names: dict[str, str] | None = None,
) -> np.ndarray:
    """Vẽ bounding box, nhãn và confidence lên frame."""
    if not detections:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    colors = class_colors_bgr or {
        "fire": (0, 69, 255),
        "smoke": (210, 210, 210),
    }
    names = display_names or {
        "fire": "LỬA",
        "smoke": "KHÓI",
    }

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        # Giới hạn tọa độ trong kích thước ảnh
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        color = colors.get(det.class_name, (0, 255, 0))
        label_name = names.get(det.class_name, det.class_name.upper())
        label_text = f"{label_name} {det.confidence * 100:.1f}%"

        # Vẽ bounding box dày 2px
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Tính kích thước nền nhãn chữ
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )

        label_y1 = max(0, y1 - text_h - 8)
        label_y2 = y1
        label_x2 = min(w, x1 + text_w + 8)

        # Nền nhãn trùng màu lớp
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (label_x2, label_y2),
            color,
            thickness=cv2.FILLED,
        )

        # Chữ nhãn (màu trắng hoặc đen tùy độ sáng màu nền)
        brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299)
        text_color = (0, 0, 0) if brightness > 160 else (255, 255, 255)

        cv2.putText(
            annotated,
            label_text,
            (x1 + 4, label_y2 - 4),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def cv2_bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """Chuyển đổi mảng numpy BGR của OpenCV sang đối tượng PIL Image (RGB)."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)
