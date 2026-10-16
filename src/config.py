from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def resolve_model_path(custom_path: Path | str | None = None) -> Path:
    """Tự động tìm kiếm file trọng số model YOLO theo thứ tự ưu tiên."""
    if custom_path:
        path = Path(custom_path).expanduser()
        if not path.is_absolute():
            path = ROOT_DIR / path
        if path.is_file():
            return path.resolve()

    candidates = [
        MODELS_DIR / "best.pt",
        ROOT_DIR / "runs" / "fire_smoke" / "fire_v1" / "weights" / "best.pt",
        ROOT_DIR / "runs" / "fire_smoke" / "experiment1" / "weights" / "best.pt",
        ROOT_DIR / "yolov8n.pt",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    # Tìm bất kỳ file best.pt nào trong runs/
    for candidate in ROOT_DIR.glob("runs/**/best.pt"):
        if candidate.is_file():
            return candidate.resolve()

    return (MODELS_DIR / "best.pt").resolve()


@dataclass
class AppConfig:
    """Cấu hình chính của ứng dụng demo phát hiện cháy và khói."""

    # Đường dẫn model và tham số suy luận
    model_path: Path = field(default_factory=resolve_model_path)
    conf_threshold: float = 0.25
    imgsz: int = 640
    device: str | None = None

    # Danh mục định dạng video hỗ trợ
    supported_formats: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

    # Tên class chuẩn trong model
    target_classes: tuple[str, ...] = ("fire", "smoke")

    # Tên hiển thị tiếng Việt trên giao diện
    class_display_names: dict[str, str] = field(
        default_factory=lambda: {
            "fire": "LỬA",
            "smoke": "KHÓI",
        }
    )

    # Màu sắc bounding box (định dạng BGR cho OpenCV)
    # Lửa: Đỏ cam (0, 69, 255), Khói: Xám khói (210, 210, 210)
    class_colors_bgr: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "fire": (0, 69, 255),
            "smoke": (210, 210, 210),
        }
    )

    # Số frame duy trì cảnh báo để chống nhấp nháy (flickering) khi detection ngắt quãng
    alert_hold_frames: int = 10

    # Tần suất suy luận (1 = mỗi frame, 2 = 1 frame suy luận / 1 frame bỏ qua để tăng tốc CPU)
    inference_interval: int = 1
