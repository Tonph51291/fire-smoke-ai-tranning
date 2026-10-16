from __future__ import annotations

import argparse
import logging
import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Đảm bảo mã hóa UTF-8 trên Windows console
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import AppConfig, resolve_model_path
from src.services.detection_service import DetectionService
from src.services.video_service import VideoService
from src.ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fire_smoke_app")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo AI phát hiện cháy và khói từ video camera."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Đường dẫn file trọng số YOLO (.pt). Mặc định tự động tìm best.pt.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Ngưỡng tin cậy confidence threshold (0.05 - 0.95). Mặc định 0.25.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Thiết bị tính toán: 'cpu', '0' (CUDA GPU). Mặc định tự chọn.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Đường dẫn file video mở sẵn khi khởi động (tùy chọn).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Khởi tạo cấu hình ứng dụng
    config = AppConfig()
    config.model_path = resolve_model_path(args.model)
    config.conf_threshold = max(0.01, min(0.99, args.conf))
    config.device = args.device

    # 2. Khởi tạo Tkinter root trước để có thể hiển thị hộp thoại lỗi nếu cần
    root = tk.Tk()

    # 3. Khởi tạo dịch vụ nhận diện AI
    try:
        detection_service = DetectionService(
            model_path=config.model_path,
            conf_threshold=config.conf_threshold,
            device=config.device,
            imgsz=config.imgsz,
        )
    except FileNotFoundError as err:
        logger.error("Không tìm thấy model: %s", err)
        messagebox.showerror(
            "Thiếu Model Weights",
            f"{err}\n\nVui lòng đặt file best.pt vào thư mục models/ rồi thử lại.",
            parent=root,
        )
        root.destroy()
        return 1
    except Exception as err:
        logger.error("Lỗi khởi tạo DetectionService: %s", err)
        messagebox.showerror(
            "Lỗi khởi tạo mô hình",
            f"Không thể khởi tạo mô hình nhận diện:\n{err}",
            parent=root,
        )
        root.destroy()
        return 1

    # 4. Khởi tạo dịch vụ video
    video_service = VideoService(
        detection_service=detection_service,
        config=config,
    )

    # 5. Khởi tạo giao diện chính
    app = MainWindow(
        root=root,
        config=config,
        detection_service=detection_service,
        video_service=video_service,
    )

    # 6. Mở video truyền qua tham số dòng lệnh nếu có
    if args.video and args.video.is_file():
        ok, msg = video_service.open_video(args.video)
        if ok:
            meta = video_service.get_metadata()
            if meta:
                app.video_name_label.config(text=f"{meta.filename} ({meta.duration_sec:.1f}s)")
                app.status_label.config(text=msg, fg="#22c55e")
                app.btn_play.config(state=tk.NORMAL)
                app.btn_stop.config(state=tk.NORMAL)

    logger.info("Khởi chạy giao diện chính...")
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
