
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from check_dataset import audit_dataset, format_report


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
RUNS_DIR = ROOT / "runs" / "fire_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv8 nhận diện fire và smoke.")
    parser.add_argument("--weights", default="yolov8n.pt", help="File trọng số pretrained có sẵn trên máy.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="Ví dụ: cpu, 0; để trống cho Ultralytics tự chọn.")
    parser.add_argument("--name", default="fire_v1", help="Tên thư mục của lần huấn luyện.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        report = audit_dataset(DATA_YAML)
        print(format_report(report))
        if not report.is_valid:
            print("Đã hủy huấn luyện vì dataset chưa hợp lệ.", file=sys.stderr)
            return 1

        weights = Path(args.weights).expanduser()
        if not weights.is_absolute():
            weights = ROOT / weights
        weights = weights.resolve()
        if not weights.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy pretrained weights: {weights}. "
                "Hãy đặt file yolov8n.pt trong thư mục project hoặc truyền --weights "
                "với đường dẫn tới file .pt hợp lệ. Script không tự tải model."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu Ultralytics. Cài dependencies bằng: "
                "venv\\Scripts\\python.exe -m pip install -r requirements.txt"
            ) from exc

        model = YOLO(str(weights))
        model.train(
            data=str(DATA_YAML),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            workers=args.workers,
            project=str(RUNS_DIR),
            name=args.name,
            exist_ok=True,
            plots=True,
            device=args.device,
        )

        best_path = Path(model.trainer.best).resolve()
        if not best_path.is_file():
            raise RuntimeError(
                f"Huấn luyện đã kết thúc nhưng không thấy best.pt tại {best_path}."
            )
        print(f"Huấn luyện hoàn tất. Best weights: {best_path}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Lỗi huấn luyện: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
