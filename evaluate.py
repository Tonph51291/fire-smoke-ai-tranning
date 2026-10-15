from __future__ import annotations

import argparse
import sys
from pathlib import Path

from check_dataset import audit_dataset, format_report


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
DEFAULT_WEIGHTS = ROOT / "runs" / "fire_smoke" / "experiment1" / "weights" / "best.pt"
CLASS_NAMES = {0: "fire", 1: "smoke"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá best.pt trên tập test độc lập."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Đường dẫn best.pt (mặc định: runs/fire_smoke/experiment1/weights/best.pt).",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Ví dụ: cpu, 0; để trống cho Ultralytics tự chọn.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = args.weights.expanduser()
    if not weights.is_absolute():
        weights = ROOT / weights
    weights = weights.resolve()
    if not weights.is_file():
        print(f"Không tìm thấy model đã huấn luyện: {weights}", file=sys.stderr)
        return 1

    try:
        report = audit_dataset(DATA_YAML)
        print(format_report(report))
        if not report.is_valid:
            print("Đã hủy đánh giá vì dataset chưa hợp lệ.", file=sys.stderr)
            return 1

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu Ultralytics. Cài dependencies bằng: "
                "venv\\Scripts\\python.exe -m pip install -r requirements.txt"
            ) from exc

        model = YOLO(str(weights))
        metrics = model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        box_metrics = metrics.box
        print("\nKết quả trên tập test:")
        print(f"Precision: {box_metrics.mp:.4f}")
        print(f"Recall: {box_metrics.mr:.4f}")
        print(f"mAP50: {box_metrics.map50:.4f}")
        print(f"mAP50-95: {box_metrics.map:.4f}")
        print("\nTheo lớp:")
        for position, class_id in enumerate(box_metrics.ap_class_index):
            class_index = int(class_id)
            precision, recall, ap50, ap = box_metrics.class_result(position)
            class_name = CLASS_NAMES.get(class_index, f"class_{class_index}")
            print(
                f"{class_name}: Precision={precision:.4f}, Recall={recall:.4f}, "
                f"mAP50={ap50:.4f}, mAP50-95={ap:.4f}"
            )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Lỗi đánh giá: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
