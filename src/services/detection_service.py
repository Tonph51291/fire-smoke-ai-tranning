from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Detection:
    """Thông tin một đối tượng lửa hoặc khói được phát hiện trong khung hình."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class DetectionResult:
    """Tổng hợp kết quả nhận diện trên một frame video."""
    detections: list[Detection]
    frame_idx: int
    timestamp_sec: float
    fire_detected: bool
    smoke_detected: bool
    max_fire_conf: float
    max_smoke_conf: float


class DetectionService:
    """Dịch vụ nạp mô hình YOLO và thực hiện suy luận nhận diện cháy và khói."""

    def __init__(
        self,
        model_path: Path | str,
        conf_threshold: float = 0.25,
        device: str | None = None,
        imgsz: int = 640,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.conf_threshold = float(conf_threshold)
        self.device = device
        self.imgsz = imgsz

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số mô hình: {self.model_path}\n"
                f"Hãy đảm bảo file best.pt đã được đặt đúng trong thư mục models/ hoặc runs/."
            )

        logger.info("Đang nạp mô hình YOLO từ: %s", self.model_path)
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài đặt thư viện ultralytics. Hãy cài đặt bằng:\n"
                "pip install -r requirements.txt"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Lỗi khi nạp mô hình YOLO: {exc}") from exc

        # Trích xuất và xác thực class names thực tế từ model
        self.class_names = self._extract_class_names()
        logger.info("Class names trong model: %s", self.class_names)

        # Kiểm tra xem model có hỗ trợ fire hoặc smoke không
        available_names = set(self.class_names.values())
        if not ("fire" in available_names or "smoke" in available_names):
            logger.warning(
                "Model hiện tại không chứa các class 'fire' hoặc 'smoke'! "
                "Các class có trong model: %s",
                available_names,
            )

    def _extract_class_names(self) -> dict[int, str]:
        """Trích xuất dictionary ánh xạ class_id -> class_name từ model."""
        raw_names = getattr(self._model, "names", {})
        if isinstance(raw_names, dict):
            return {int(k): str(v).strip().lower() for k, v in raw_names.items()}
        if isinstance(raw_names, (list, tuple)):
            return {i: str(name).strip().lower() for i, name in enumerate(raw_names)}
        return {}

    def set_conf_threshold(self, conf: float) -> None:
        """Cập nhật ngưỡng confidence trong quá trình chạy."""
        self.conf_threshold = max(0.01, min(0.99, float(conf)))

    def predict(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp_sec: float = 0.0,
    ) -> DetectionResult:
        """Thực hiện suy luận trên một frame ảnh BGR.
        
        Trả về DetectionResult chứa danh sách detection và các thông số tổng hợp.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(
                detections=[],
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                fire_detected=False,
                smoke_detected=False,
                max_fire_conf=0.0,
                max_smoke_conf=0.0,
            )

        predict_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.conf_threshold,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz

        try:
            results = self._model.predict(**predict_kwargs)
        except Exception as exc:
            logger.error("Lỗi suy luận YOLO: %s", exc)
            return DetectionResult(
                detections=[],
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                fire_detected=False,
                smoke_detected=False,
                max_fire_conf=0.0,
                max_smoke_conf=0.0,
            )

        detections: list[Detection] = []
        max_fire_conf = 0.0
        max_smoke_conf = 0.0

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                xyxy = box.xyxy[0].tolist()
                bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                    )
                )

                if class_name == "fire":
                    max_fire_conf = max(max_fire_conf, confidence)
                elif class_name == "smoke":
                    max_smoke_conf = max(max_smoke_conf, confidence)

        return DetectionResult(
            detections=detections,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            fire_detected=(max_fire_conf > 0.0),
            smoke_detected=(max_smoke_conf > 0.0),
            max_fire_conf=max_fire_conf,
            max_smoke_conf=max_smoke_conf,
        )
