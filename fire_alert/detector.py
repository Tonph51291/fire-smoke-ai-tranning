from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fire_alert.config import Detection

logger = logging.getLogger(__name__)


class Detector:
    """Nap mo hinh YOLO va thuc hien suy luan phat hien lua/khoi."""

    def __init__(
        self,
        model_path: Path | str,
        conf_threshold: float = 0.25,
        device: str | None = None,
        imgsz: int | tuple[int, int] | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz = imgsz

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Khong tim thay file trong so mo hinh: {self.model_path}"
            )

        logger.info("Dang tai mo hinh AI: %s ...", self.model_path)
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))

    def _get_class_name(self, class_id: int) -> str:
        """Lay ten lop chuan hoa tu model.names."""
        names = self.model.names
        if isinstance(names, dict):
            name = names.get(class_id, "")
        elif isinstance(names, (list, tuple)) and class_id < len(names):
            name = names[class_id]
        else:
            name = str(class_id)
        return str(name).strip().lower()

    def predict(self, frame: Any) -> tuple[list[Detection], Any]:
        """Chay du doan tren mot khung hinh, tra ve danh sach Detection va ket qua goc."""
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.conf_threshold,
            "device": self.device,
            "verbose": False,
        }
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz

        results = self.model.predict(**kwargs)
        raw_result = results[0]

        detections: list[Detection] = []
        if raw_result.boxes is not None:
            for box in raw_result.boxes:
                class_id = int(box.cls[0])
                class_name = self._get_class_name(class_id)
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                    )
                )

        return detections, raw_result
