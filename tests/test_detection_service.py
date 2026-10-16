from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

from src.config import AppConfig, resolve_model_path
from src.services.detection_service import DetectionService, DetectionResult


class TestDetectionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AppConfig()
        cls.model_path = resolve_model_path()
        cls.service = DetectionService(
            model_path=cls.model_path,
            conf_threshold=0.25,
            device="cpu",
            imgsz=320,  # Dùng 320 để test nhanh trên CPU
        )

    def test_model_loaded_successfully(self) -> None:
        self.assertIsNotNone(self.service._model)
        self.assertIn(0, self.service.class_names)
        self.assertEqual(self.service.class_names.get(0), "fire")
        self.assertEqual(self.service.class_names.get(1), "smoke")

    def test_missing_model_raises_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            DetectionService(model_path="non_existent_weights_xyz.pt")

    def test_predict_blank_frame(self) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result: DetectionResult = self.service.predict(blank, frame_idx=1, timestamp_sec=0.04)
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.frame_idx, 1)
        self.assertAlmostEqual(result.timestamp_sec, 0.04)
        self.assertIsInstance(result.detections, list)

    def test_threshold_update(self) -> None:
        self.service.set_conf_threshold(0.75)
        self.assertAlmostEqual(self.service.conf_threshold, 0.75)
        self.service.set_conf_threshold(0.25)
        self.assertAlmostEqual(self.service.conf_threshold, 0.25)


if __name__ == "__main__":
    unittest.main()
