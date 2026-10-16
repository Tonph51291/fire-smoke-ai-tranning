from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import tkinter as tk

from src.config import AppConfig
from src.services.detection_service import Detection, DetectionResult
from src.ui.main_window import MainWindow


class TestAlertLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = tk.Tk()
        cls.root.withdraw()  # Ẩn cửa sổ khi test

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.destroy()

    def setUp(self) -> None:
        self.config = AppConfig()
        self.mock_det_service = MagicMock()
        self.mock_vid_service = MagicMock()
        self.mock_vid_service.frame_queue = MagicMock()
        self.mock_vid_service.frame_queue.empty.return_value = True

        self.window = MainWindow(
            root=self.root,
            config=self.config,
            detection_service=self.mock_det_service,
            video_service=self.mock_vid_service,
        )

    def test_state_1_no_detection(self) -> None:
        res = DetectionResult(
            detections=[],
            frame_idx=1,
            timestamp_sec=0.1,
            fire_detected=False,
            smoke_detected=False,
            max_fire_conf=0.0,
            max_smoke_conf=0.0,
        )
        self.window._update_alert_status(res)
        self.assertEqual(self.window._last_alert_kind, "none")
        self.assertIn("CHƯA PHÁT HIỆN", self.window.alert_title.cget("text"))

    def test_state_2_fire_detection(self) -> None:
        res = DetectionResult(
            detections=[Detection(0, "fire", 0.91, (10, 10, 100, 100))],
            frame_idx=2,
            timestamp_sec=0.2,
            fire_detected=True,
            smoke_detected=False,
            max_fire_conf=0.91,
            max_smoke_conf=0.0,
        )
        self.window._update_alert_status(res)
        self.assertEqual(self.window._last_alert_kind, "fire")
        self.assertIn("PHÁT HIỆN LỬA", self.window.alert_title.cget("text"))
        self.assertIn("91.0%", self.window.alert_details.cget("text"))

    def test_state_3_smoke_detection(self) -> None:
        res = DetectionResult(
            detections=[Detection(1, "smoke", 0.78, (10, 10, 100, 100))],
            frame_idx=3,
            timestamp_sec=0.3,
            fire_detected=False,
            smoke_detected=True,
            max_fire_conf=0.0,
            max_smoke_conf=0.78,
        )
        self.window._update_alert_status(res)
        self.assertEqual(self.window._last_alert_kind, "smoke")
        self.assertIn("PHÁT HIỆN KHÓI", self.window.alert_title.cget("text"))
        self.assertIn("78.0%", self.window.alert_details.cget("text"))

    def test_state_4_both_fire_and_smoke(self) -> None:
        res = DetectionResult(
            detections=[
                Detection(0, "fire", 0.89, (10, 10, 100, 100)),
                Detection(1, "smoke", 0.72, (150, 150, 200, 200)),
            ],
            frame_idx=4,
            timestamp_sec=0.4,
            fire_detected=True,
            smoke_detected=True,
            max_fire_conf=0.89,
            max_smoke_conf=0.72,
        )
        self.window._update_alert_status(res)
        self.assertEqual(self.window._last_alert_kind, "fire_and_smoke")
        self.assertIn("PHÁT HIỆN LỬA VÀ KHÓI", self.window.alert_title.cget("text"))
        self.assertIn("89.0%", self.window.alert_details.cget("text"))
        self.assertIn("72.0%", self.window.alert_details.cget("text"))


if __name__ == "__main__":
    unittest.main()
