from __future__ import annotations

import unittest
import numpy as np

from src.services.detection_service import Detection
from src.utils.video_utils import (
    cv2_bgr_to_pil,
    draw_detections,
    format_timestamp,
    resize_frame_to_fit,
)


class TestVideoUtils(unittest.TestCase):
    def test_format_timestamp(self) -> None:
        self.assertEqual(format_timestamp(0.0, show_millis=True), "00:00.0")
        self.assertEqual(format_timestamp(15.24, show_millis=True), "00:15.2")
        self.assertEqual(format_timestamp(65.0, show_millis=False), "01:05")
        self.assertEqual(format_timestamp(3665.5, show_millis=True), "01:01:05.5")
        self.assertEqual(format_timestamp(-5.0, show_millis=True), "00:00.0")

    def test_resize_frame_to_fit(self) -> None:
        # 1920x1080 -> fit into 960x540
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized, scale = resize_frame_to_fit(frame, 960, 540)
        self.assertEqual(resized.shape[1], 960)
        self.assertEqual(resized.shape[0], 540)
        self.assertAlmostEqual(scale, 0.5)

        # Aspect ratio preservation: 1000x500 into 500x500
        frame2 = np.zeros((500, 1000, 3), dtype=np.uint8)
        resized2, scale2 = resize_frame_to_fit(frame2, 500, 500)
        self.assertEqual(resized2.shape[1], 500)
        self.assertEqual(resized2.shape[0], 250)
        self.assertAlmostEqual(scale2, 0.5)

    def test_draw_detections(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [
            Detection(class_id=0, class_name="fire", confidence=0.885, bbox=(50, 50, 200, 200)),
            Detection(class_id=1, class_name="smoke", confidence=0.742, bbox=(250, 100, 400, 300)),
        ]
        annotated = draw_detections(frame, dets)
        self.assertEqual(annotated.shape, frame.shape)
        # Should not be identical to black frame because bounding boxes are drawn
        self.assertTrue(np.any(annotated > 0))

    def test_cv2_bgr_to_pil(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Pure Blue in BGR
        pil_img = cv2_bgr_to_pil(frame)
        self.assertEqual(pil_img.size, (100, 100))
        # In RGB, blue is at index 2
        r, g, b = pil_img.getpixel((0, 0))
        self.assertEqual(b, 255)
        self.assertEqual(r, 0)


if __name__ == "__main__":
    unittest.main()
