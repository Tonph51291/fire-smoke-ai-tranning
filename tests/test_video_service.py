from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig, resolve_model_path
from src.services.detection_service import DetectionService
from src.services.video_service import VideoPlaybackState, VideoService


def create_dummy_video(path: Path, num_frames: int = 15, fps: float = 10.0) -> None:
    """Tạo một file video giả lập để kiểm thử."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for i in range(num_frames):
        frame = np.full((240, 320, 3), fill_value=i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestVideoService(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AppConfig()
        cls.detection_service = DetectionService(
            model_path=resolve_model_path(),
            conf_threshold=0.25,
            device="cpu",
            imgsz=320,
        )
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.dummy_video_path = Path(cls.temp_dir.name) / "test_sample.mp4"
        create_dummy_video(cls.dummy_video_path, num_frames=15, fps=10.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        self.service = VideoService(
            detection_service=self.detection_service,
            config=self.config,
        )

    def tearDown(self) -> None:
        self.service.release()

    def test_open_invalid_video(self) -> None:
        ok, msg = self.service.open_video(Path("non_existent_file.mp4"))
        self.assertFalse(ok)
        self.assertIn("Không tìm thấy", msg)

    def test_open_unsupported_format(self) -> None:
        dummy_txt = Path(self.temp_dir.name) / "test.txt"
        dummy_txt.write_text("hello")
        ok, msg = self.service.open_video(dummy_txt)
        self.assertFalse(ok)
        self.assertIn("không được hỗ trợ", msg)

    def test_open_valid_video(self) -> None:
        ok, msg = self.service.open_video(self.dummy_video_path)
        self.assertTrue(ok)
        self.assertEqual(self.service.state, VideoPlaybackState.READY)
        meta = self.service.get_metadata()
        self.assertIsNotNone(meta)
        self.assertEqual(meta.width, 320)
        self.assertEqual(meta.height, 240)
        self.assertEqual(meta.total_frames, 15)

    def test_playback_lifecycle(self) -> None:
        ok, _ = self.service.open_video(self.dummy_video_path)
        self.assertTrue(ok)

        # 1. Phát video
        self.service.play()
        self.assertEqual(self.service.state, VideoPlaybackState.PLAYING)

        # Đợi một chút để worker thread đọc và phân tích frame
        time.sleep(0.3)
        self.assertFalse(self.service.frame_queue.empty())
        data = self.service.frame_queue.get_nowait()
        self.assertIsNotNone(data.frame)
        self.assertGreaterEqual(data.current_frame, 1)

        # 2. Tạm dừng
        self.service.pause()
        self.assertEqual(self.service.state, VideoPlaybackState.PAUSED)

        # 3. Tiếp tục
        self.service.resume()
        self.assertEqual(self.service.state, VideoPlaybackState.PLAYING)

        # 4. Dừng
        self.service.stop()
        self.assertEqual(self.service.state, VideoPlaybackState.STOPPED)


if __name__ == "__main__":
    unittest.main()
