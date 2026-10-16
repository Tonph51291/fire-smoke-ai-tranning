from __future__ import annotations

import unittest
from pathlib import Path

from src.config import AppConfig, resolve_model_path


class TestAppConfig(unittest.TestCase):
    def test_default_config(self) -> None:
        cfg = AppConfig()
        self.assertIsNotNone(cfg.model_path)
        self.assertTrue(cfg.model_path.is_file(), f"Model path {cfg.model_path} should exist")
        self.assertAlmostEqual(cfg.conf_threshold, 0.25)
        self.assertEqual(cfg.imgsz, 640)
        self.assertIn("fire", cfg.target_classes)
        self.assertIn("smoke", cfg.target_classes)
        self.assertIn(".mp4", cfg.supported_formats)

    def test_resolve_model_path_custom_existing(self) -> None:
        cfg = AppConfig()
        resolved = resolve_model_path(cfg.model_path)
        self.assertEqual(resolved, cfg.model_path.resolve())

    def test_resolve_model_path_fallback(self) -> None:
        resolved = resolve_model_path(Path("non_existent_model_123.pt"))
        # Should fall back to best.pt or yolov8n.pt
        self.assertTrue(resolved.is_file())


if __name__ == "__main__":
    unittest.main()
