from __future__ import annotations

import csv
from datetime import datetime
import logging
from pathlib import Path
import threading
from typing import Any

logger = logging.getLogger(__name__)


class EvidenceSaver:
    """Luu tru hinh anh bang chung va ghi nhat ky log.csv khi xay ra bao dong."""

    def __init__(self, evidence_dir: Path, cooldown: float = 30.0) -> None:
        self.evidence_dir = Path(evidence_dir).resolve()
        self.cooldown = cooldown
        self.last_saved_at: float = float("-inf")
        self._lock = threading.Lock()

    def should_save(self, now: float) -> bool:
        """Kiem tra xem da qua thoi gian cooldown de luu bang chung chua."""
        return (now - self.last_saved_at) >= self.cooldown

    def save_async(
        self,
        frame: Any,
        alarm_type: str,
        confidence: float,
        now: float,
    ) -> bool:
        """Khoi chay luong ngam luu anh va ghi log neu du dieu kien."""
        if not self.should_save(now):
            return False

        self.last_saved_at = now
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        image_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_type = alarm_type.lower().replace(" ", "_")
        image_name = f"{image_timestamp}_{safe_type}_{confidence:.2f}.jpg"

        frame_copy = frame.copy()
        threading.Thread(
            target=self._write_files,
            args=(frame_copy, image_name, timestamp, alarm_type, confidence),
            name="alert-evidence-writer",
            daemon=True,
        ).start()
        return True

    def _write_files(
        self,
        frame: Any,
        image_name: str,
        timestamp: str,
        alarm_type: str,
        confidence: float,
    ) -> None:
        """Thuc thi ghi file anh va csv tren luong chay ngam."""
        import cv2

        try:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.error("Khong the tao thu muc evidence: %s", error)
            return

        image_path = self.evidence_dir / image_name
        saved_name = ""
        try:
            if cv2.imwrite(str(image_path), frame):
                saved_name = image_name
                logger.info("[DA LUU BANG CHUNG] %s", image_path)
            else:
                logger.error("Khong the luu file anh: %s", image_path)
        except Exception as error:
            logger.error("Loi khi luu anh evidence: %s", error)

        log_path = self.evidence_dir / "log.csv"
        try:
            with self._lock:
                needs_header = not log_path.exists() or log_path.stat().st_size == 0
                with log_path.open("a", newline="", encoding="utf-8-sig") as log_file:
                    writer = csv.writer(log_file)
                    if needs_header:
                        writer.writerow(
                            ["thoi_gian", "loai_phat_hien", "do_tin_cay_cao_nhat", "ten_file_anh"]
                        )
                    writer.writerow(
                        [timestamp, alarm_type, f"{confidence:.4f}", saved_name]
                    )
        except OSError as error:
            logger.error("Khong the ghi log CSV: %s", error)
