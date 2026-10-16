from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.services.detection_service import DetectionResult, DetectionService

logger = logging.getLogger(__name__)


class VideoPlaybackState(str, Enum):
    """Trạng thái phát của video."""
    IDLE = "IDLE"           # Chưa chọn video
    READY = "READY"         # Đã tải video, sẵn sàng phát
    PLAYING = "PLAYING"     # Đang phát
    PAUSED = "PAUSED"       # Đang tạm dừng
    STOPPED = "STOPPED"     # Đã dừng
    COMPLETED = "COMPLETED" # Đã phát hết video


@dataclass
class VideoMetadata:
    """Thông tin kỹ thuật của file video."""
    path: Path
    filename: str
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int


@dataclass
class ProcessedFrameData:
    """Gói dữ liệu một khung hình đã qua xử lý gửi lên giao diện."""
    frame: np.ndarray
    result: DetectionResult
    current_frame: int
    total_frames: int
    timestamp_sec: float
    fps: float


class VideoService:
    """Dịch vụ mở video, đọc từng frame, thực hiện suy luận và điều khiển phát."""

    def __init__(
        self,
        detection_service: DetectionService,
        config: AppConfig,
    ) -> None:
        self.detection_service = detection_service
        self.config = config

        self._state = VideoPlaybackState.IDLE
        self._state_lock = threading.Lock()

        self._metadata: VideoMetadata | None = None
        self._cap: cv2.VideoCapture | None = None

        # Queue chứa frame đã qua nhận diện gửi sang UI (giới hạn maxsize=2 để không ngốn RAM)
        self.frame_queue: queue.Queue[ProcessedFrameData] = queue.Queue(maxsize=2)

        # Các biến điều khiển luồng ngầm
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        # Quản lý frame và timestamp
        self._current_frame_idx = 0
        self._current_timestamp = 0.0

    @property
    def state(self) -> VideoPlaybackState:
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: VideoPlaybackState) -> None:
        with self._state_lock:
            self._state = new_state
        logger.info("Trạng thái VideoService chuyển sang: %s", new_state.value)

    def get_metadata(self) -> VideoMetadata | None:
        return self._metadata

    def open_video(self, video_path: Path | str) -> tuple[bool, str]:
        """Mở file video và đọc metadata."""
        self.stop()

        path = Path(video_path).resolve()
        if not path.is_file():
            return False, f"Không tìm thấy file video: {path}"

        suffix = path.suffix.lower()
        if suffix not in self.config.supported_formats:
            return (
                False,
                f"Định dạng '{suffix}' không được hỗ trợ. "
                f"Vui lòng chọn: {', '.join(self.config.supported_formats)}",
            )

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return False, f"Không thể mở video bằng OpenCV: {path.name}"

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # Giá trị mặc định an toàn

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        duration_sec = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._metadata = VideoMetadata(
            path=path,
            filename=path.name,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            width=width,
            height=height,
        )
        self._cap = cap
        self._current_frame_idx = 0
        self._current_timestamp = 0.0

        # Đọc thử 1 frame đầu tiên để hiển thị sẵn sàng
        ok, first_frame = self._cap.read()
        if ok and first_frame is not None:
            # Reset vị trí về đầu video
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Đưa frame đầu tiên vào queue để hiển thị lên màn hình
            from src.services.detection_service import DetectionResult
            dummy_result = DetectionResult(
                detections=[],
                frame_idx=0,
                timestamp_sec=0.0,
                fire_detected=False,
                smoke_detected=False,
                max_fire_conf=0.0,
                max_smoke_conf=0.0,
            )
            self._put_frame(first_frame, dummy_result, 0, 0.0)

        self._set_state(VideoPlaybackState.READY)
        return True, f"Đã mở video: {path.name} ({width}x{height}, {fps:.1f} FPS)"

    def _put_frame(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        frame_idx: int,
        timestamp_sec: float,
    ) -> None:
        """Đưa frame vào queue UI, loại bỏ frame cũ nếu queue đầy để tránh lag."""
        total = self._metadata.total_frames if self._metadata else 0
        fps = self._metadata.fps if self._metadata else 25.0
        data = ProcessedFrameData(
            frame=frame,
            result=result,
            current_frame=frame_idx,
            total_frames=total,
            timestamp_sec=timestamp_sec,
            fps=fps,
        )
        try:
            self.frame_queue.put_nowait(data)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(data)

    def play(self) -> bool:
        """Bắt đầu phát video trên luồng nền riêng."""
        if self._cap is None or not self._cap.isOpened():
            return False

        if self.state == VideoPlaybackState.PLAYING:
            return True

        if self.state == VideoPlaybackState.PAUSED:
            self.resume()
            return True

        if self.state == VideoPlaybackState.COMPLETED or self.state == VideoPlaybackState.STOPPED:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._current_frame_idx = 0
            self._current_timestamp = 0.0

        self._stop_event.clear()
        self._pause_event.clear()
        self._set_state(VideoPlaybackState.PLAYING)

        self._worker_thread = threading.Thread(
            target=self._playback_loop,
            name="video-playback-worker",
            daemon=True,
        )
        self._worker_thread.start()
        return True

    def pause(self) -> None:
        """Tạm dừng phát video."""
        if self.state == VideoPlaybackState.PLAYING:
            self._pause_event.set()
            self._set_state(VideoPlaybackState.PAUSED)

    def resume(self) -> None:
        """Tiếp tục phát video sau khi tạm dừng."""
        if self.state == VideoPlaybackState.PAUSED:
            self._pause_event.clear()
            self._set_state(VideoPlaybackState.PLAYING)

    def stop(self) -> None:
        """Dừng phát video và đưa vị trí về đầu."""
        self._stop_event.set()
        self._pause_event.clear()

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        if self._cap is not None and self._cap.isOpened():
            try:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass

        self._current_frame_idx = 0
        self._current_timestamp = 0.0

        # Làm rỗng queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if self._metadata is not None:
            self._set_state(VideoPlaybackState.STOPPED)
        else:
            self._set_state(VideoPlaybackState.IDLE)

    def release(self) -> None:
        """Giải phóng hoàn toàn tài nguyên video."""
        self.stop()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                logger.warning("Lỗi khi giải phóng VideoCapture: %s", exc)
            finally:
                self._cap = None
        self._metadata = None
        self._set_state(VideoPlaybackState.IDLE)

    def _playback_loop(self) -> None:
        """Vòng lặp đọc frame và suy luận chạy ngầm."""
        if self._cap is None or self._metadata is None:
            return

        fps = self._metadata.fps if self._metadata.fps > 0 else 25.0
        frame_interval = 1.0 / fps

        interval_counter = 0
        last_result = None

        logger.info("Bắt đầu vòng lặp phát video tại frame %d", self._current_frame_idx)

        while not self._stop_event.is_set():
            # Xử lý khi tạm dừng
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            loop_start = time.perf_counter()

            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Video đã kết thúc
                logger.info("Đã phát đến cuối video.")
                self._set_state(VideoPlaybackState.COMPLETED)
                break

            self._current_frame_idx += 1
            # Tính timestamp chính xác từ frame index và fps
            self._current_timestamp = (self._current_frame_idx - 1) / fps

            interval_counter += 1
            should_infer = (interval_counter % self.config.inference_interval == 0)

            if should_infer or last_result is None:
                last_result = self.detection_service.predict(
                    frame=frame,
                    frame_idx=self._current_frame_idx,
                    timestamp_sec=self._current_timestamp,
                )
            else:
                # Tái sử dụng kết quả detection trước đó cho frame xen kẽ
                last_result.frame_idx = self._current_frame_idx
                last_result.timestamp_sec = self._current_timestamp

            # Đưa frame kèm kết quả nhận diện sang UI
            self._put_frame(frame, last_result, self._current_frame_idx, self._current_timestamp)

            # Tính toán thời gian ngủ để duy trì tốc độ phát khớp với FPS của video
            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0.002:
                time.sleep(sleep_time)

        logger.info("Vòng lặp phát video kết thúc.")
