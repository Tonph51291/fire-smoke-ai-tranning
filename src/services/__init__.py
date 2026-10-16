"""Service layer for AI detection and video stream processing.
"""

from src.services.detection_service import Detection, DetectionResult, DetectionService
from src.services.video_service import VideoMetadata, VideoPlaybackState, VideoService

__all__ = [
    "Detection",
    "DetectionResult",
    "DetectionService",
    "VideoMetadata",
    "VideoPlaybackState",
    "VideoService",
]
