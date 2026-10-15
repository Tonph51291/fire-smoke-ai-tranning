from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve_default_weights() -> Path:
    """Tu dong tim best.pt trong cac thu muc runs/fire_smoke neu khong chi dinh."""
    candidates = [
        ROOT / "runs" / "fire_smoke" / "fire_v1" / "weights" / "best.pt",
        ROOT / "runs" / "fire_smoke" / "experiment1" / "weights" / "best.pt",
        ROOT / "yolov8n.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    found = list(ROOT.glob("runs/**/best.pt"))
    if found:
        return found[0]
    return candidates[0]


class AlertState(str, Enum):
    """Trang thai canh bao he thong."""
    NORMAL = "BINH THUONG"
    SUSPICIOUS = "NGHI NGO"
    ALARM = "BAO DONG"


@dataclass(slots=True)
class Detection:
    """Thong tin mot doi tuong duoc phat hien."""
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class Config:
    """Chua toan bo thong so cau hinh cho he thong camera nhan dien."""
    # Nguon camera va mo hinh
    model_path: Path = field(default_factory=resolve_default_weights)
    camera_index: str = "0"
    cam_width: int | None = None
    cam_height: int | None = None
    imgsz: int | tuple[int, int] | None = None
    conf_threshold: float = 0.25
    device: str | None = None
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0

    # Bo loc thoi gian va chuyen dong
    window_size: int = 10
    min_hits: int = 5
    flicker_min: float = 0.0
    enable_motion_filter: bool = False
    inference_every: int = 2

    # Bang chung va luu tru
    evidence_cooldown: float = 30.0
    evidence_dir: Path = field(default_factory=lambda: ROOT / "evidence")

    # Am thanh va giong doc
    enable_sound: bool = True
    enable_voice: bool = True
    beep_pause: float = 1.0
    voice_every: int = 3

    # Hien thi giao dien
    window_name: str = "Fire and Smoke Detection - Tecnix AI"
    class_names: dict[str, str] = field(
        default_factory=lambda: {"fire": "LUA", "smoke": "KHOI"}
    )
    state_colors: dict[AlertState, tuple[int, int, int]] = field(
        default_factory=lambda: {
            AlertState.NORMAL: (0, 200, 0),        # Xanh la
            AlertState.SUSPICIOUS: (0, 165, 255),  # Cam
            AlertState.ALARM: (0, 0, 255),         # Do
        }
    )
