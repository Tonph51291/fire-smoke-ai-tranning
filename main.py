from __future__ import annotations

import argparse
from collections import deque
import logging
from pathlib import Path
import sys
import time

from fire_alert import (
    AlertManager,
    AlertState,
    Camera,
    Config,
    Detector,
    Display,
    EvidenceSaver,
    MotionFilter,
    TemporalFilter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fire_alert")


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Nhan dien fire/smoke voi camera va coi canh bao.")
    parser.add_argument("--source", default=cfg.camera_index, help="Chi so webcam hoac URL RTSP.")
    parser.add_argument("--weights", type=Path, default=cfg.model_path, help="Duong dan toi best.pt.")
    parser.add_argument("--conf", type=float, default=cfg.conf_threshold, help="Nguong tin cay (0..1).")
    parser.add_argument("--device", default=cfg.device, help="Vi du: cpu, 0.")
    parser.add_argument("--reconnect-attempts", type=int, default=cfg.reconnect_attempts)
    parser.add_argument("--reconnect-delay", type=float, default=cfg.reconnect_delay)
    args = parser.parse_args()

    cfg.camera_index, cfg.model_path = args.source, args.weights
    cfg.conf_threshold, cfg.device = args.conf, args.device
    cfg.reconnect_attempts, cfg.reconnect_delay = args.reconnect_attempts, args.reconnect_delay
    return cfg


def main() -> int:
    cfg = parse_args()
    camera = Camera(cfg.camera_index, cfg.reconnect_attempts, cfg.reconnect_delay)
    if not camera.open():
        return 1

    try:
        detector = Detector(cfg.model_path, cfg.conf_threshold, cfg.device, cfg.imgsz)
    except FileNotFoundError as err:
        logger.error(str(err))
        camera.release()
        return 1

    temporal_filter = TemporalFilter(cfg.window_size, cfg.min_hits)
    motion_filter = MotionFilter(cfg.enable_motion_filter, cfg.flicker_min)
    alert_mgr = AlertManager(cfg.enable_sound, cfg.enable_voice, cfg.beep_pause, cfg.voice_every)
    evidence_saver = EvidenceSaver(cfg.evidence_dir, cfg.evidence_cooldown)
    display = Display(cfg.window_name, cfg.state_colors, cfg.window_size)

    alert_mgr.start()
    frame_times: deque[float] = deque(maxlen=30)
    cached_dets, cached_raw, frame_idx = [], None, 0
    prev_state = AlertState.NORMAL

    logger.info("Camera san sang. Nhan [Q] de thoat, [M] de bat/tat am thanh.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if cached_raw is None or (frame_idx - 1) % cfg.inference_every == 0:
                raw_dets, cached_raw = detector.predict(frame)
                cached_dets = motion_filter.filter(frame, raw_dets)

            state, class_hits, confirmed = temporal_filter.update(cached_dets)
            now = time.monotonic()
            frame_times.append(now)
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0

            present = [cfg.class_names[c] for c in ("fire", "smoke") if class_hits.get(c, 0) > 0]
            det_text = " + ".join(present) if present else "KHONG PHAT HIEN"

            if state == AlertState.ALARM:
                alarm_type = "LUA VA KHOI" if len(confirmed) == 2 else cfg.class_names[next(iter(confirmed))]
                kind = "lua_va_khoi" if len(confirmed) == 2 else ("khoi" if "smoke" in confirmed else "lua")
                alert_mgr.trigger(kind)
                if prev_state != AlertState.ALARM:
                    conf = temporal_filter.get_max_confirmed_confidence(cached_dets, confirmed)
                    evidence_saver.save_async(frame, alarm_type, conf, now)
            else:
                alert_mgr.clear()

            prev_state = state
            out = display.render(frame, cached_raw, state, det_text, confirmed, class_hits, fps, alert_mgr.is_sound_enabled())
            display.show(out)

            key = display.wait_key(1)
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                alert_mgr.toggle_sound()
    finally:
        alert_mgr.stop()
        camera.release()
        display.close()
        logger.info("Da dong camera va giai phong tai nguyen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
