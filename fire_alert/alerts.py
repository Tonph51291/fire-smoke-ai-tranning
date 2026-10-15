from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time

logger = logging.getLogger(__name__)


def speak_text(text: str) -> None:
    """Doc to canh bao bang PowerShell System.Speech, khong can thu vien ngoai."""
    try:
        command = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.Speak('{text}')"
        )
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            creationflags=flags,
            timeout=8,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as error:
        logger.error("Khong the phat giong doc canh bao: %s", error)


class AlertManager:
    """Quan ly coi hu va giong doc canh bao tren luong rieng biet."""

    def __init__(
        self,
        enable_sound: bool = True,
        enable_voice: bool = True,
        beep_pause: float = 1.0,
        voice_every: int = 3,
    ) -> None:
        self.enable_voice = enable_voice
        self.beep_pause = beep_pause
        self.voice_every = voice_every

        self._alert_active = threading.Event()
        self._sound_enabled = threading.Event()
        self._stop_event = threading.Event()

        if enable_sound:
            self._sound_enabled.set()

        self._alarm_kind = "lua"
        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._voice_thread: threading.Thread | None = None

    def start(self) -> None:
        """Khoi dong luong am thanh chay ngam."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name="alert-sound-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def _run_loop(self) -> None:
        """Vong lap chay ngam phat coi va dieu phoi giong doc."""
        try:
            import winsound
        except ImportError:
            logger.warning("Khong ho tro winsound tren he thong; am thanh bi vo hieu hoa.")
            return

        beep_count = 0
        while not self._stop_event.is_set():
            if self._alert_active.is_set() and self._sound_enabled.is_set():
                # 1. Phat coi hu 2 tong (1000Hz -> 1500Hz, moi tong 300ms)
                try:
                    winsound.Beep(1000, 300)
                    if (
                        self._stop_event.is_set()
                        or not self._alert_active.is_set()
                        or not self._sound_enabled.is_set()
                    ):
                        continue
                    winsound.Beep(1500, 300)
                except RuntimeError as error:
                    logger.error("Loi khi phat tieng coi: %s", error)
                    self._sound_enabled.clear()
                    self._stop_event.wait(0.5)
                    continue

                beep_count += 1

                # 2. Phat giong doc canh bao sau moi voice_every luot coi
                if self.enable_voice and (beep_count % self.voice_every == 0):
                    with self._lock:
                        kind = self._alarm_kind

                    if kind == "lua_va_khoi":
                        message = "Canh bao! Phat hien lua va khoi"
                    elif kind == "khoi":
                        message = "Canh bao! Phat hien khoi"
                    else:
                        message = "Canh bao! Phat hien lua"

                    if self._voice_thread is not None and self._voice_thread.is_alive():
                        self._voice_thread.join(timeout=3.0)

                    if (
                        self._sound_enabled.is_set()
                        and not self._stop_event.is_set()
                        and self._alert_active.is_set()
                    ):
                        self._voice_thread = threading.Thread(
                            target=speak_text,
                            args=(message,),
                            name="alert-voice",
                            daemon=True,
                        )
                        self._voice_thread.start()

                # Nghi giua cac luot coi (dung ngay neu het bao dong hoac an Q/M)
                start_pause = time.monotonic()
                while (
                    time.monotonic() - start_pause < self.beep_pause
                    and not self._stop_event.is_set()
                    and self._alert_active.is_set()
                    and self._sound_enabled.is_set()
                ):
                    self._stop_event.wait(0.05)
            else:
                beep_count = 0
                self._stop_event.wait(0.05)

    def trigger(self, kind: str) -> None:
        """Kich hoat trang thai canh bao am thanh."""
        with self._lock:
            self._alarm_kind = kind
        self._alert_active.set()

    def clear(self) -> None:
        """Tat trang thai canh bao am thanh."""
        self._alert_active.clear()

    def toggle_sound(self) -> bool:
        """Bat/tat tieng (phim M). Tra ve trang thai am thanh sau khi doi."""
        if self._sound_enabled.is_set():
            self._sound_enabled.clear()
            logger.info("[AM THANH] Da TAT coi hu va giong doc.")
            return False
        else:
            self._sound_enabled.set()
            logger.info("[AM THANH] Da BAT coi hu va giong doc.")
            return True

    def is_sound_enabled(self) -> bool:
        return self._sound_enabled.is_set()

    def is_alert_active(self) -> bool:
        return self._alert_active.is_set()

    def stop(self) -> None:
        """Dung toan bo am thanh va luong chay ngam khi thoat."""
        self._alert_active.clear()
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
        if self._voice_thread is not None and self._voice_thread.is_alive():
            self._voice_thread.join(timeout=1.0)
            self._voice_thread = None
