from __future__ import annotations

import logging
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageTk

from src.services.video_service import VideoPlaybackState
from src.utils.video_utils import (
    cv2_bgr_to_pil,
    draw_detections,
    format_timestamp,
    resize_frame_to_fit,
)

if TYPE_CHECKING:
    from src.config import AppConfig
    from src.services.detection_service import DetectionResult, DetectionService
    from src.services.video_service import ProcessedFrameData, VideoService

logger = logging.getLogger(__name__)


class MainWindow:
    """Cửa sổ giao diện chính của ứng dụng demo phát hiện cháy và khói."""

    def __init__(
        self,
        root: tk.Tk,
        config: AppConfig,
        detection_service: DetectionService,
        video_service: VideoService,
    ) -> None:
        self.root = root
        self.config = config
        self.detection_service = detection_service
        self.video_service = video_service

        # Thiết lập thuộc tính cửa sổ chính
        self.root.title("Tecnix AI - Demo Nhận Diện Cháy & Khói Từ Video Camera")
        self.root.geometry("1100x780")
        self.root.minsize(960, 680)
        self.root.configure(bg="#0f172a")

        # Quản lý hình ảnh hiện thời trên Canvas
        self._current_photo: ImageTk.PhotoImage | None = None
        self._last_result: DetectionResult | None = None

        # Bộ đếm khung hình duy trì cảnh báo (chống nhấp nháy khi detection bị đứt quãng)
        self._alert_hold_counter = 0
        self._last_alert_kind: str = "none"
        self._last_alert_details: dict[str, Any] = {}

        # Xây dựng giao diện
        self._setup_styles()
        self._create_widgets()

        # Bắt sự kiện đóng cửa sổ
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Khởi động vòng lặp kiểm tra hàng đợi video
        self.root.after(20, self._poll_video_queue)

    def _setup_styles(self) -> None:
        """Cấu hình style ttk giao diện."""
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            "TButton",
            font=("Segoe UI", 9, "bold"),
            padding=6,
            background="#334155",
            foreground="#f8fafc",
            borderwidth=0,
        )
        style.map(
            "TButton",
            background=[("active", "#475569"), ("disabled", "#1e293b")],
            foreground=[("disabled", "#64748b")],
        )

        style.configure(
            "Action.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=8,
            background="#2563eb",
            foreground="#ffffff",
        )
        style.map(
            "Action.TButton",
            background=[("active", "#1d4ed8"), ("disabled", "#1e293b")],
        )

        style.configure(
            "Stop.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=8,
            background="#dc2626",
            foreground="#ffffff",
        )
        style.map(
            "Stop.TButton",
            background=[("active", "#b91c1c"), ("disabled", "#1e293b")],
        )

        style.configure(
            "Horizontal.TScale",
            background="#1e293b",
            troughcolor="#0f172a",
        )

    def _create_widgets(self) -> None:
        """Tạo các thành phần giao diện."""
        # ========================================================
        # 1. TOP BAR: Tiêu đề và nút chọn file
        # ========================================================
        top_frame = tk.Frame(self.root, bg="#1e293b", padx=15, pady=10)
        top_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            top_frame,
            text="HỆ THỐNG AI PHÁT HIỆN CHÁY & KHÓI",
            font=("Segoe UI", 13, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
        )
        title_label.pack(side=tk.LEFT)

        model_info_text = f"Model: {self.config.model_path.name}"
        self.model_badge = tk.Label(
            top_frame,
            text=model_info_text,
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#0f172a",
            padx=8,
            pady=3,
        )
        self.model_badge.pack(side=tk.LEFT, padx=15)

        self.btn_select = tk.Button(
            top_frame,
            text="📁 Chọn video...",
            command=self._on_select_video,
            font=("Segoe UI", 9, "bold"),
            bg="#3b82f6",
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
        )
        self.btn_select.pack(side=tk.RIGHT)

        self.video_name_label = tk.Label(
            top_frame,
            text="Chưa chọn video nào",
            font=("Segoe UI", 9, "italic"),
            fg="#cbd5e1",
            bg="#1e293b",
        )
        self.video_name_label.pack(side=tk.RIGHT, padx=12)

        # ========================================================
        # 2. KHU VỰC CẢNH BÁO NỔI BẬT (ALERT PANEL)
        # ========================================================
        self.alert_frame = tk.Frame(self.root, bg="#1e293b", padx=15, pady=12, relief=tk.SOLID, bd=1)
        self.alert_frame.pack(fill=tk.X, padx=15, pady=(12, 6))

        self.alert_title = tk.Label(
            self.alert_frame,
            text="CHƯA PHÁT HIỆN LỬA HOẶC KHÓI",
            font=("Segoe UI", 14, "bold"),
            fg="#22c55e",
            bg="#1e293b",
        )
        self.alert_title.pack(anchor="w")

        self.alert_details = tk.Label(
            self.alert_frame,
            text="Trạng thái bình thường | Hệ thống đang sẵn sàng giám sát",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#1e293b",
        )
        self.alert_details.pack(anchor="w", pady=(3, 0))

        # ========================================================
        # 3. KHU VỰC TRUNG TÂM: MÀN HÌNH VIDEO & THÔNG TIN
        # ========================================================
        center_frame = tk.Frame(self.root, bg="#0f172a", padx=15, pady=6)
        center_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas hiển thị video
        self.video_canvas = tk.Canvas(
            center_frame,
            bg="#020617",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.video_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Panel thông số bên phải video
        stats_panel = tk.Frame(center_frame, bg="#1e293b", width=260, padx=14, pady=12)
        stats_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        stats_panel.pack_propagate(False)

        stats_header = tk.Label(
            stats_panel,
            text="THÔNG SỐ PHÁT",
            font=("Segoe UI", 10, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
        )
        stats_header.pack(anchor="w", pady=(0, 10))

        self.time_label = tk.Label(
            stats_panel,
            text="Thời gian: 00:00.0 / 00:00.0",
            font=("Consolas", 10),
            fg="#e2e8f0",
            bg="#1e293b",
        )
        self.time_label.pack(anchor="w", pady=3)

        self.frame_label = tk.Label(
            stats_panel,
            text="Khung hình: 0 / 0",
            font=("Consolas", 10),
            fg="#e2e8f0",
            bg="#1e293b",
        )
        self.frame_label.pack(anchor="w", pady=3)

        self.fps_label = tk.Label(
            stats_panel,
            text="Tốc độ: 0.0 FPS",
            font=("Consolas", 10),
            fg="#e2e8f0",
            bg="#1e293b",
        )
        self.fps_label.pack(anchor="w", pady=3)

        self.det_count_label = tk.Label(
            stats_panel,
            text="Phát hiện: 0 đối tượng",
            font=("Consolas", 10),
            fg="#e2e8f0",
            bg="#1e293b",
        )
        self.det_count_label.pack(anchor="w", pady=3)

        # Phân cách
        sep = tk.Frame(stats_panel, height=1, bg="#334155")
        sep.pack(fill=tk.X, pady=12)

        # Điều khiển độ nhạy Confidence
        conf_header = tk.Label(
            stats_panel,
            text="ĐỘ NHẠY (CONFIDENCE)",
            font=("Segoe UI", 10, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
        )
        conf_header.pack(anchor="w")

        self.conf_val_label = tk.Label(
            stats_panel,
            text=f"{int(self.config.conf_threshold * 100)}%",
            font=("Segoe UI", 11, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
        )
        self.conf_val_label.pack(anchor="w", pady=(2, 4))

        self.conf_slider = ttk.Scale(
            stats_panel,
            from_=5,
            to=95,
            value=self.config.conf_threshold * 100,
            command=self._on_conf_slider_change,
            orient=tk.HORIZONTAL,
            style="Horizontal.TScale",
        )
        self.conf_slider.pack(fill=tk.X, pady=(0, 10))

        # Ghi chú hướng dẫn
        note_box = tk.Label(
            stats_panel,
            text=(
                "💡 Mẹo:\n"
                "• Hạ confidence để tăng độ nhạy phát hiện khói mờ.\n"
                "• Tăng confidence nếu có cảnh báo giả do ánh đèn vàng.\n"
                "• Bounding box Lửa màu đỏ, Khói màu xám sáng."
            ),
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#0f172a",
            justify=tk.LEFT,
            padx=8,
            pady=8,
            wraplength=220,
        )
        note_box.pack(fill=tk.X, side=tk.BOTTOM, pady=6)

        # ========================================================
        # 4. THANH ĐIỀU KHIỂN PHÁT (PLAYBACK CONTROLS)
        # ========================================================
        ctrl_frame = tk.Frame(self.root, bg="#1e293b", padx=15, pady=10)
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_play = tk.Button(
            ctrl_frame,
            text="▶ Phát",
            command=self._on_play,
            font=("Segoe UI", 10, "bold"),
            bg="#16a34a",
            fg="#ffffff",
            activebackground="#15803d",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
        )
        self.btn_play.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_pause = tk.Button(
            ctrl_frame,
            text="⏸ Tạm dừng",
            command=self._on_pause,
            font=("Segoe UI", 10, "bold"),
            bg="#d97706",
            fg="#ffffff",
            activebackground="#b45309",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
        )
        self.btn_pause.pack(side=tk.LEFT, padx=4)

        self.btn_resume = tk.Button(
            ctrl_frame,
            text="⏯ Tiếp tục",
            command=self._on_resume,
            font=("Segoe UI", 10, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
        )
        self.btn_resume.pack(side=tk.LEFT, padx=4)

        self.btn_stop = tk.Button(
            ctrl_frame,
            text="⏹ Dừng",
            command=self._on_stop,
            font=("Segoe UI", 10, "bold"),
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=(4, 15))

        self.status_label = tk.Label(
            ctrl_frame,
            text="Sẵn sàng. Hãy chọn video để bắt đầu.",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b",
        )
        self.status_label.pack(side=tk.RIGHT)

    # ========================================================
    # XỬ LÝ SỰ KIỆN GIAO DIỆN
    # ========================================================
    def _on_select_video(self) -> None:
        """Mở hộp thoại chọn file video."""
        filetypes = [
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("MP4 Video", "*.mp4"),
            ("AVI Video", "*.avi"),
            ("QuickTime MOV", "*.mov"),
            ("Matroska MKV", "*.mkv"),
            ("All Files", "*.*"),
        ]
        chosen = filedialog.askopenfilename(
            title="Chọn file video camera",
            filetypes=filetypes,
        )
        if not chosen:
            return

        ok, msg = self.video_service.open_video(chosen)
        if not ok:
            messagebox.showerror("Lỗi mở video", msg)
            self.status_label.config(text=f"Lỗi: {msg}", fg="#ef4444")
            return

        meta = self.video_service.get_metadata()
        if meta:
            self.video_name_label.config(text=f"{meta.filename} ({meta.duration_sec:.1f}s)")
            self.status_label.config(text=msg, fg="#22c55e")
            self._update_stats_display(0, meta.total_frames, 0.0, meta.duration_sec, meta.fps, 0)

        # Cập nhật trạng thái các nút
        self.btn_play.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)

        # Đặt lại cảnh báo về trạng thái bình thường
        self._reset_alert_panel()

    def _on_play(self) -> None:
        """Bắt đầu phát video."""
        success = self.video_service.play()
        if success:
            self.btn_play.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL)
            self.btn_resume.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.status_label.config(text="Đang phát và phân tích video...", fg="#38bdf8")

    def _on_pause(self) -> None:
        """Tạm dừng video."""
        self.video_service.pause()
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.NORMAL)
        self.status_label.config(text="Đã tạm dừng video.", fg="#facc15")

    def _on_resume(self) -> None:
        """Tiếp tục phát video."""
        self.video_service.resume()
        self.btn_resume.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.status_label.config(text="Đang phát và phân tích video...", fg="#38bdf8")

    def _on_stop(self) -> None:
        """Dừng video."""
        self.video_service.stop()
        self.btn_play.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.DISABLED)
        self.status_label.config(text="Đã dừng video.", fg="#94a3b8")
        self._reset_alert_panel()

    def _on_conf_slider_change(self, val: str) -> None:
        """Khi người dùng kéo thanh điều chỉnh Confidence."""
        conf_percent = int(float(val))
        self.conf_val_label.config(text=f"{conf_percent}%")
        conf_float = conf_percent / 100.0
        self.config.conf_threshold = conf_float
        self.detection_service.set_conf_threshold(conf_float)

    def _on_close(self) -> None:
        """Xử lý đóng cửa sổ an toàn, giải phóng tài nguyên."""
        logger.info("Đang đóng ứng dụng...")
        self.video_service.release()
        self.root.destroy()

    # ========================================================
    # XỬ LÝ KHUNG HÌNH VÀ CẬP NHẬT CẢNH BÁO
    # ========================================================
    def _poll_video_queue(self) -> None:
        """Kiểm tra hàng đợi frame định kỳ trên main thread của Tkinter."""
        try:
            while not self.video_service.frame_queue.empty():
                data: ProcessedFrameData = self.video_service.frame_queue.get_nowait()
                self._render_frame(data)
        except queue.Empty:
            pass

        # Kiểm tra nếu video vừa kết thúc
        if self.video_service.state == VideoPlaybackState.COMPLETED:
            if self.btn_play["state"] == tk.DISABLED:
                self.btn_play.config(state=tk.NORMAL)
                self.btn_pause.config(state=tk.DISABLED)
                self.btn_resume.config(state=tk.DISABLED)
                self.status_label.config(text="Video đã kết thúc.", fg="#a855f7")
                self._reset_alert_panel()
                messagebox.showinfo("Thông báo", "Đã phát hết file video.")

        # Tiếp tục đặt lịch kiểm tra sau 20ms
        self.root.after(20, self._poll_video_queue)

    def _render_frame(self, data: ProcessedFrameData) -> None:
        """Vẽ bounding box lên frame và hiển thị lên Canvas Tkinter."""
        annotated = draw_detections(
            frame=data.frame,
            detections=data.result.detections,
            class_colors_bgr=self.config.class_colors_bgr,
            display_names=self.config.class_display_names,
        )

        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            canvas_w, canvas_h = 720, 480

        resized, _ = resize_frame_to_fit(annotated, canvas_w, canvas_h)
        pil_img = cv2_bgr_to_pil(resized)
        self._current_photo = ImageTk.PhotoImage(image=pil_img)

        # Căn giữa ảnh trên Canvas
        self.video_canvas.delete("all")
        x = canvas_w // 2
        y = canvas_h // 2
        self.video_canvas.create_image(x, y, image=self._current_photo, anchor=tk.CENTER)

        # Cập nhật thông số thống kê
        meta = self.video_service.get_metadata()
        total_duration = meta.duration_sec if meta else 0.0
        self._update_stats_display(
            current_frame=data.current_frame,
            total_frames=data.total_frames,
            current_time=data.timestamp_sec,
            total_duration=total_duration,
            fps=data.fps,
            detection_count=len(data.result.detections),
        )

        # Cập nhật khu vực cảnh báo
        self._update_alert_status(data.result)

    def _update_stats_display(
        self,
        current_frame: int,
        total_frames: int,
        current_time: float,
        total_duration: float,
        fps: float,
        detection_count: int,
    ) -> None:
        """Cập nhật các số liệu thống kê thời gian thực."""
        cur_t = format_timestamp(current_time, show_millis=True)
        tot_t = format_timestamp(total_duration, show_millis=False)
        self.time_label.config(text=f"Thời gian: {cur_t} / {tot_t}")
        self.frame_label.config(text=f"Khung hình: {current_frame} / {total_frames}")
        self.fps_label.config(text=f"Tốc độ: {fps:.1f} FPS")
        self.det_count_label.config(text=f"Phát hiện: {detection_count} đối tượng")

    def _update_alert_status(self, result: DetectionResult) -> None:
        """Cập nhật khu vực cảnh báo theo 4 trạng thái yêu cầu của nghiệp vụ."""
        has_fire = result.fire_detected
        has_smoke = result.smoke_detected
        time_str = format_timestamp(result.timestamp_sec, show_millis=True)

        # Xác định trạng thái tức thời của frame này
        if has_fire and has_smoke:
            current_kind = "fire_and_smoke"
        elif has_fire:
            current_kind = "fire"
        elif has_smoke:
            current_kind = "smoke"
        else:
            current_kind = "none"

        # Logic chống nhấp nháy (Debounce / Hold window):
        # Nếu có phát hiện mới -> làm mới bộ đếm hold
        if current_kind != "none":
            self._last_alert_kind = current_kind
            self._alert_hold_counter = self.config.alert_hold_frames
            self._last_alert_details = {
                "kind": current_kind,
                "fire_conf": result.max_fire_conf,
                "smoke_conf": result.max_smoke_conf,
                "timestamp_str": time_str,
                "frame_idx": result.frame_idx,
            }
        else:
            # Không có detection trong frame này
            if self._alert_hold_counter > 0:
                self._alert_hold_counter -= 1
            else:
                self._last_alert_kind = "none"

        # Hiển thị trạng thái tương ứng
        if self._last_alert_kind == "fire_and_smoke":
            # TRẠNG THÁI 4: Phát hiện cả lửa và khói
            fc = self._last_alert_details.get("fire_conf", 0.0) * 100
            sc = self._last_alert_details.get("smoke_conf", 0.0) * 100
            ts = self._last_alert_details.get("timestamp_str", time_str)
            self.alert_frame.config(bg="#7f1d1d", highlightbackground="#ef4444")
            self.alert_title.config(
                text="🔥 CẢNH BÁO: PHÁT HIỆN LỬA VÀ KHÓI",
                fg="#ffffff",
                bg="#7f1d1d",
            )
            self.alert_details.config(
                text=f"Lửa: {fc:.1f}% | Khói: {sc:.1f}% | Thời điểm: {ts}",
                fg="#fecaca",
                bg="#7f1d1d",
            )
        elif self._last_alert_kind == "fire":
            # TRẠNG THÁI 2: Phát hiện lửa
            fc = self._last_alert_details.get("fire_conf", 0.0) * 100
            ts = self._last_alert_details.get("timestamp_str", time_str)
            self.alert_frame.config(bg="#991b1b", highlightbackground="#dc2626")
            self.alert_title.config(
                text="🔥 CẢNH BÁO: PHÁT HIỆN LỬA",
                fg="#ffffff",
                bg="#991b1b",
            )
            self.alert_details.config(
                text=f"Độ tin cậy cao nhất: {fc:.1f}% | Thời điểm: {ts}",
                fg="#fecaca",
                bg="#991b1b",
            )
        elif self._last_alert_kind == "smoke":
            # TRẠNG THÁI 3: Phát hiện khói
            sc = self._last_alert_details.get("smoke_conf", 0.0) * 100
            ts = self._last_alert_details.get("timestamp_str", time_str)
            self.alert_frame.config(bg="#78350f", highlightbackground="#d97706")
            self.alert_title.config(
                text="💨 CẢNH BÁO: PHÁT HIỆN KHÓI",
                fg="#ffffff",
                bg="#78350f",
            )
            self.alert_details.config(
                text=f"Độ tin cậy cao nhất: {sc:.1f}% | Thời điểm: {ts}",
                fg="#fde68a",
                bg="#78350f",
            )
        else:
            # TRẠNG THÁI 1: Chưa phát hiện
            self._reset_alert_panel()

    def _reset_alert_panel(self) -> None:
        """Đưa khu vực cảnh báo về trạng thái bình thường (Trạng thái 1)."""
        self._last_alert_kind = "none"
        self._alert_hold_counter = 0
        self.alert_frame.config(bg="#1e293b", highlightbackground="#334155")
        self.alert_title.config(
            text="CHƯA PHÁT HIỆN LỬA HOẶC KHÓI",
            fg="#22c55e",
            bg="#1e293b",
        )
        self.alert_details.config(
            text="Trạng thái bình thường | Chưa ghi nhận dấu hiệu cháy hoặc khói",
            fg="#94a3b8",
            bg="#1e293b",
        )
