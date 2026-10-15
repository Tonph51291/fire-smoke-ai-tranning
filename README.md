# Fire & Smoke AI Detection System (Tecnix AI)

Hệ thống AI nhận diện Lửa và Khói theo thời gian thực từ Webcam hoặc camera IP/RTSP sử dụng mô hình **YOLOv8 (Ultralytics)**, kết hợp bộ lọc thời gian chống báo động giả, hệ thống còi hú 2 tầng âm, giọng đọc tiếng Việt và tự động lưu bằng chứng (ảnh + nhật ký CSV).

---

## 1. Cấu trúc thư mục dự án

```text
fire-smoke-ai-tranning/
├── main.py                     # Điểm khởi chạy camera chính (tối ưu, mạch lạc)
├── fire_alert/                 # Package chứa toàn bộ logic ứng dụng nhận diện
│   ├── __init__.py             # Export các thành phần chính
│   ├── config.py               # Dataclass Config, Detection, Enum AlertState
│   ├── camera.py               # Quản lý VideoCapture, tự động kết nối lại khi mất mạng/rớt frame
│   ├── detector.py             # Nạp mô hình YOLOv8, chạy suy luận, trích xuất class động
│   ├── filters.py              # TemporalFilter (lọc K/N khung) & MotionFilter (lọc chuyển động)
│   ├── alerts.py               # AlertManager quản lý thread còi hú (winsound) và giọng đọc tiếng Việt
│   ├── display.py              # Vẽ HUD thông tin, bounding box, banner đỏ, viền nhấp nháy
│   └── evidence.py             # Lưu ảnh bằng chứng (.jpg) và nhật ký log.csv có cooldown
├── train.py                    # Script huấn luyện mô hình YOLOv8
├── evaluate.py                 # Script đánh giá mô hình trên tập validation
├── check_dataset.py            # Script kiểm tra cấu trúc dữ liệu và nhãn YOLO
├── data.yaml                   # File cấu hình đường dẫn dataset và danh sách class (fire, smoke)
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── evidence/                   # Thư mục lưu ảnh và file log.csv khi có báo động
├── dataset/                    # Dữ liệu hình ảnh và nhãn train/val
└── runs/                       # Thư mục lưu kết quả huấn luyện (best.pt, charts...)
```

---

## 2. Mô tả kiến trúc Package `fire_alert`

Hệ thống được thiết kế theo hướng module hóa (Clean Code), tách biệt hoàn toàn giữa xử lý hình ảnh, logic lọc và cảnh báo ngoại vi:

| Module | Nhiệm vụ chính |
| :--- | :--- |
| **`config.py`** | Quản lý toàn bộ thông số tập trung (`Config`): đường dẫn model, ngưỡng `conf_threshold=0.25`, kích thước cửa sổ lọc `window_size=10`, `min_hits=5`, thời gian cooldown ảnh `30s`, cài đặt âm thanh. Chứa dataclass `Detection` và Enum `AlertState` (`NORMAL`, `SUSPICIOUS`, `ALARM`). |
| **`camera.py`** | Class `Camera` bao bọc `cv2.VideoCapture`. Hỗ trợ cấu hình độ phân giải và cơ chế tự động thử lại (`reconnect`) lên đến 5 lần nếu camera hoặc stream RTSP bị ngắt quãng. |
| **`detector.py`** | Class `Detector` nạp file trọng số (`best.pt`), đọc tên lớp động từ `model.names` (không cố định thứ tự lớp). Trả về danh sách `Detection` chuẩn hóa. |
| **`filters.py`** | `TemporalFilter` duy trì sliding window $K/N$ khung hình gần nhất để phân loại trạng thái (`BINH THUONG` $\rightarrow$ `NGHI NGO` $\rightarrow$ `BAO DONG`), loại bỏ hiện tượng nhấp nháy báo giả. `MotionFilter` lọc chuyển động vi sai giữa các frame. |
| **`alerts.py`** | Class `AlertManager` chạy trên **thread nền riêng**, không gây giật lag luồng video: phát còi hú 2 tầng âm (`1000Hz` và `1500Hz` qua `winsound`) và điều phối giọng đọc tiếng Việt qua PowerShell `System.Speech` mỗi 3 lượt còi. Hỗ trợ dừng sạch ngay lập tức khi nhấn phím Q. |
| **`display.py`** | Class `Display` vẽ bounding box, HUD (FPS, trạng thái, chế độ âm thanh), banner cảnh báo đỉnh màn hình và viền đỏ nhấp nháy chu kỳ 0.5s khi ở trạng thái `BAO DONG`. |
| **`evidence.py`** | Class `EvidenceSaver` chạy thread nền tự động chụp khung hình hiện tại lưu thành file `.jpg` và ghi thông tin sự kiện vào `evidence/log.csv` với cơ chế cooldown 30 giây để tránh tràn ổ đĩa. |

---

## 3. Cài đặt và Môi trường

Dự án yêu cầu hệ điều hành **Windows 10/11**, **Python 3.10** và sử dụng môi trường ảo `venv`:

```powershell
# 1. Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1

# 2. Cài đặt hoặc kiểm tra dependencies
pip install -r requirements.txt
```

---

## 4. Hướng dẫn sử dụng Camera nhận diện

### Chạy cơ bản với Webcam mặc định (Camera 0)
```powershell
python main.py
```

### Chạy với Camera IP / Luồng RTSP
```powershell
python main.py --source "rtsp://username:password@192.168.1.100:554/stream1"
```

### Các tùy chọn tham số dòng lệnh
```powershell
python main.py --weights runs/fire_smoke/fire_v1/weights/best.pt --conf 0.3 --device 0
```
- `--source`: Chỉ số webcam (ví dụ `0`, `1`) hoặc link RTSP.
- `--weights`: Đường dẫn tới file trọng số model đã huấn luyện (`best.pt`).
- `--conf`: Ngưỡng tin cậy confidence từ `0.0` đến `1.0` (mặc định `0.25`).
- `--device`: Thiết bị chạy AI (`0` cho GPU NVIDIA hoặc `cpu`).
- `--reconnect-attempts`: Số lần thử kết nối lại camera khi mất tín hiệu (mặc định `5`).
- `--reconnect-delay`: Thời gian chờ giữa các lần kết nối lại (mặc định `2.0` giây).

### Phím tắt khi đang quan sát
- **`M`** (hoặc `m`): Bật / Tắt toàn bộ còi hú và giọng đọc cảnh báo.
- **`Q`** (hoặc `q`): Thoát chương trình, tự động ngắt còi và giải phóng camera an toàn.

---

## 5. Quy trình Huấn luyện & Đánh giá mô hình (Phần Train)

1. **Kiểm tra dữ liệu nhãn:**
   ```powershell
   python check_dataset.py
   ```
2. **Huấn luyện mô hình YOLOv8:**
   ```powershell
   python train.py
   ```
3. **Đánh giá hiệu năng:**
   ```powershell
   python evaluate.py
   ```
Các trọng số sau huấn luyện sẽ được lưu tại thư mục `runs/fire_smoke/.../weights/best.pt`.
