# Ứng Dụng AI Demo Phát Hiện Cháy Và Khói Từ Video Camera

Ứng dụng máy tính (Desktop App) chạy trên Windows cho phép người dùng chọn video camera từ máy tính, phát video trực tiếp trên giao diện đồ họa (Tkinter), phân tích khung hình theo thời gian thực bằng mô hình **Ultralytics YOLO** để phát hiện lửa và khói, vẽ bounding box, hiển thị độ tin cậy (confidence), thời điểm phát hiện và hiển thị cảnh báo trực quan trên màn hình.

---

## 1. Cấu trúc Source Code

Mã nguồn được tổ chức theo cấu trúc chuẩn Clean Code trong thư mục `src/`, phân tách rõ ràng trách nhiệm giữa giao diện (UI), dịch vụ nhận diện (Detection Service), xử lý video (Video Service) và tiện ích (Utils):

```text
fire-smoke-ai-tranning/
├── src/
│   ├── __init__.py
│   ├── main.py                     # Entry point chính của ứng dụng
│   ├── config.py                   # Quản lý cấu hình, đường dẫn model, ngưỡng conf
│   ├── services/
│   │   ├── __init__.py
│   │   ├── detection_service.py    # Nạp YOLO, quản lý model, suy luận frame
│   │   └── video_service.py        # Đọc video nền (thread-safe), điều khiển phát/dừng
│   ├── ui/
│   │   ├── __init__.py
│   │   └── main_window.py          # Giao diện Tkinter, màn hình video, panel cảnh báo
│   └── utils/
│       ├── __init__.py
│       └── video_utils.py          # Vẽ bounding box, format timestamp, resize frame
├── models/
│   └── best.pt                     # Trọng số mô hình YOLO (lớp: 0: fire, 1: smoke)
├── tests/
│   ├── test_config.py              # Test cấu hình và tự động resolve model
│   ├── test_detection_service.py   # Test nạp model và suy luận trên frame
│   ├── test_video_service.py       # Test vòng đời mở/phát/dừng video
│   ├── test_video_utils.py         # Test tiện ích chuyển đổi ảnh và vẽ box
│   └── test_alert_logic.py         # Test 4 trạng thái cảnh báo trên giao diện
├── requirements.txt                # Thư viện phụ thuộc
├── .gitignore                      # Loại trừ venv, cache, file nặng, video test
└── README.md                       # Tài liệu hướng dẫn sử dụng
```

---

## 2. Yêu cầu Môi trường

- **Hệ điều hành:** Windows 10 / 11 (64-bit).
- **Python:** Phiên bản 3.10 trở lên.
- **Phần cứng:** 
  - Khuyến nghị CPU từ 4 nhân trở lên hoặc GPU NVIDIA có hỗ trợ CUDA.
  - Chạy mượt mà trên cả CPU thông thường.

---

## 3. Cài đặt Môi trường & Dependencies

### Bước 1: Mở PowerShell và di chuyển vào thư mục dự án
```powershell
cd d:\tecnix\fire-smoke-ai-tranning
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
```powershell
# Tạo môi trường ảo venv
python -m venv venv

# Kích hoạt môi trường ảo trên Windows
.\venv\Scripts\Activate.ps1
```
*(Nếu gặp lỗi PowerShell ExecutionPolicy, mở PowerShell quyền Administrator và chạy `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`).*

### Bước 3: Cài đặt các thư viện cần thiết
```powershell
pip install -r requirements.txt
```

Các thư viện chính được sử dụng:
- `ultralytics`: Thư viện YOLOv8 để nhận diện vật thể.
- `opencv-python`: Đọc frame video và xử lý đồ họa máy tính.
- `Pillow`: Chuyển đổi khung hình sang định dạng hiển thị trong Tkinter.
- `PyYAML`: Xử lý cấu hình dataset.
- `tkinter`: Thư viện giao diện tiêu chuẩn tích hợp sẵn trong Python.

---

## 4. Vị trí đặt Model Weights

Ứng dụng tự động tìm kiếm file trọng số model theo thứ tự ưu tiên sau:
1. `models/best.pt`
2. `runs/fire_smoke/fire_v1/weights/best.pt`
3. `runs/**/best.pt`
4. `yolov8n.pt`

**Khuyến nghị:** Đặt file `best.pt` đã huấn luyện nhận diện lửa (`fire`) và khói (`smoke`) vào thư mục:
```text
models/best.pt
```

---

## 5. Hướng dẫn Khởi chạy Ứng dụng

Từ thư mục gốc của dự án, chạy lệnh:

```powershell
python -m src.main
```

Hoặc chỉ định trực tiếp file model và video qua dòng lệnh:
```powershell
python -m src.main --model models/best.pt --conf 0.25 --video sample_demo.mp4
```

### Các tham số tùy chọn dòng lệnh:
- `--model <path>`: Chỉ định đường dẫn tới file trọng số `.pt`.
- `--conf <float>`: Ngưỡng confidence ban đầu (mặc định: `0.25`).
- `--device <str>`: Thiết bị suy luận: `'cpu'` hoặc `'0'` (GPU CUDA).
- `--video <path>`: Đường dẫn file video để tự động mở khi vừa khởi động.

---

## 6. Hướng dẫn Sử dụng Giao diện

1. **Chọn video:**
   - Nhấn nút **`📁 Chọn video...`** ở góc trên bên phải.
   - Chọn file video từ máy tính (hỗ trợ các định dạng `.mp4`, `.avi`, `.mov`, `.mkv`).
   - Tên video và thời lượng sẽ xuất hiện trên thanh tiêu đề.

2. **Điều khiển phát video:**
   - **`▶ Phát`**: Bắt đầu phát video và chạy phân tích AI theo từng frame.
   - **`⏸ Tạm dừng`**: Dừng tạm thời tại frame hiện tại để quan sát chi tiết.
   - **`⏯ Tiếp tục`**: Tiếp tục phát tiếp từ thời điểm tạm dừng.
   - **`⏹ Dừng`**: Dừng phát và đưa video quay trở lại thời điểm ban đầu (00:00.0).

3. **Điều chỉnh độ nhạy (Confidence Threshold):**
   - Kéo thanh trượt **ĐỘ NHẠY (CONFIDENCE)** ở cột bên phải (từ `5%` đến `95%`).
   - Ngưỡng nhận diện sẽ được cập nhật tức thì trên các frame tiếp theo mà không cần khởi động lại ứng dụng.
   - *Mẹo:* Nếu phát hiện khói mờ hoặc lửa ở xa, hãy hạ ngưỡng xuống khoảng `15% - 25%`. Nếu có vật thể màu vàng gây báo sai, hãy nâng ngưỡng lên `35% - 50%`.

4. **Khu vực Cảnh báo (Alert Panel):**
   - **Trạng thái 1: Chưa phát hiện**
     - Nền tối / xanh lá dịu: `CHƯA PHÁT HIỆN LỬA HOẶC KHÓI`
   - **Trạng thái 2: Phát hiện lửa**
     - Nền đỏ cảnh báo: `🔥 CẢNH BÁO: PHÁT HIỆN LỬA`
     - Kèm độ tin cậy cao nhất (%) và thời điểm chính xác trong video.
   - **Trạng thái 3: Phát hiện khói**
     - Nền vàng hổ phách: `💨 CẢNH BÁO: PHÁT HIỆN KHÓI`
     - Kèm độ tin cậy cao nhất (%) và thời điểm chính xác trong video.
   - **Trạng thái 4: Phát hiện cả lửa và khói**
     - Nền đỏ thẫm tương phản cao: `🔥 CẢNH BÁO: PHÁT HIỆN LỬA VÀ KHÓI`
     - Kèm độ tin cậy của cả lửa, khói và thời điểm tương ứng.

5. **Chống nhấp nháy (Debounce/Hold):**
   - Giao diện tích hợp cơ chế giữ cảnh báo trong 10 frame kế tiếp khi detection bị chập chờn giữa các frame, giúp người quan sát không bị mỏi mắt. Khi hoàn toàn không còn đối tượng, hệ thống tự động quay về Trạng thái 1.

---

## 7. Chạy Kiểm Thử (Unit Tests)

Bộ kiểm thử tự động kiểm tra 19 kịch bản trên toàn bộ các module (cấu hình, dịch vụ suy luận, video service, format timestamp, logic cảnh báo):

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Kết quả mong đợi:
```text
Ran 19 tests in ~8.8s
OK
```

---

## 8. Các Lỗi Thường Gặp & Cách Xử Lý

| Hiện tượng | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **Hộp thoại báo "Thiếu Model Weights"** | Chưa có file `best.pt` trong thư mục `models/` hoặc `runs/` | Đặt file trọng số `best.pt` vào thư mục `models/` hoặc truyền cờ `--model <path>` |
| **Không mở được video hoặc màn hình đen** | Codec video không tương thích với OpenCV trên máy | Đảm bảo file video là MP4 với codec chuẩn H.264/AVC, hoặc dùng định dạng AVI/MKV tiêu chuẩn |
| **Tốc độ khung hình (FPS) thấp khi chạy trên CPU** | Kích thước frame lớn hoặc CPU quá tải | Vào [src/config.py](file:///d:/tecnix/fire-smoke-ai-tranning/src/config.py) chỉnh `inference_interval = 2` để suy luận xen kẽ 1 frame nhận diện / 1 frame tái sử dụng, giúp tăng gấp đôi tốc độ phát |
| **Độ nhạy nhận diện quá thấp/cao** | Ngưỡng confidence chưa phù hợp với ánh sáng video | Kéo thanh trượt `ĐỘ NHẠY (CONFIDENCE)` trực tiếp trên thanh công cụ giao diện |
| **Lỗi UnicodeEncodeError trên Console Windows** | Bảng mã console Windows dùng CP1252 | `src/main.py` đã tự động cấu hình `sys.stdout.reconfigure(encoding="utf-8")`. Nếu chạy script khác, hãy gõ `$env:PYTHONIOENCODING="utf-8"` trước khi chạy |
