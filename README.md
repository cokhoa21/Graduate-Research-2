# Cookie Risk Prediction Project

## Tổng quan

Dự án này cung cấp một hệ thống hoàn chỉnh để:
- **Trích xuất cookies** từ trình duyệt (Chrome Extension) hoặc tự động (Selenium).
- **Dự đoán mức độ rủi ro của cookies** bằng mô hình học máy.
- **Hiển thị kết quả trực quan** cho người dùng cuối.

## Cấu trúc thư mục
.
├── API/ # Backend API phục vụ dự đoán
├── Extension/ # Chrome Extension trích xuất và dự đoán cookies
├── SeleniumExtension/ # Flask server + Selenium để cào cookies nâng cao
├── train/ # Code và tài nguyên huấn luyện mô hình
├── README.md # File hướng dẫn tổng thể (bạn đang đọc)


## Hướng dẫn nhanh

### 1. Cài đặt môi trường

- Đảm bảo đã cài Python 3.8+ và pip.
- Đảm bảo đã cài Google Chrome (cho Selenium).

### 2. Cài đặt các thành phần

**API:**
```bash
cd API
pip install -r requirements.txt
python server.py
```

**SeleniumExtension:**
```bash
cd SeleniumExtension
pip install -r requirements.txt
python server.py
```

**Extension:**
- Mở Chrome, vào `chrome://extensions/`
- Bật Developer mode
- Chọn “Load unpacked” và trỏ tới thư mục `Extension`

**Train:**
```bash
cd train
pip install -r requirements.txt
python train.py
```

### 3. Sử dụng

- **Trích xuất cookies:** Dùng Extension trên Chrome, nhấn “Trích xuất cookies”.
- **Lấy cookies nâng cao:** Nhấn “Lấy cookies nâng cao (Selenium)” (yêu cầu server SeleniumExtension đang chạy).
- **Dự đoán:** Nhấn “Dự đoán” để gửi dữ liệu lên API và xem kết quả.
- **Huấn luyện lại model:** Chạy script trong thư mục `train`.

### 4. Tích hợp & mở rộng

- Extension có thể gọi API Flask/Selenium để lấy cookies nâng cao khi cần.
- Có thể thay đổi URL API trong giao diện Extension.
- Có thể mở rộng backend, model, hoặc giao diện theo nhu cầu.


## Liên hệ

- Tác giả: Cồ Huy Khoa
- Email: huykhoa21@gmail.com

---

**Xem thêm hướng dẫn chi tiết trong từng thư mục con!**
