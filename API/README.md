# API

## Mô tả
Đây là backend API phục vụ cho việc dự đoán mức độ rủi ro của cookies dựa trên dữ liệu đầu vào (chuỗi số hoặc mảng JSON).

## Cài đặt

1. Cài đặt Python 3.8+ và pip.
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## Chạy server

```bash
python server.py
```

Mặc định server sẽ chạy ở `http://0.0.0.0:8000`.

## Sử dụng API

### Dự đoán rủi ro cookie

- **Endpoint:** `/predict`
- **Phương thức:** POST
- **Body (JSON):**
  ```json
  {
    "sequence": [27, 51, 58, 36, ...]
  }
  ```
- **Kết quả mẫu:**
  ```json
  {
    "predicted_class": "high",
    "probabilities": [0.1, 0.2, 0.3, 0.35, 0.05]
  }
  ```

## Ghi chú
- Có thể chỉnh sửa logic dự đoán trong file `server.py`.