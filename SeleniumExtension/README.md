# SeleniumExtension

## Mô tả
Thư mục này chứa server Flask sử dụng Selenium để tự động truy cập website và trích xuất cookies nâng cao (bao gồm cookies khó lấy, cookies động, cookies từ iframe...).

## Cài đặt

1. Cài đặt Python 3.8+ và pip.
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
   (Bao gồm: selenium, flask, flask-cors, webdriver-manager, ...)

3. Đảm bảo đã cài đặt Google Chrome trên máy.

## Chạy server

```bash
python server.py
```

Server sẽ chạy ở `http://localhost:5000`.

## API

### Trích xuất cookies

- **Endpoint:** `/extract_cookies`
- **Phương thức:** POST
- **Body (JSON):**
  ```json
  { "url": "https://example.com" }
  ```
- **Kết quả:**
  ```json
  {
    "cookies": [
      { "name": "...", "value": "...", "domain": "...", ... }
    ]
  }
  ```

## Lưu ý

- Extension sẽ gọi API này để lấy cookies nâng cao.
- Có thể cần chỉnh sửa CORS hoặc port nếu chạy trên server khác.