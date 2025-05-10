# Chrome Extension: Cookie Risk Prediction Tool

## Mô tả
Extension cho phép trích xuất cookies từ trang web hiện tại, gửi dữ liệu lên API để dự đoán mức độ rủi ro, và hiển thị kết quả trực quan.

## Cài đặt

1. Mở Chrome, vào `chrome://extensions/`.
2. Bật **Chế độ dành cho nhà phát triển** (Developer mode).
3. Nhấn **Tải tiện ích đã giải nén** (Load unpacked) và chọn thư mục `Extension`.

## Sử dụng

- **Trích xuất cookies:** Nhấn nút "Trích xuất cookies" để lấy cookies từ trang hiện tại.
- **Lấy cookies nâng cao (Selenium):** Nhấn nút này để gửi yêu cầu về server Selenium (nếu server đang chạy).
- **Dự đoán:** Sau khi có dữ liệu, nhấn "Dự đoán" để gửi lên API và xem kết quả.
- **Xóa:** Xóa dữ liệu hiện tại.

## Cấu hình

- Có thể thay đổi URL API trong giao diện extension.

## Yêu cầu

- Để sử dụng tính năng Selenium, cần chạy server SeleniumExtension trên máy.