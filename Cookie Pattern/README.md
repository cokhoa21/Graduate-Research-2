# Cookie Pattern Analysis

Thư mục này chứa các script để phân tích và xử lý dữ liệu cookies từ các website.

## Cấu trúc thư mục

```
Cookie Pattern/
├── input/          # Thư mục chứa dữ liệu đầu vào
├── output/         # Thư mục chứa kết quả đầu ra
├── filtered/       # Thư mục chứa dữ liệu đã được lọc
├── statistics/     # Thư mục chứa các báo cáo thống kê
├── cookie_pattern.py           # Script chính để phân tích pattern của cookies
├── cookie_statistics.py        # Script để tạo báo cáo thống kê
├── filter_patterns.py          # Script để lọc các pattern cụ thể
├── remove_duplicate_cookies.py # Script để loại bỏ cookies trùng lặp
└── result_data.csv            # File dữ liệu kết quả
```

## Mô tả các file

### 1. cookie_pattern.py
Script chính để phân tích pattern của cookies từ các website.
- Đọc dữ liệu cookies từ file input
- Phân tích các pattern và đặc điểm của cookies
- Lưu kết quả phân tích vào file output

### 2. cookie_statistics.py
Script để tạo báo cáo thống kê về cookies.
- Tạo các báo cáo thống kê chi tiết
- Phân tích tần suất xuất hiện của các loại cookies
- Xuất kết quả dưới dạng báo cáo

### 3. filter_patterns.py
Script để lọc các pattern cụ thể từ dữ liệu cookies.
- Lọc cookies theo các tiêu chí nhất định
- Tạo các bộ lọc tùy chỉnh
- Xuất kết quả đã lọc

### 4. remove_duplicate_cookies.py
Script để loại bỏ các cookies trùng lặp dựa trên tên và giá trị.
- Đọc dữ liệu từ file CSV
- Loại bỏ các dòng trùng lặp (giữ lại bản ghi đầu tiên)
- Lưu kết quả vào file CSV mới

## Cách sử dụng

### 1. Phân tích pattern cookies
```bash
python cookie_pattern.py <input_file> <output_file>
```

### 2. Tạo báo cáo thống kê
```bash
python cookie_statistics.py <input_file> <output_file>
```

### 3. Lọc pattern
```bash
python filter_patterns.py <input_file> <output_file>
```

### 4. Loại bỏ cookies trùng lặp
```bash
python remove_duplicate_cookies.py <input_file> <output_file>
```

## Yêu cầu
- Python 3.x
- pandas
- Các thư viện Python khác (nếu có)

## Lưu ý
- Đảm bảo các thư mục input, output, filtered và statistics đã được tạo trước khi chạy script
- File input phải ở định dạng CSV và có các cột cần thiết (name, value, etc.)
- Kết quả sẽ được lưu vào thư mục output với tên file được chỉ định 