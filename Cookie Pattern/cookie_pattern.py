import re
from collections import defaultdict
import csv
import json
from datetime import datetime

def parse_cookie_string(cookie_str):
    """
    Phân tích chuỗi cookie thành tên và giá trị
    
    Args:
        cookie_str (str): Chuỗi cookie cần phân tích
        
    Returns:
        tuple: (name, value) của cookie
    """
    # Kiểm tra nếu cookie đã có format name=value
    if '=' in cookie_str:
        parts = cookie_str.split('=', 1)
        return parts[0].strip(), parts[1].strip()
    
    # Nếu cookie chỉ là một giá trị không có tên
    return None, cookie_str.strip()

def extract_pattern_from_values(values):
    """
    Trích xuất pattern từ danh sách các values của cookie
    
    Args:
        values (list): Danh sách các giá trị cookie cùng tên
        
    Returns:
        str: Pattern được trích xuất
    """
    if not values or len(values) == 0:
        return ""
    
    # Kiểm tra các pattern đặc biệt dựa trên tên cookie
    if values and is_special_cookie_format(values[0]):
        special_pattern = extract_special_cookie_pattern(values)
        if special_pattern:
            return special_pattern
    
    if len(values) == 1:
        # Nếu chỉ có một giá trị, trích xuất pattern đơn giản
        return extract_simple_pattern(values[0])
    
    # Nếu có nhiều giá trị, phân tích so sánh để tìm pattern chung
    return extract_common_pattern(values)

def is_special_cookie_format(value):
    """
    Kiểm tra xem giá trị cookie có phải là định dạng đặc biệt cần xử lý riêng
    """
    # CookieConsent định dạng JSON-like
    if value.startswith('{stamp:') and ('necessary:' in value or 'preferences:' in value):
        return True
    
    # OptanonConsent với các query params
    if 'isIABGlobal=' in value and ('datestamp=' in value or 'version=' in value):
        return True
    
    # DateTime ISO format
    if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
        return True
    
    # UUID với timestamp (vuid format)
    if re.match(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}%7C\d+', value):
        return True
    
    # Player ID format (pl...)
    if re.match(r'pl\d+\.\d+', value):
        return True
    
    return False

def extract_special_cookie_pattern(values):
    """
    Trích xuất pattern cho các định dạng cookie đặc biệt
    """
    first_value = values[0]
    
    # CookieConsent định dạng JSON-like
    if first_value.startswith('{stamp:'):
        return r"\{stamp:%27[A-Za-z0-9+/=]+%27%2Cnecessary:(true|false)(%2C\w+:(true|false))*%2Cver:\d+%2Cutc:\d+%2Cregion:%27[a-z]{2}%27\}"
    
    # OptanonConsent với các query params
    if 'isIABGlobal=' in first_value:
        return r"isIABGlobal=(true|false)&datestamp=[^&]+&version=\d+\.\d+\.\d+&hosts=&consentId=[0-9a-f-]+&interactionCount=\d+&landingPath=\w+&groups=[^&]+&geolocation=[^&]+(&AwaitingReconsent=(true|false))?"
    
    # DateTime ISO format (OptanonAlertBoxClosed)
    if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', first_value):
        return r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
    
    # UUID với timestamp (vuid format)
    if re.match(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}%7C\d+', first_value):
        return r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}%7C\d+"
    
    # Player ID format (pl...)
    if re.match(r'pl\d+\.\d+', first_value):
        return r"pl\d+\.\d+"
    
    # Google Analytics
    if first_value.startswith('GA1.'):
        return r"GA1\.2\.\d+\.\d+"
    
    return None

def extract_simple_pattern(value):
    """
    Trích xuất pattern đơn giản từ một giá trị
    """
    # Xử lý riêng cho các giá trị ngắn
    if len(value) <= 3:
        if value.isdigit():
            return "[0-9]+"
        if value == "-3":  # CookieConsent specific value
            return "-[0-9]"
        return re.escape(value)
    
    # Thay thế chuỗi số liên tiếp bằng [0-9]+
    pattern = re.sub(r'\d+', '[0-9]+', value)
    
    # Xử lý các trường hợp đặc biệt
    # JWT/Base64
    if value.startswith('ey') and re.match(r'^[A-Za-z0-9+/=]+$', value):
        return "ey[A-Za-z0-9+/=]+"
    
    # Google Analytics
    if value.startswith('GA1.'):
        parts = value.split('.')
        if len(parts) == 4:
            return "GA1.2.[0-9]+.[0-9]+"
    
    # Timestamp pairs
    if re.match(r'^\d+\|\d+$', value):
        nums = value.split('|')
        if len(nums[0]) == len(nums[1]):
            return f"[0-9]{len(nums[0])}|[0-9]{len(nums[0])}" 
        else:
            return "[0-9]+|[0-9]+"
    
    return pattern

def extract_common_pattern(values):
    """
    Trích xuất pattern chung từ nhiều giá trị
    """
    # Thử tìm pattern dựa trên tên cookie thường gặp
    cookie_name = None
    
    # Kiểm tra xem tất cả các giá trị có cùng độ dài không
    lengths = [len(v) for v in values]
    if min(lengths) == max(lengths):
        # Nếu cùng độ dài, so sánh từng vị trí ký tự
        return extract_pattern_by_position(values)
    
    # Nếu khác độ dài, tìm các phần thông thường
    return extract_pattern_by_common_parts(values)

def extract_pattern_by_position(values):
    """
    Trích xuất pattern bằng cách so sánh từng vị trí ký tự
    """
    if not values:
        return ""
    
    pattern = ""
    length = len(values[0])
    
    for i in range(length):
        chars = [v[i] for v in values if i < len(v)]
        unique_chars = set(chars)
        
        if len(unique_chars) == 1:
            # Nếu ký tự ở vị trí này giống nhau trong tất cả các giá trị
            pattern += chars[0]
        elif all(c.isdigit() for c in unique_chars):
            # Nếu tất cả đều là số
            pattern += "[0-9]"
        elif all(c.isalpha() for c in unique_chars):
            # Nếu tất cả đều là chữ cái
            if all(c.isupper() for c in unique_chars):
                pattern += "[A-Z]"
            elif all(c.islower() for c in unique_chars):
                pattern += "[a-z]"
            else:
                pattern += "[A-Za-z]"
        else:
            # Nếu là hỗn hợp các loại ký tự
            pattern += "."  # Bất kỳ ký tự nào
    
    # Tối ưu hóa pattern 
    return optimize_pattern(pattern)

def extract_pattern_by_common_parts(values):
    """
    Trích xuất pattern bằng cách tìm các phần thông thường
    """
    # Kiểm tra nếu tất cả các giá trị đều khác nhau hoàn toàn
    if len(set(values)) == len(values):
        # Thử phân tích sâu hơn trước khi quyết định dùng ".*"
        if all(is_special_format(v) for v in values):
            return analyze_special_formats(values)
        
        # Thử tìm cấu trúc chung nếu định dạng tương tự nhau
        if are_structurally_similar(values):
            return derive_structural_pattern(values)
        
        return ".*"
    
    # Tìm tiền tố chung
    prefix = longest_common_prefix(values)
    # Tìm hậu tố chung
    suffix = longest_common_suffix(values)
    
    # Tạo pattern kết hợp
    if prefix and suffix:
        if prefix == suffix and len(prefix) * 2 >= len(values[0]):
            # Nếu tiền tố và hậu tố giống nhau và chiếm phần lớn giá trị
            return prefix
        else:
            middle = ".*"  # Bất kỳ ký tự nào ở giữa
            return f"{prefix}{middle}{suffix}"
    elif prefix:
        return f"{prefix}.*"
    elif suffix:
        return f".*{suffix}"
    else:
        # Không có phần chung, thử phương pháp khác
        simple_patterns = [extract_simple_pattern(v) for v in values]
        if len(set(simple_patterns)) == 1:
            # Nếu tất cả đều có cùng simple pattern
            return simple_patterns[0]
        else:
            # Thử phân tích sâu hơn trước khi quyết định dùng ".*"
            if all(is_special_format(v) for v in values):
                return analyze_special_formats(values)
            
            # Thử tìm cấu trúc chung nếu định dạng tương tự nhau
            if are_structurally_similar(values):
                return derive_structural_pattern(values)
                
            return ".*"

def are_structurally_similar(values):
    """
    Kiểm tra xem các giá trị có cấu trúc tương tự nhau không
    """
    # Chuyển đổi các giá trị thành cấu trúc đơn giản hóa để so sánh
    simplified = []
    for v in values:
        # Thay thế số bằng 'D', chữ thường bằng 'l', chữ hoa bằng 'U'
        s = re.sub(r'\d+', 'D', v)
        s = re.sub(r'[a-z]+', 'l', s)
        s = re.sub(r'[A-Z]+', 'U', s)
        simplified.append(s)
    
    # Nếu tất cả các giá trị đơn giản hóa giống nhau, chúng có cấu trúc tương tự
    return len(set(simplified)) <= 3  # Cho phép một chút biến thể

def derive_structural_pattern(values):
    """
    Tạo pattern dựa trên cấu trúc chung của các giá trị
    """
    # Chia các giá trị thành phần số và không phải số
    parts = []
    for v in values:
        part_patterns = []
        
        # Phân tích từng phần là số hoặc không phải số
        for part in re.findall(r'\d+|[^\d]+', v):
            if part.isdigit():
                part_patterns.append(f"[0-9]{len(part) if all(len(val) == len(values[0]) for val in values) else '+'}")
            else:
                # Kiểm tra xem phần này có xuất hiện trong tất cả các giá trị không
                appears_in_all = all(part in val for val in values)
                if appears_in_all:
                    part_patterns.append(re.escape(part))
                else:
                    part_patterns.append(".*")
        
        parts.append("".join(part_patterns))
    
    # Nếu tất cả các mẫu giống nhau, trả về một mẫu
    if len(set(parts)) == 1:
        return parts[0]
    
    # Nếu không, tìm pattern chung nhất
    return find_common_structural_pattern(parts)

def find_common_structural_pattern(patterns):
    """
    Tìm pattern chung cho các pattern cấu trúc
    """
    # Chia mỗi pattern thành các phần tĩnh và động
    parts = []
    for pattern in patterns:
        parts.append(re.findall(r'\[\d\]\+|\[\d\]\{\d+\}|\.\*|[^\[\.\*\+\{\}]+', pattern))
    
    # Nếu số lượng phần bằng nhau, thử kết hợp chúng
    if all(len(p) == len(parts[0]) for p in parts):
        result = []
        for i in range(len(parts[0])):
            segment_parts = [p[i] for p in parts]
            if len(set(segment_parts)) == 1:
                result.append(segment_parts[0])
            else:
                # Nếu tất cả đều là các pattern số
                if all(re.match(r'\[\d\]', p) for p in segment_parts):
                    result.append("[0-9]+")
                else:
                    result.append(".*")
        
        return "".join(result)
    
    # Nếu số lượng phần khác nhau, trả về pattern tổng quát hơn
    prefixes = [re.escape(longest_common_prefix(patterns))]
    suffixes = [re.escape(longest_common_suffix(patterns))]
    
    # Loại bỏ các tiền tố/hậu tố trống
    parts = [p for p in prefixes + suffixes if p]
    
    if not parts:
        return ".*"
    
    if len(parts) == 1:
        return f"{parts[0]}.*"
    
    return f"{parts[0]}.*{parts[1]}"

def is_special_format(value):
    """
    Kiểm tra xem giá trị có thuộc định dạng đặc biệt nào không
    """
    # Kiểm tra JWT/Base64
    if value.startswith('ey') and re.match(r'^[A-Za-z0-9+/=]+$', value):
        return True
    
    # Kiểm tra Google Analytics
    if value.startswith('GA1.'):
        return True
    
    # Kiểm tra timestamp pairs
    if re.match(r'^\d+\|\d+$', value):
        return True
    
    # Kiểm tra UUID
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value.lower()):
        return True
    
    # Kiểm tra CookieConsent JSON-like
    if value.startswith('{stamp:'):
        return True
    
    # Kiểm tra ISO date
    try:
        if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
            return True
    except:
        pass
    
    # Kiểm tra URL encoded params
    if '=' in value and '%' in value and '&' in value:
        return True
    
    return False

def analyze_special_formats(values):
    """
    Phân tích các giá trị có định dạng đặc biệt để tìm pattern phù hợp
    """
    # Kiểm tra JWT/Base64
    if all(v.startswith('ey') and re.match(r'^[A-Za-z0-9+/=]+$', v) for v in values):
        return "ey[A-Za-z0-9+/=]+"
    
    # Kiểm tra Google Analytics
    if all(v.startswith('GA1.') for v in values):
        return "GA1.2.[0-9]+.[0-9]+"
    
    # Kiểm tra timestamp pairs
    if all(re.match(r'^\d+\|\d+$', v) for v in values):
        return "[0-9]+|[0-9]+"
    
    # Kiểm tra UUID
    if all(re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', v.lower()) for v in values):
        return "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    
    # Kiểm tra format CookieConsent
    if all(v.startswith('{stamp:') for v in values):
        return r"\{stamp:%27[A-Za-z0-9+/=]+%27%2C.*\}"
    
    # Kiểm tra ISO dates
    if all(re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', v) for v in values):
        return r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
    
    # Kiểm tra URL encoded params
    if all('=' in v and '%' in v and '&' in v for v in values):
        # Tìm các tham số chung
        common_params = find_common_url_params(values)
        if common_params:
            return common_params
    
    return ".*"

def find_common_url_params(values):
    """
    Tìm các tham số chung trong các chuỗi URL encoded
    """
    # Parse các cặp key-value từ chuỗi
    all_params = []
    for v in values:
        params = {}
        try:
            # Phân tích chuỗi URL-encoded một cách thủ công
            parts = v.split('&')
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    params[key] = value
        except:
            continue
        all_params.append(params)
    
    if not all_params:
        return None
    
    # Tìm các khóa chung trong tất cả các giá trị
    common_keys = set(all_params[0].keys())
    for params in all_params[1:]:
        common_keys &= set(params.keys())
    
    if not common_keys:
        return None
    
    # Tạo pattern với các khóa thông thường
    common_keys = sorted(list(common_keys))
    pattern_parts = []
    
    for key in common_keys:
        pattern_parts.append(f"{key}=[^&]+")
    
    return "&".join(pattern_parts) + "(&[^&]+)*"

def longest_common_prefix(strings):
    """Tìm tiền tố chung dài nhất của danh sách các chuỗi"""
    if not strings:
        return ""
    
    shortest = min(strings, key=len)
    for i, char in enumerate(shortest):
        for other in strings:
            if other[i] != char:
                return shortest[:i]
    return shortest

def longest_common_suffix(strings):
    """Tìm hậu tố chung dài nhất của danh sách các chuỗi"""
    if not strings:
        return ""
    
    # Đảo ngược các chuỗi, tìm tiền tố chung, sau đó đảo ngược lại
    reversed_strings = [s[::-1] for s in strings]
    common_prefix = longest_common_prefix(reversed_strings)
    return common_prefix[::-1]

def optimize_pattern(pattern):
    """
    Tối ưu hóa pattern bằng cách gộp các ký tự liên tiếp giống nhau
    """
    # Gộp các ký tự liên tiếp giống nhau
    optimized = pattern
    replacements = [
        (r'(\[0-9\])\1+', r'\1+'),
        (r'(\[A-Za-z\])\1+', r'\1+'),
        (r'(\[A-Z\])\1+', r'\1+'),
        (r'(\[a-z\])\1+', r'\1+'),
        (r'(\.)\1+', r'\1+')
    ]
    
    for pattern, replacement in replacements:
        optimized = re.sub(pattern, replacement, optimized)
    
    return optimized

def analyze_cookies(cookie_list, cookie_names=None):
    """
    Phân tích danh sách cookies để tìm pattern theo tên cookie
    
    Args:
        cookie_list (list): Danh sách các chuỗi cookie
        cookie_names (list): Danh sách tên cookie cần chú ý đặc biệt
        
    Returns:
        dict: Dictionary với key là tên cookie và value là pattern
    """
    # Nhóm cookies theo tên
    cookie_groups = defaultdict(list)
    
    for cookie_str in cookie_list:
        name, value = parse_cookie_string(cookie_str)
        if name is None:
            name = "_unnamed_cookie_"
        cookie_groups[name].append(value)
    
    # Trích xuất pattern cho mỗi nhóm
    cookie_patterns = {}
    for name, values in cookie_groups.items():
        # Đưa tên cookie cho hàm extract_pattern nếu cần pattern đặc biệt
        pattern = extract_pattern_from_values(values)
        
        # Xử lý đặc biệt cho một số cookie phổ biến
        if name == "CookieConsent" and pattern == ".*":
            if any(v.startswith('{stamp:') for v in values):
                pattern = r"\{stamp:%27[A-Za-z0-9+/=]+%27%2Cnecessary:(true|false)(%2C\w+:(true|false))*%2Cver:\d+%2Cutc:\d+%2Cregion:%27[a-z]{2}%27\}"
            elif any(v == "-3" for v in values):
                pattern = "-[0-9]"
        
        elif name == "_ga" and pattern == ".*":
            pattern = "GA1.2.[0-9]+.[0-9]+"
        
        elif name == "OptanonAlertBoxClosed" and pattern == ".*":
            pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
        
        elif name == "OptanonConsent" and pattern == ".*":
            pattern = r"isIABGlobal=(true|false)&datestamp=[^&]+&version=\d+\.\d+\.\d+&hosts=&consentId=[0-9a-f-]+&interactionCount=\d+&landingPath=\w+&groups=[^&]+&geolocation=[^&]+(&AwaitingReconsent=(true|false))?"
        
        elif name == "vuid" and pattern == ".*":
            if any('%7C' in v for v in values):
                pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}%7C\d+"
            elif any(re.match(r'pl\d+\.\d+', v) for v in values):
                pattern = r"pl\d+\.\d+"
        
        
        cookie_patterns[name] = {
            "pattern": pattern,
            "count": len(values),
            "examples": values  # xuất hết values
        }
    
    return cookie_patterns

def run_analysis(cookie_list):
    """
    Chạy phân tích danh sách cookies và hiển thị kết quả
    
    Args:
        cookie_list (list): Danh sách các chuỗi cookie
    """
    print("===== PHÂN TÍCH PATTERN COOKIE =====\n")
    
    # Phân tích cookies
    cookie_patterns = analyze_cookies(cookie_list)
    
    # Hiển thị kết quả
    print(f"Tìm thấy {len(cookie_patterns)} nhóm cookie:\n")
    
    for name, data in cookie_patterns.items():
        print(f"Cookie: {name}")
        print(f"Pattern: {data['pattern']}")
        print(f"Số lượng: {data['count']}")
        print(f"Ví dụ: {', '.join(data['examples'][:3] if len(data['examples']) > 3 else data['examples'])}")
        print()

def read_cookies_from_csv(input_file):
    cookies = []
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if 'name' in reader.fieldnames and 'value' in reader.fieldnames:
            for row in reader:
                cookies.append((row['name'], row['value']))
        elif 'cookie' in reader.fieldnames:
            for row in reader:
                name, value = parse_cookie_string(row['cookie'])
                cookies.append((name, value))
        else:
            raise Exception("CSV phải có cột 'name,value' hoặc 'cookie'")
    return cookies

def write_patterns_to_csv(output_file, pattern_dict):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'value', 'pattern'])
        for name, data in pattern_dict.items():
            for value in data['examples']:
                writer.writerow([name, value, data['pattern']])

def analyze_cookies_csv(input_file, output_file):
    cookies = read_cookies_from_csv(input_file)
    # cookies: list of (name, value)
    cookie_groups = defaultdict(list)
    for name, value in cookies:
        if name is None:
            name = "_unnamed_cookie_"
        cookie_groups[name].append(value)
    # Trích xuất pattern cho mỗi nhóm
    cookie_patterns = {}
    for name, values in cookie_groups.items():
        pattern = extract_pattern_from_values(values)
        
        # Xử lý đặc biệt cho một số cookie phổ biến
        if name == "CookieConsent" and pattern == ".*":
            if any(v.startswith('{stamp:') for v in values):
                pattern = r"\{stamp:%27[A-Za-z0-9+/=]+%27%2Cnecessary:(true|false)(%2C\w+:(true|false))*%2Cver:\d+%2Cutc:\d+%2Cregion:%27[a-z]{2}%27\}"
            elif any(v == "-3" for v in values):
                pattern = "-[0-9]"
        
        elif name == "_ga" and pattern == ".*":
            pattern = "GA1.2.[0-9]+.[0-9]+"
        
        elif name == "OptanonAlertBoxClosed" and pattern == ".*":
            pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
        
        elif name == "OptanonConsent" and pattern == ".*":
            pattern = r"isIABGlobal=(true|false)&datestamp=[^&]+&version=\d+\.\d+\.\d+&hosts=&consentId=[0-9a-f-]+&interactionCount=\d+&landingPath=\w+&groups=[^&]+&geolocation=[^&]+(&AwaitingReconsent=(true|false))?"
        
        elif name == "vuid" and pattern == ".*":
            if any('%7C' in v for v in values):
                pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}%7C\d+"
            elif any(re.match(r'pl\d+\.\d+', v) for v in values):
                pattern = r"pl\d+\.\d+"
        
        cookie_patterns[name] = {
            "pattern": pattern,
            "count": len(values),
            "examples": values  # xuất hết value
        }
    write_patterns_to_csv(output_file, cookie_patterns)
    print(f"Đã ghi kết quả ra {output_file}")

if __name__ == "__main__":
    # Ví dụ sử dụng:
    # python pattern_analyze.py input.csv output.csv
    import sys
    if len(sys.argv) == 3:
        analyze_cookies_csv(sys.argv[1], sys.argv[2])
    else:
        print("Cách dùng: python pattern_analyze.py input.csv output.csv")