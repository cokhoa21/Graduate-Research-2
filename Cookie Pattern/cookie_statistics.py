import csv
from collections import defaultdict
import re

def get_cookie_category(name, pattern, labels):
    """
    Phân loại cookie dựa trên tên, pattern và labels
    
    Args:
        name (str): Tên cookie
        pattern (str): Pattern của cookie
        labels (list): Danh sách các label
        
    Returns:
        str: Loại cookie
    """
    # Phân loại dựa trên tên cookie
    name_lower = name.lower()
    
    # Session ID cookies
    if name_lower == "phpsessid":
        return "PHP Session ID"
    elif name_lower == "jsessionid":
        return "Java Session ID"
    elif name_lower == "loc":
        return "Session ID"
    elif any(x in name_lower for x in ['session', 'sess', 'sid']):
        return "Session ID"
    
    # Google Analytics & Ads cookies
    if name_lower.startswith('_ga'):
        return "Google Analytics"
    elif name_lower == "__gads":
        return "Google Ads"
    elif name_lower.startswith('_gid'):
        return "Google Analytics"
    elif name_lower.startswith('_gat'):
        return "Google Analytics"
    elif name_lower == "ide":
        return "Google Ads"
    
    # Facebook cookies
    if name_lower == "_fbp":
        return "Facebook Pixel"
    elif name_lower.startswith('_fb'):
        return "Facebook"
    
    # LinkedIn cookies
    if name_lower == "bcookie":
        return "LinkedIn"
    elif name_lower.startswith('li_'):
        return "LinkedIn"
    
    # YouTube cookies
    if name_lower == "visitor_info1_live":
        return "YouTube"
    elif name_lower == "ysc":
        return "YouTube"
    elif name_lower == "vuid":
        return "Vimeo"
    
    # UUID cookies
    if name_lower == "uuid":
        return "UUID"
    elif re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', pattern or ''):
        return "UUID"
    
    # Authentication cookies
    if any(x in name_lower for x in ['auth', 'token', 'jwt', 'login', 'user']):
        return "Authentication"
    
    # Security cookies
    if any(x in name_lower for x in ['security', 'secure', 'csrf', 'xsrf']):
        return "Security"
    
    # Preference cookies
    if any(x in name_lower for x in ['pref', 'preference', 'settings', 'config']):
        return "Preference"
    
    # Phân loại dựa trên pattern
    if pattern:
        # Google Analytics pattern
        if re.search(r'\d+\.\d+\.\d+\.\d+\.\d+\.\d+', pattern):
            return "Google Analytics"
        
        # Facebook Pixel pattern
        if re.search(r'fb\.1\.\d+\.\d+', pattern):
            return "Facebook Pixel"
        
        # UUID pattern
        if re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', pattern):
            return "UUID"
        
        # Session ID patterns
        if re.search(r'[a-z0-9]{26,32}', pattern):
            return "PHP Session ID"
        if re.search(r'[0-9]{4}[a-zA-Z0-9]{4,20}:[0-9a-zA-Z]{8,10}', pattern):
            return "Java Session ID"
    
    # Phân loại dựa trên labels
    if labels:
        label_str = '|'.join(labels).lower()
        
        # Analytics
        if any(x in label_str for x in ['analytics', 'statistics']):
            return "Analytics"
        
        # Advertising
        if any(x in label_str for x in ['advertising', 'marketing']):
            return "Advertising"
        
        # Necessary
        if any(x in label_str for x in ['necessary', 'essential']):
            return "Necessary"
        
        # Preference
        if any(x in label_str for x in ['preference', 'preferences']):
            return "Preference"
        
        # Social Media
        if any(x in label_str for x in ['social', 'social media']):
            return "Social Media"
    
    return "Other"

def analyze_label_distribution(labels):
    """
    Phân tích phân phối của các label
    
    Args:
        labels (list): Danh sách các label
        
    Returns:
        dict: Thống kê về phân phối label
    """
    label_stats = {}
    for label in labels:
        if label not in label_stats:
            label_stats[label] = 0
        label_stats[label] += 1
    
    # Sắp xếp theo số lượng giảm dần
    sorted_stats = dict(sorted(label_stats.items(), key=lambda x: x[1], reverse=True))
    return sorted_stats

def write_pattern_summary_to_csv(output_file, pattern_dict):
    """
    Write a summary of patterns grouped by cookie category
    """
    # Tạo dictionary để nhóm theo category
    category_groups = defaultdict(list)
    
    for name, data in pattern_dict.items():
        # Collect labels
        labels = set()
        for cookie in data['examples']:
            labels.add(cookie['label'])
        
        # Xác định category
        category = get_cookie_category(name, data['pattern'], list(labels))
        
        # Phân tích phân phối label
        label_distribution = analyze_label_distribution([cookie['label'] for cookie in data['examples']])
        
        # Collect domains
        domains = set()
        for cookie in data['examples']:
            domains.add(cookie['domain'])
        
        category_groups[category].append({
            'name': name,
            'pattern': data['pattern'],
            'count': data['count'],
            'labels': '|'.join(sorted(labels)),
            'label_distribution': label_distribution,
            'domains': '|'.join(sorted(domains))
        })
    
    # Ghi ra file CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['category', 'cookie_name', 'pattern', 'count', 'labels', 
                     'label_distribution', 'domains']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sắp xếp categories theo thứ tự ưu tiên
        category_order = [
            "PHP Session ID", "Java Session ID", "Session ID",
            "Google Analytics", "Google Ads",
            "Facebook Pixel", "Facebook",
            "LinkedIn", "YouTube", "Vimeo",
            "UUID", "Authentication", "Security",
            "Preference", "Analytics", "Advertising",
            "Necessary", "Social Media", "Other"
        ]
        
        # Sắp xếp categories theo thứ tự ưu tiên
        sorted_categories = sorted(
            category_groups.keys(),
            key=lambda x: category_order.index(x) if x in category_order else len(category_order)
        )
        
        for category in sorted_categories:
            cookies = category_groups[category]
            # Sắp xếp cookies trong mỗi category theo count giảm dần
            cookies.sort(key=lambda x: x['count'], reverse=True)
            
            for cookie in cookies:
                row = {
                    'category': category,
                    'cookie_name': cookie['name'],
                    'pattern': cookie['pattern'],
                    'count': cookie['count'],
                    'labels': cookie['labels'],
                    'label_distribution': '; '.join([f"{label}: {count}" for label, count in cookie['label_distribution'].items()]),
                    'domains': cookie['domains']
                }
                writer.writerow(row)

def write_unique_patterns_to_csv(output_file, pattern_dict):
    """
    Write unique patterns to a separate CSV file, including metadata about the pattern
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'pattern', 'count', 'domains', 'paths', 'first_party_domains', 
                     'labels', 'label_distribution', 'cookie_category', 'cmp_origins', 
                     'session_count', 'http_only_count', 'host_only_count', 'secure_count', 
                     'same_site_values']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for name, data in pattern_dict.items():
            # Collect unique values for each field
            domains = set()
            paths = set()
            first_party_domains = set()
            labels = set()
            cmp_origins = set()
            session_count = 0
            http_only_count = 0
            host_only_count = 0
            secure_count = 0
            same_site_values = set()
            
            for cookie in data['examples']:
                domains.add(cookie['domain'])
                paths.add(cookie['path'])
                first_party_domains.add(cookie['first_party_domain'])
                labels.add(cookie['label'])
                cmp_origins.add(cookie['cmp_origin'])
                if cookie['session'] == 'true':
                    session_count += 1
                if cookie['http_only'] == 'true':
                    http_only_count += 1
                if cookie['host_only'] == 'true':
                    host_only_count += 1
                if cookie['secure'] == 'true':
                    secure_count += 1
                same_site_values.add(cookie['same_site'])
            
            # Phân tích phân phối label
            label_distribution = analyze_label_distribution([cookie['label'] for cookie in data['examples']])
            label_distribution_str = '; '.join([f"{label}: {count}" for label, count in label_distribution.items()])
            
            # Xác định loại cookie
            cookie_category = get_cookie_category(name, data['pattern'], list(labels))
            
            row = {
                'name': name,
                'pattern': data['pattern'],
                'count': data['count'],
                'domains': '|'.join(sorted(domains)),
                'paths': '|'.join(sorted(paths)),
                'first_party_domains': '|'.join(sorted(first_party_domains)),
                'labels': '|'.join(sorted(labels)),
                'label_distribution': label_distribution_str,
                'cookie_category': cookie_category,
                'cmp_origins': '|'.join(sorted(cmp_origins)),
                'session_count': session_count,
                'http_only_count': http_only_count,
                'host_only_count': host_only_count,
                'secure_count': secure_count,
                'same_site_values': '|'.join(sorted(same_site_values))
            }
            writer.writerow(row)

def analyze_cookies_statistics(input_file, pattern_summary_file, category_summary_file):
    """
    Phân tích thống kê cookies từ file input và xuất ra các file thống kê
    
    Args:
        input_file (str): Đường dẫn đến file input CSV chứa dữ liệu cookie
        pattern_summary_file (str): Đường dẫn đến file output CSV chứa thống kê pattern
        category_summary_file (str): Đường dẫn đến file output CSV chứa thống kê theo category
    """
    # Đọc dữ liệu từ file input
    cookies = []
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        required_fields = ['visit_id', 'name', 'domain', 'path', 'first_party_domain', 
                         'label', 'cmp_origin', 'value', 'session', 'http_only', 
                         'host_only', 'secure', 'same_site']
        
        # Kiểm tra các trường bắt buộc
        missing_fields = [field for field in required_fields if field not in reader.fieldnames]
        if missing_fields:
            raise Exception(f"CSV file thiếu các trường bắt buộc: {', '.join(missing_fields)}")
        
        for row in reader:
            cookie_data = {
                'visit_id': row['visit_id'],
                'name': row['name'],
                'domain': row['domain'],
                'path': row['path'],
                'first_party_domain': row['first_party_domain'],
                'label': row['label'],
                'cmp_origin': row['cmp_origin'],
                'value': row['value'],
                'session': row['session'],
                'http_only': row['http_only'],
                'host_only': row['host_only'],
                'secure': row['secure'],
                'same_site': row['same_site']
            }
            cookies.append(cookie_data)
    
    # Nhóm cookies theo tên
    cookie_groups = defaultdict(list)
    for cookie in cookies:
        name = cookie['name']
        if name is None:
            name = "_unnamed_cookie_"
        cookie_groups[name].append(cookie)
    
    # Tạo pattern dictionary
    pattern_dict = {}
    for name, cookies in cookie_groups.items():
        values = [cookie['value'] for cookie in cookies]
        pattern_dict[name] = {
            "pattern": values[0],  # Sử dụng giá trị đầu tiên làm pattern
            "count": len(cookies),
            "examples": cookies
        }
    
    # Xuất thống kê
    write_unique_patterns_to_csv(pattern_summary_file, pattern_dict)
    write_pattern_summary_to_csv(category_summary_file, pattern_dict)
    
    print(f"Đã ghi thống kê pattern ra {pattern_summary_file}")
    print(f"Đã ghi thống kê theo category ra {category_summary_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Cách dùng: python cookie_statistics.py input.csv pattern_summary.csv category_summary.csv")
        print("\nTrong đó:")
        print("  input.csv: File CSV chứa dữ liệu cookie đầu vào")
        print("  pattern_summary.csv: File CSV chứa thống kê pattern")
        print("  category_summary.csv: File CSV chứa thống kê theo category")
        sys.exit(1)
    
    try:
        analyze_cookies_statistics(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        sys.exit(1) 