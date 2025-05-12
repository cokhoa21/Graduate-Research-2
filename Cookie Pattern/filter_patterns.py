import csv
import sys

def read_patterns_from_csv(input_file):
    """
    Đọc dữ liệu từ file CSV chứa các pattern
    
    Args:
        input_file (str): Đường dẫn đến file CSV đầu vào
        
    Returns:
        list: Danh sách các dòng dữ liệu từ CSV
    """
    patterns = []
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patterns.append(row)
    return patterns

def filter_patterns(patterns):
    """
    Lọc và loại bỏ các pattern là ".*"
    
    Args:
        patterns (list): Danh sách các pattern cần lọc
        
    Returns:
        tuple: (patterns_filtered, patterns_removed) - danh sách pattern đã lọc và danh sách pattern bị loại bỏ
    """
    patterns_filtered = []
    patterns_removed = []
    
    for pattern in patterns:
        if pattern['pattern'] == '.*':
            patterns_removed.append(pattern)
        else:
            patterns_filtered.append(pattern)
    
    return patterns_filtered, patterns_removed

def write_patterns_to_csv(patterns, output_file):
    """
    Ghi danh sách pattern ra file CSV
    
    Args:
        patterns (list): Danh sách các pattern cần ghi
        output_file (str): Đường dẫn đến file CSV đầu ra
    """
    if not patterns:
        print(f"Không có pattern nào để ghi ra file {output_file}")
        return
        
    fieldnames = patterns[0].keys()
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(patterns)

def main():
    if len(sys.argv) != 4:
        print("Cách dùng: python filter_patterns.py input.csv filtered_output.csv removed_output.csv")
        sys.exit(1)
        
    input_file = sys.argv[1]
    filtered_output = sys.argv[2]
    removed_output = sys.argv[3]
    
    try:
        # Đọc dữ liệu từ file CSV
        patterns = read_patterns_from_csv(input_file)
        
        if not patterns:
            print(f"Không tìm thấy dữ liệu trong file {input_file}")
            sys.exit(1)
        
        # Lọc pattern
        patterns_filtered, patterns_removed = filter_patterns(patterns)
        
        # Ghi kết quả ra file
        write_patterns_to_csv(patterns_filtered, filtered_output)
        write_patterns_to_csv(patterns_removed, removed_output)
        
        print(f"\nĐã ghi các pattern đã lọc ra {filtered_output}")
        print(f"Đã ghi các pattern bị loại bỏ ra {removed_output}")
        
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 