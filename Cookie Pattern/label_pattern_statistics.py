import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_csv_and_save(input_file_path, output_file_path=None, create_plot=True):
    try:
        # Đọc file CSV đầu vào
        df = pd.read_csv(input_file_path)
        
        # Nếu không chỉ định file đầu ra, tạo tên file tự động
        if output_file_path is None:
            file_dir = os.path.dirname(input_file_path)
            file_name = os.path.basename(input_file_path)
            output_file_path = os.path.join(file_dir, "statistics_" + file_name)
        
        # 1. Thống kê các loại nhãn ở trường "label"
        label_stats = None
        if 'label' in df.columns:
            label_stats = df['label'].value_counts().reset_index()
            label_stats.columns = ['Label', 'Count']
        
        # 2. Thống kê các loại pattern
        pattern_stats = None
        if 'pattern' in df.columns:
            pattern_stats = df['pattern'].value_counts().reset_index()
            pattern_stats.columns = ['Pattern', 'Count']
        
        # Lưu kết quả vào file CSV
        with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Phần header
            writer.writerow(['Thống kê dữ liệu từ file:', input_file_path])
            writer.writerow(['Tổng số bản ghi:', len(df)])
            writer.writerow([])
            
            # Phần thống kê nhãn
            if label_stats is not None:
                writer.writerow(['THỐNG KÊ NHÃN (LABEL)'])
                writer.writerow(['Tổng số loại nhãn khác nhau:', len(label_stats)])
                writer.writerow(['Label', 'Số lượng'])
                for _, row in label_stats.iterrows():
                    writer.writerow([row['Label'], row['Count']])
                writer.writerow([])
            
            # Phần thống kê pattern
            if pattern_stats is not None:
                writer.writerow(['THỐNG KÊ PATTERN'])
                writer.writerow(['Tổng số loại pattern khác nhau:', len(pattern_stats)])
                writer.writerow(['Pattern', 'Số lượng'])
                for _, row in pattern_stats.iterrows():
                    writer.writerow([row['Pattern'], row['Count']])
        
        # Tạo biểu đồ thống kê nhãn
        if create_plot and label_stats is not None:
            create_label_statistics_plot(df, input_file_path)
        
        print(f"Đã lưu kết quả thống kê vào file: {output_file_path}")
        return True
        
    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")
        return False

def create_label_statistics_plot(df, input_file_path):
    """
    Tạo biểu đồ thống kê các nhãn 0, 1, 2, 3, 4
    """
    try:
        # Thiết lập style cho matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Tạo figure với nhiều subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Thống kê nhãn dữ liệu\nFile: {os.path.basename(input_file_path)}', 
                     fontsize=16, fontweight='bold')
        
        # Lấy dữ liệu nhãn
        labels = df['label'].tolist()
        label_counts = Counter(labels)
        
        # Đảm bảo có đủ các nhãn từ 0-4 (nếu thiếu thì gán giá trị 0)
        all_labels = [0, 1, 2, 3, 4]
        counts = [label_counts.get(label, 0) for label in all_labels]
        
        # 1. Biểu đồ cột (Bar Chart)
        axes[0, 0].bar(all_labels, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[0, 0].set_title('Biểu đồ cột - Phân bố nhãn', fontweight='bold')
        axes[0, 0].set_xlabel('Nhãn')
        axes[0, 0].set_ylabel('Số lượng')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Thêm số liệu trên mỗi cột
        for i, count in enumerate(counts):
            axes[0, 0].text(i, count + max(counts)*0.01, str(count), 
                           ha='center', va='bottom', fontweight='bold')
        
        # 2. Biểu đồ tròn (Pie Chart)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        # Chỉ hiển thị các nhãn có giá trị > 0
        non_zero_labels = []
        non_zero_counts = []
        non_zero_colors = []
        
        for i, count in enumerate(counts):
            if count > 0:
                non_zero_labels.append(f'Nhãn {all_labels[i]}')
                non_zero_counts.append(count)
                non_zero_colors.append(colors[i])
        
        if non_zero_counts:
            wedges, texts, autotexts = axes[0, 1].pie(non_zero_counts, labels=non_zero_labels, 
                                                     colors=non_zero_colors, autopct='%1.1f%%', 
                                                     startangle=90)
            axes[0, 1].set_title('Biểu đồ tròn - Tỷ lệ phân bố nhãn', fontweight='bold')
            
            # Tăng kích thước chữ cho pie chart
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 3. Biểu đồ đường (Line Chart)
        axes[1, 0].plot(all_labels, counts, marker='o', linewidth=2, markersize=8, color='#E17055')
        axes[1, 0].fill_between(all_labels, counts, alpha=0.3, color='#E17055')
        axes[1, 0].set_title('Biểu đồ đường - xu hướng phân bố nhãn', fontweight='bold')
        axes[1, 0].set_xlabel('Nhãn')
        axes[1, 0].set_ylabel('Số lượng')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(all_labels)
        
        # Thêm số liệu trên mỗi điểm
        for i, count in enumerate(counts):
            axes[1, 0].annotate(str(count), (all_labels[i], count), 
                               textcoords="offset points", xytext=(0,10), ha='center',
                               fontweight='bold')
        
        # 4. Bảng thống kê chi tiết
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        # Tạo dữ liệu cho bảng
        table_data = []
        total_count = sum(counts)
        
        for i, count in enumerate(counts):
            percentage = (count / total_count * 100) if total_count > 0 else 0
            table_data.append([f'Nhãn {all_labels[i]}', count, f'{percentage:.1f}%'])
        
        table_data.append(['TỔNG CỘNG', total_count, '100.0%'])
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Nhãn', 'Số lượng', 'Tỷ lệ %'],
                                cellLoc='center',
                                loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Tô màu header
        for i in range(3):
            table[(0, i)].set_facecolor('#34495E')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Tô màu dòng tổng cộng
        for i in range(3):
            table[(len(table_data), i)].set_facecolor('#BDC3C7')
            table[(len(table_data), i)].set_text_props(weight='bold')
        
        axes[1, 1].set_title('Bảng thống kê chi tiết', fontweight='bold')
        
        # Điều chỉnh layout
        plt.tight_layout()
        
        # Lưu biểu đồ
        file_dir = os.path.dirname(input_file_path)
        file_name = os.path.splitext(os.path.basename(input_file_path))[0]
        plot_path = os.path.join(file_dir, f"label_statistics_{file_name}.png")
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Đã lưu biểu đồ thống kê vào file: {plot_path}")
        
        # Hiển thị biểu đồ
        plt.show()
        
        # In thống kê ra console
        print("\n" + "="*50)
        print("THỐNG KÊ NHÃN CHI TIẾT")
        print("="*50)
        for i, count in enumerate(counts):
            percentage = (count / total_count * 100) if total_count > 0 else 0
            print(f"Nhãn {all_labels[i]}: {count:>6} ({percentage:>5.1f}%)")
        print("-"*50)
        print(f"Tổng cộng: {total_count:>6} (100.0%)")
        print("="*50)
        
    except Exception as e:
        print(f"Lỗi khi tạo biểu đồ: {str(e)}")

if __name__ == "__main__":
    input_file = input("Nhập đường dẫn đến file CSV cần phân tích: ")
    output_file = input("Nhập đường dẫn file CSV đầu ra (để trống để tạo tự động): ")
    create_plot = input("Có muốn tạo biểu đồ thống kê không? (y/n, mặc định: y): ").strip().lower()
    
    if output_file.strip() == "":
        output_file = None
    
    if create_plot == "" or create_plot == "y":
        create_plot = True
    else:
        create_plot = False
        
    analyze_csv_and_save(input_file, output_file, create_plot)