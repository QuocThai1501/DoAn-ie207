# TÀI NGUYÊN CUỐI KÌ  
**IE207.P21 - Đồ án**  

## Thông tin cơ bản  
**GVHD:** TS. Nguyễn Văn Kiệt
**Tên đề tài:** XÂY DỰNG HỆ THỐNG KHUYẾN NGHỊ KHÁCH SẠN DỰA VÀO DỮ LIỆU DU KHÁCH

## Danh sách thành viên  

| STT | Họ và tên           | MSSV     |
|-----|---------------------|----------|
| 1   | Lê Quốc Thái        | 22521318 |
| 2   | Lê Thái Khánh Ngân  | 22520930 |

## Danh sách liên kết  
- **Drive:** [https://drive.google.com/drive/folders/1zlo_rdoK3LOt8TYgrQh-kYfLOD1Cw8Va?usp=sharing]
- **Github:** [https://github.com/QuocThai1501/DoAn-ie207.git]
- **Link đến video demo:** Đang bổ sung

## Tổ chức folder

    📦 DoAn-ie207
    ├── 📂 code/                                # Chứa các file code trước khi xây dựng mô hình
    ├── 📂 hotel_links_booking/                 # 63 file txt link các trang chi tiết khách sạn của mỗi tỉnh
    ├── 📂 hotels_data/                         # 63 file csv thuộc tính các khách sạn của mỗi tỉnh
    ├── 📂 Hotel_Recommendation_System/         # Chứa các file code mô hình, code demo giao diện
    ├── 📂 Traveloka/                           # Code cào link trang Traveloka và các file dữ liệu
    ├── 📄 .gitignore                           # Bỏ qua file không cần thiết
    ├── 📄 hotels_data_final                    # Dữ liệu khách sạn 63 tỉnh sau khi tiền xử lý
    ├── 📄 tourist_dataset_10k                  # Dữ liệu du khách giả lập
    └── 📄 README.md                            # Tài nguyên cuối kì, hướng dẫn cài đặt và sử dụng hệ thống
    
## Hướng dẫn cài đặt và sử dụng hệ thống trên local

### Cài đặt thủ công

1. **Clone repository**
```bash
git clone https://github.com/QuocThai1501/DoAn-ie207.git
```

2. **Khởi chạy development**
   - Terminal:
     ```bash
    cd Hotel_Recommendation_System
    streamlit run demo1.py # nếu muốn khởi chạy demo 1
    streamlit run demo2.py # nếu muốn khởi chạy demo 2
    streamlit run demo3.py # nếu muốn khởi chạy demo 3
     ```
     

### Hướng dẫn sử dụng hệ thống

    - Demo 1: nhập giá trị ID du khách (từ 1 đến 10000) vào ô rồi nhấn enter