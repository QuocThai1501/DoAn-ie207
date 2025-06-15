# TÀI NGUYÊN CUỐI KÌ  
**IE207.P21 - Đồ án**   
**GVHD:** TS. Nguyễn Văn Kiệt  
**Tên đề tài:** Xây dựng hệ thống khuyến nghị khách sạn dựa vào dữ liệu du khách

## Danh sách thành viên  

| STT | Họ và tên           | MSSV     |
|-----|---------------------|----------|
| 1   | Lê Quốc Thái        | 22521318 |
| 2   | Lê Thái Khánh Ngân  | 22520930 |

## Danh sách liên kết

- **Drive:** [https://drive.google.com/drive/folders/1F7JppmD3FCvqKGlkcCIHELLuSxC2Xr70?usp=sharing]
- **Github:** [https://github.com/QuocThai1501/DoAn-ie207.git]

## Tổ chức folder

    📦 DoAn-ie207
    ├── 📂 code/                           # Chứa các file code trước khi xây dựng mô hình
    ├── 📂 hotel_links_booking/            # 63 file txt link các trang chi tiết khách sạn của mỗi tỉnh
    ├── 📂 hotels_data/                    # 63 file csv thuộc tính các khách sạn của mỗi tỉnh
    ├── 📂 Hotel_Recommendation_System/    # Chứa các file code mô hình, code demo giao diện
    ├── 📂 Traveloka/                      # Code cào link trang Traveloka và các file dữ liệu
    ├── 📄 .gitignore                      # Bỏ qua file không cần thiết
    ├── 📄 hotels_data_final.csv           # Dữ liệu khách sạn 63 tỉnh sau khi tiền xử lý
    ├── 📄 tourist_dataset_10k.csv         # Dữ liệu du khách giả lập
    └── 📘 README.md                       # Tài nguyên cuối kì, hướng dẫn cài đặt và sử dụng hệ thống
    
## Hướng dẫn cài đặt và sử dụng hệ thống trên local

### Cài đặt thủ công

1. **Clone repository**
```bash
git clone https://github.com/QuocThai1501/DoAn-ie207.git
```

2. **Khởi chạy development**
   - Terminal 1 (khởi chạy demo 1):
     ```bash
     cd Hotel_Recommendation_System
     streamlit run demo1.py
     ```

   - Terminal 2 (khởi chạy demo 2):
     ```bash
     cd Hotel_Recommendation_System
     streamlit run demo2.py
     ```
     
### Hướng dẫn sử dụng hệ thống

**Demo 1**: Nhập giá trị ID du khách (từ 1 đến 10000) vào ô "Enter User ID" ở sidebar rồi nhấn enter. Hệ thống sẽ trích xuất thông tin của du khách có ID vừa được nhập và hiển thị ở "User Profile" bên dưới. Ở phía giao diện chính, các tiêu chí được dùng để tìm khách sạn sẽ hiển thị ở "Your Search Criteria". Cuối cùng cần click vào nút "Search Hotels" để tìm và hiển thị khách sạn đáp ứng các tiêu chí.

**Demo 2**: Kích hoạt các tiêu chí lọc ở sidebar, gồm: địa điểm, ngân sách, thời gian nhận/trả phòng, và lựa chọn các tiện ích yêu thích ở "Facilities and Requirements". Ngoài ra có thể tự nhập vào các tiện ích khác ở "Additional Preferences". Sau đó click "Find Hotels" để hệ thống lưu thông tin, rồi click "Search Hotels" để hiển thị danh sách khách sạn phù hợp.