
# PTIT - XỬ LÝ ẢNH SỐ | NHÓM 01

## 1. Giới thiệu dự án

**Tên đề tài:** Xây dựng bộ công cụ nâng cao chất lượng ảnh

**Môn học:** Xử lý ảnh số

**Bài tập:** Buổi 1 - Phần B (Bài tập nhóm)

**Nhóm thực hiện:** Nhóm 01

**Repository:** https://github.com/NguyenMinhTuanTesting/PTIT-XuLyAnh-Nhom01

### Mục tiêu

Dự án xây dựng và đánh giá các phương pháp nâng cao chất lượng ảnh bằng Python.

Các nội dung chính:

- Tự cài đặt thuật toán xử lý ảnh bằng NumPy.
- Kiểm chứng kết quả với OpenCV.
- Thực nghiệm ảnh thiếu sáng, ảnh nhiễu và ảnh mờ.
- Đánh giá bằng PSNR và SSIM.
- Xây dựng quy trình nâng cao chất lượng ảnh.
- Phân tích các trường hợp xử lý ảnh không hiệu quả.
- Thảo luận về nguồn dữ liệu và đạo đức xử lý ảnh.

---

## 2. Công nghệ sử dụng

| Công nghệ | Mục đích |
|---|---|
| Python 3.10+ | Ngôn ngữ lập trình |
| NumPy | Xử lý ma trận và cài đặt thuật toán |
| OpenCV | Xử lý ảnh và kiểm chứng |
| scikit-image | Ảnh mẫu, PSNR, SSIM |
| Matplotlib | Hiển thị ảnh và biểu đồ |
| SciPy | Hỗ trợ tính toán |
| Jupyter Notebook | Chạy và trình bày thực nghiệm |
| PyCharm | Môi trường phát triển |

---

## 3. Cấu trúc thư mục

```text
PTIT-XuLyAnh-Nhom01/
│
├── main.py
│   └── File chạy chính của chương trình
│
├── pipeline.py
│   └── Chứa thuật toán và các bước thực nghiệm B1-B5
│
├── config.py
│   └── Cấu hình nhóm và danh sách thành viên
│
├── requirements.txt
│   └── Danh sách thư viện Python
│
├── B1_Nhom01.ipynb
│   └── Notebook thực nghiệm
│
├── README.md
│   └── Hướng dẫn sử dụng dự án
│
├── data/
│   └── Ảnh đầu vào và ảnh suy giảm
│
├── results/
│   └── Ảnh kết quả, bảng số liệu và biểu đồ
│
└── BaoCao_B1_Nhom01_TiengViet.pdf
    └── Báo cáo kết quả thực nghiệm
```

**Lưu ý:** Một số file có thể vẫn mang tên `NhomXX` nếu chưa được đổi tên trong project.

---

## 4. Hướng dẫn cài đặt và chạy chương trình

### Bước 1: Cài đặt phần mềm

Yêu cầu:

- Python 3.10 trở lên.
- PyCharm hoặc VS Code.
- Git để tải project.

Kiểm tra Python:

```bash
python --version
```

### Bước 2: Tải source code

Mở Terminal hoặc CMD:

```bash
git clone https://github.com/NguyenMinhTuanTesting/PTIT-XuLyAnh-Nhom01.git
```

Di chuyển vào thư mục project:

```bash
cd PTIT-XuLyAnh-Nhom01
```

Hoặc tải ZIP trực tiếp từ GitHub rồi giải nén.

### Bước 3: Mở project bằng PyCharm

1. Mở PyCharm.
2. Chọn File > Open.
3. Chọn thư mục project.
4. Thiết lập Python Interpreter.
5. Mở Terminal trong PyCharm.

### Bước 4: Tạo môi trường Python (khuyến nghị)

```bash
python -m venv .venv
```

Trên Windows, kích hoạt bằng:

```bash
.venv\Scripts\activate
```

Nếu PyCharm đã tạo virtual environment thì có thể bỏ qua bước này.

### Bước 5: Cài đặt thư viện

Chạy:

```bash
python -m pip install -r requirements.txt
```

### Bước 6: Chạy toàn bộ chương trình

```bash
python main.py
```

Hoặc mở `main.py` trong PyCharm và nhấn Run.

Chương trình sẽ tự động:

1. Chuẩn bị dữ liệu ảnh.
2. Chạy và kiểm chứng thuật toán B1.
3. Thực nghiệm PSNR và SSIM cho B2.
4. Xử lý 6 ảnh bằng pipeline B3.
5. Minh họa các trường hợp thất bại B4.
6. Lưu bảng số liệu và hình ảnh vào thư mục `results/`.

**Lưu ý:** Phiên bản `pipeline.py` đã chỉnh sửa có thể không tự tạo báo cáo PDF. Sử dụng báo cáo tiếng Việt được cung cấp riêng và kiểm tra số liệu trước khi nộp.

---

## 5. Nội dung thực hiện

### B1 - Tự cài đặt thuật toán

Các hàm được xây dựng:

```python
hist_equalize(img)
conv2d(img, kernel, padding='reflect')
median_filter(img, k)
laplacian_sharpen(img, c)
```

Yêu cầu:

- Tự cài đặt bằng NumPy.
- So sánh kết quả với OpenCV.
- Tính sai khác tuyệt đối trung bình (MAE).
- So sánh thời gian thực thi.

**Kết quả:**

`results/B1_comparison.csv`

### B2 - Thực nghiệm tham số

Các nội dung khảo sát:

- Gamma trong khoảng 0.3 đến 1.0.
- Lọc trung bình với kernel 3, 5, 7, 9.
- Lọc Gaussian với kernel 3, 5, 7, 9.
- Lọc trung vị với kernel 3, 5, 7, 9.
- Unsharp masking với các hệ số k.

Các chỉ số đánh giá:

**PSNR:** Đánh giá sai khác giữa ảnh xử lý và ảnh tham chiếu. PSNR càng cao thường cho thấy sai số càng thấp.

**SSIM:** Đánh giá độ tương đồng cấu trúc giữa hai ảnh. Giá trị càng gần 1 càng tốt.

**Kết quả:**

- `results/B2_metrics.csv`
- `results/B2_metrics_plot.png`
- `results/B2_gaussian_noise_plot.png`

### B3 - Quy trình nâng cao chất lượng ảnh

Pipeline kết hợp:

```text
Ảnh đầu vào
      |
      v
Khử nhiễu
      |
      v
Tăng tương phản
      |
      v
Làm sắc nét
      |
      v
Ảnh kết quả
```

Chương trình áp dụng pipeline lên 6 ảnh và so sánh ảnh trước/sau.

**Kết quả:**

- `results/B3_six_before_after.png`
- `results/B3_metrics.csv`

### B4 - Phân tích trường hợp thất bại

Phân tích các hiện tượng:

- Khuếch đại nhiễu khi xử lý ảnh.
- Xuất hiện quầng sáng hoặc viền giả khi làm sắc nét.

Đưa ra nguyên nhân và hướng khắc phục.

**Kết quả:**

`results/B4_failure_cases.png`

### B5 - Dữ liệu và đạo đức

Nội dung nghiên cứu:

- Nguồn ảnh và quyền sử dụng.
- Trách nhiệm khi chỉnh sửa ảnh.
- Ranh giới giữa cải thiện chất lượng và làm sai lệch thông tin.
- Rủi ro khi chỉnh sửa ảnh y tế hoặc ảnh làm bằng chứng.

---

## 6. Bộ dữ liệu ảnh

Dự án sử dụng ảnh mẫu từ thư viện `scikit-image`.

Bộ dữ liệu gồm 6 ảnh nguồn khác nhau, được mô phỏng suy giảm thành:

| Loại ảnh | Số lượng |
|---|---:|
| Ảnh thiếu sáng | 2 |
| Ảnh nhiễu | 2 |
| Ảnh mờ | 2 |
| Tổng | 6 |

Ngoài ra, chương trình sử dụng một ảnh tham chiếu để tạo các phiên bản suy giảm phục vụ đánh giá định lượng.

**Lưu ý:** Đây là dữ liệu suy giảm mô phỏng bằng thuật toán, không phải toàn bộ ảnh bị suy giảm tự nhiên.

Cần kiểm tra nguồn gốc và điều kiện sử dụng riêng của từng ảnh mẫu trước khi phân phối dữ liệu.

---

## 7. Chạy Jupyter Notebook

Ngoài file Python, bài tập yêu cầu nộp notebook `.ipynb`.

Cài đặt Jupyter nếu chưa có:

```bash
python -m pip install notebook
```

Khởi chạy:

```bash
python -m notebook
```

Mở:

```text
B1_Nhom01.ipynb
```

Chọn:

**Kernel > Restart & Run All**

Đảm bảo notebook chạy thành công từ đầu đến cuối.

Notebook sử dụng các hàm trong `pipeline.py`, vì vậy phải giữ file này cùng project.

---

## 8. Thành viên nhóm

| STT | Họ và tên | MSSV | Đóng góp |
|---|---|---|---|
| 1 | Nguyễn Xuân Hải Anh | K24DTCN283 | 9% |
| 2 | Nguyễn Đức Công | K24DTCN099 | 10% |
| 3 | Lê Quang Duy | K24DTCN279 | 8% |
| 4 | Nguyễn Khánh Duy | K24DTCN289 | 9% |
| 5 | Ngô Hoài Hận | K23DTCN603 | 10% |
| 6 | Trần Kim Nga | K23DTCN608 | 8% |
| 7 | Phạm Ngọc Nghĩa | K24DTCN104 | 9% |
| 8 | Nguyễn Xuân Thanh | K24DTCN284 | 8% |
| 9 | Phạm Duy Thuận | K23DTCN614 | 9% |
| 10 | Ngô Hữu Trí | K24DTCN280 | 8% |
| 11 | Nguyễn Minh Tuấn | K24DTCN282 | 12% |
| | **Tổng cộng** | | **100%** |

Tỷ lệ đóng góp cần được các thành viên xác nhận theo công việc thực tế.

---

## 9. Hướng dẫn cập nhật code lên GitHub

Kiểm tra thay đổi:

```bash
git status
```

Thêm các file đã sửa:

```bash
git add .
```

Tạo commit:

```bash
git commit -m "Update image processing project"
```

Đẩy lên GitHub:

```bash
git push origin main
```

Nếu cần lấy code mới nhất từ GitHub:

```bash
git pull origin main
```

**Khuyến nghị:** Luôn pull trước khi chỉnh sửa nếu có nhiều thành viên cùng làm việc trên repository.

---

## 10. Sản phẩm nộp bài

Theo yêu cầu môn học, sản phẩm cuối cùng gồm:

1. Notebook `.ipynb` chạy được từ đầu đến cuối.
2. Báo cáo PDF dài 4-6 trang.
3. Bộ dữ liệu ảnh hoặc đường dẫn đến dữ liệu.
4. Các file Python phụ thuộc để notebook chạy được.

Đóng gói thành:

```text
B1_Nhom01.zip
```

Trước khi nộp cần kiểm tra:

- Code chạy không lỗi.
- Notebook chạy thành công bằng Restart & Run All.
- Có đầy đủ kết quả B1-B5.
- Báo cáo có số liệu và hình minh họa phù hợp.
- Có nguồn và quyền sử dụng ảnh.
- Có bảng phân công thành viên.
- Có khai báo phạm vi sử dụng công cụ AI.

---

## 11. Lưu ý

- Không sử dụng ảnh có khuôn mặt người khác khi chưa được đồng ý.
- Không sử dụng kết quả chỉnh sửa ảnh để làm sai lệch sự thật.
- Không sửa số liệu thực nghiệm để làm đẹp kết quả.
- Các nhận xét trong báo cáo phải phù hợp với kết quả thực tế.
- Nhóm cần xác nhận số lượng 11 thành viên với giảng viên vì đề bài quy định nhóm 3-5 sinh viên.

---

## 12. Liên kết dự án

**GitHub:** https://github.com/NguyenMinhTuanTesting/PTIT-XuLyAnh-Nhom01

**Môn học:** Xử lý ảnh số - PTIT

**Đề tài:** Xây dựng bộ công cụ nâng cao chất lượng ảnh

**Nhóm:** 01
