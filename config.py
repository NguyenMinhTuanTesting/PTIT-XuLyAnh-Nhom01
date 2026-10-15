
GROUP_NUMBER = '01'

MEMBERS = [
    ('Nguyễn Xuân Hải Anh', 'K24DTCN283', 'B1: cài đặt và kiểm chứng cân bằng histogram', 9),
    ('Nguyễn Đức Công', 'K24DTCN099', 'B2: khảo sát gamma, tổng hợp PSNR', 10),
    ('Lê Quang Duy', 'K24DTCN279', 'B3: xử lý ảnh thiếu sáng, đối chiếu ảnh', 8),
    ('Nguyễn Khánh Duy', 'K24DTCN289', 'B4: phân tích khuếch đại nhiễu', 9),
    ('Ngô Hoài Hận', 'K23DTCN603', 'B1: convolution, median và Laplacian', 10),
    ('Trần Kim Nga', 'K23DTCN608', 'B2: khảo sát bộ lọc và SSIM', 8),
    ('Phạm Ngọc Nghĩa', 'K24DTCN104', 'B3: xây dựng pipeline kết hợp', 9),
    ('Nguyễn Xuân Thanh', 'K24DTCN284', 'B5: nguồn ảnh, giấy phép và đạo đức', 8),
    ('Phạm Duy Thuận', 'K23DTCN614', 'B2: khảo sát unsharp và biểu đồ', 9),
    ('Ngô Hữu Trí', 'K24DTCN280', 'B3: đánh giá 6 ảnh trước và sau', 8),
    ('Nguyễn Minh Tuấn', 'K24DTCN282', 'Tổng hợp báo cáo, kiểm thử notebook và phân tích B4-B5', 12),
]

assert sum(member[3] for member in MEMBERS) == 100
