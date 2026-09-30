# Thư mục dữ liệu

Đặt dữ liệu vào đây (hoặc trỏ tới chỗ khác bằng `--set paths.dataset_root=...`).

```
data/
├── EarVN2.0/                  <- BẮT BUỘC: chép nguyên bộ dữ liệu vào đây, giữ nguyên cấu trúc
│   ├── Description.txt
│   └── Images/                <- code tự nhận ra lớp này (không cần sửa đường dẫn)
│       ├── 001.ALI_HD/
│       │   ├── 001 (1).jpg
│       │   ├── 001 (2).jpg
│       │   └── ...            (ảnh để trực tiếp, CHƯA chia train/val/test: script 04 tự chia 45/15/còn lại)
│       ├── 002.LeDuong_BL/
│       └── ...
├── gender_labels.csv          <- tuỳ chọn: nhãn giới tính từng người (xem gender_labels.example.csv)
├── EarVN1.0/                  <- tuỳ chọn, cho Q3 (kiểm tra trùng người) và E8
│   ├── 1/ 2/ ...              (mỗi thư mục = 1 người)
└── AWE/                       <- tuỳ chọn, cho E8 (mỗi thư mục = 1 người)
```

Cũng chấp nhận: các thư mục người nằm ngay trong `EarVN2.0/` (không có lớp `Images/`), hoặc mỗi thư mục
người có sẵn `train/ val/ test/` (tên không phân biệt hoa thường, chấp nhận valid/validation, training, testing).
Nếu `EarVN2.0/` chỉ chứa đúng một thư mục con (như `Images/`) và trong đó có nhiều thư mục, code tự dùng thư mục
con đó; script 01 in ra đường dẫn thực sự được dùng.

**Mã người (subject)** = đúng tên thư mục, ví dụ `001.ALI_HD`. Thứ tự người theo **số đứng đầu** tên thư mục
(001, 002, …, 010, …, 100), nên `gender.male_count: 300` nghĩa là các thư mục `001.…` đến `300.…` là nam. Nếu
dùng `gender_labels.csv` thì cột `subject` phải ghi đúng tên thư mục đầy đủ.

## Quy ước

- **Tên thư mục subject** được sắp theo thứ tự tự nhiên (1, 2, …, 10), không phải thứ tự chữ (1, 10, 2).
- **Giới tính**: nếu có `data/gender_labels.csv` (cột `subject,gender`, giá trị `M`/`F`) thì dùng file này.
  Nếu không, lấy theo thứ tự thư mục: `gender.male_count` thư mục đầu là nam (giống EarVN1.0).
  Script 01 in ra các thư mục quanh ranh giới nam/nữ để bạn kiểm tra.
- **Định dạng ảnh**: jpg, jpeg, png, bmp, webp, tif (không phân biệt hoa thường). Ảnh xám, RGBA, CMYK,
  16-bit, ảnh có EXIF xoay đều được đọc đúng. Ảnh hỏng/rỗng và ảnh JPEG bị cắt cụt (thiếu byte) được ghi vào
  `outputs/metadata/scan_problems.csv` và tự động loại (giữ ảnh cắt cụt: `dataset.keep_truncated: true`).
- **Ảnh trùng từng byte** (cùng md5) luôn được coi là ảnh gần trùng, kể cả ảnh rất nhỏ, nên không bao giờ
  nằm ở hai phía train/test; trùng giữa hai người khác nhau được liệt kê để soát nhãn (script 03).
- File ẩn (`.DS_Store`, `Thumbs.db`, `__MACOSX`) bị bỏ qua. File không phải ảnh được liệt kê trong
  `scan_problems.csv`.
- Dữ liệu gốc **không bao giờ bị sửa**: mọi kết quả nằm trong `outputs/`.
