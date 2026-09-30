# Thư mục dữ liệu

Đặt dữ liệu vào đây (hoặc trỏ tới chỗ khác bằng `--set paths.dataset_root=...`).

```
data/
├── EarVN2.0/                  <- BẮT BUỘC: mỗi thư mục con là 1 subject
│   ├── 001/
│   │   ├── train/  val/  test/   (nếu đã có split của script cũ; tên không phân biệt hoa thường,
│   │   │                          chấp nhận cả valid/validation, training, testing)
│   │   └── ... hoặc để ảnh trực tiếp trong 001/ (chưa chia)
│   ├── 002/
│   └── ...
├── gender_labels.csv          <- NÊN CÓ: nhãn giới tính tường minh (xem gender_labels.example.csv)
├── EarVN1.0/                  <- tuỳ chọn, cho Q3 (kiểm tra trùng người) và E8
│   ├── 1/ 2/ ...              (mỗi thư mục = 1 người)
└── AWE/                       <- tuỳ chọn, cho E8 (mỗi thư mục = 1 người)
```

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
