# EarVN2.0 Benchmark

Mã nguồn chạy toàn bộ benchmark cho bài release **EarVN2.0**, theo đúng kế hoạch đã thống nhất với thầy
(E0–E8, E3b, kiểm soát chất lượng Q1–Q3). Mỗi thí nghiệm là một script đánh số; chạy theo thứ tự, kết quả
tự xuất ra thư mục `outputs/results/` (mỗi thí nghiệm một file CSV, đủ Top-1/Top-5, precision, recall, F1, …)
và tự điền vào bảng (`outputs/tables/all_tables.md`, kèm bản LaTeX).

- 30 mô hình thuộc 11 họ, tải qua `timm` với tag pretrained cố định (`configs/models.yaml`).
- Tiền xử lý **letterbox** (giữ tỉ lệ tai + đệm cho vuông), không kéo dãn.
- Split **S1** 45/15/còn lại (identity), **S2** 5-fold theo người (gender), **S3** người mới (E7/E8).
- Ảnh gần trùng được giữ cùng một phía khi chia (tránh rò rỉ train → test).
- Chạy lại an toàn: run nào xong rồi sẽ tự bỏ qua; tắt máy giữa chừng thì chạy lại lệnh cũ.

---

## ⚠️ Bắt buộc trước khi chạy thật

Bốn việc dưới đây **code không tự làm thay được**; làm sai thì kết quả vẫn ra số nhưng số sai.

1. **Nhãn giới tính.** Tạo `data/gender_labels.csv` (cột `subject,gender`, giá trị `M`/`F`, xem
   `data/gender_labels.example.csv`), **hoặc** đặt đúng số thư mục nam `gender.male_count` trong
   `configs/default.yaml`. Tham số này cố ý **không có giá trị mặc định**: script 01 sẽ dừng cho tới khi bạn
   khai báo. Sau khi chạy 01, xem các dòng log in ra quanh ranh giới nam/nữ và mở ~10 thư mục quanh ranh giới
   để xác nhận bằng mắt.
2. **Chốt ngưỡng ảnh gần trùng (DINOv2 và pHash).** Sau khi chạy script 02, mở
   `outputs/qc/threshold_review_dino.jpg` (và `threshold_review_phash.jpg`). Mỗi dòng là một cặp ảnh cùng người
   nằm ngay sát ngưỡng: **xanh** = đang bị tính là trùng, **đỏ** = không tính là trùng.
   - Dòng xanh phải là *cùng một bức ảnh* (cắt lại, resize, nén lại); nếu thấy hai bức ảnh *khác nhau* bị tính
     là trùng → ngưỡng đang lỏng: **tăng** `qc.dino_cosine_min` (hoặc giảm `qc.phash_hamming_max`).
   - Dòng đỏ phải là *hai bức ảnh khác nhau*; nếu thấy bản sao bị bỏ sót → **giảm** `qc.dino_cosine_min`
     (hoặc tăng `qc.phash_hamming_max`).
   - Có thể ghi kết luận vào cột `is_duplicate_by_eye` của file `.csv` cùng tên để lưu lại căn cứ.
   - Sửa ngưỡng xong thì chạy lại 02 (đặc trưng đã có cache, chỉ mất vài giây), rồi mới chạy 03–04.
   - Giá trị mặc định (cosine 0.95, Hamming 6) **chưa được kiểm chứng trên EarVN2.0**: toàn bộ test của project
     chạy với trọng số ngẫu nhiên vì môi trường phát triển không tải được trọng số pretrained, nên ngưỡng này
     chỉ là điểm xuất phát.
3. **Soát danh sách ảnh nghi vấn** `outputs/qc/outliers_for_review.csv` (ảnh lạ trong thư mục, ảnh trùng giữa
   hai người, ảnh quá nhỏ), ghi `y` vào cột `remove` cho ảnh cần loại, rồi chạy `03_qc_labels.py --apply`.
4. **Kích thước đầu vào không vuông cho E0.** Lấy giá trị script 01 gợi ý trong
   `outputs/metadata/dataset_stats.md` và đặt vào `preprocess.nonsquare_hw`.

Những gì **đã** được kiểm thử: toàn bộ logic (đọc ảnh, QC, chia dữ liệu, huấn luyện, few-shot, phân cụm,
đo lường, sinh bảng, chuyển GPU ↔ CPU) trên dữ liệu giả với 169 test. Những gì **chưa** kiểm được và cần để ý ở lần chạy thật đầu
tiên: ngưỡng QC trên ảnh thật (mục 2), LR tìm được có hội tụ tốt với trọng số pretrained thật không (xem
`outputs/runs/lr_sweep/*/log.csv`), và thời gian chạy thực tế so với ước tính ở mục 9, và cơ chế chuyển GPU ↔ CPU (mục 8) trên GPU thật —
cơ chế này được kiểm thử bằng lỗi hết bộ nhớ giả lập vì máy phát triển không có GPU.

## 1. Cấu trúc thư mục

```
earvn2-benchmark/
├── README.md                 <- file này
├── requirements.txt          <- thư viện cần cài
├── requirements-dev.txt      <- + pytest (để chạy test)
├── run_all.py                <- chạy cả giai đoạn bằng 1 lệnh (Windows/Linux/macOS)
├── configs/
│   ├── default.yaml          <- MỌI tham số (đường dẫn, split, tiền xử lý, huấn luyện, few-shot...)
│   └── models.yaml           <- danh sách 30 mô hình, họ, tag pretrained
├── data/
│   ├── README.md             <- cách đặt dữ liệu
│   ├── EarVN2.0/             <- ĐẶT DATASET VÀO ĐÂY (mỗi thư mục con = 1 subject)
│   └── gender_labels.example.csv
├── earbench/                 <- thư viện dùng chung (không cần sửa)
│   ├── config.py  paths.py  utils.py  common.py
│   ├── scan.py        (quét dataset, đọc ảnh an toàn, gán giới tính)
│   ├── qc.py          (pHash, ảnh gần trùng, ảnh lạ, trùng với EarVN1.0)
│   ├── splits.py      (S1/S2/S3, bốc episode few-shot)
│   ├── preprocess.py  (letterbox / stretch / non-square, augmentation)
│   ├── models.py  train.py  features.py  fewshot.py
│   ├── device.py      (ưu tiên GPU; hết bộ nhớ → micro-batch → CPU → tự quay lại GPU)
│   ├── metrics.py     (accuracy, balanced acc, Hungarian, Rank-k/CMC, EER, TAR@FAR)
│   └── tables.py      (sinh bảng Markdown + LaTeX)
├── scripts/                  <- 17 script, chạy theo thứ tự số
├── tools/make_synthetic_dataset.py   <- tạo dataset giả để chạy thử
├── tests/                    <- 169 test (đơn vị + chạy toàn bộ pipeline)
└── outputs/                  <- (tự sinh) mọi kết quả, không bao giờ ghi vào data/
```

## 2. Cài đặt

Yêu cầu: Python ≥ 3.9 (đã test 3.11), GPU NVIDIA khuyến nghị (RTX 3090/4090), ~30 GB ổ trống.

```bash
# 1) môi trường riêng
conda create -n earbench python=3.11 -y
conda activate earbench            # hoặc: python -m venv .venv && source .venv/bin/activate

# 2) PyTorch đúng phiên bản CUDA của máy (xem https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) các thư viện còn lại
pip install -r requirements.txt            # hoặc requirements-dev.txt nếu muốn chạy test

# 4) kiểm tra môi trường + tải trước trọng số 30 mô hình (khoảng 5 GB, cần internet)
python scripts/00_check_env.py --download
```

Trọng số pretrained được `timm` tải từ Hugging Face và lưu cache (`~/.cache/huggingface`). Máy không có
internet: tải trên máy khác, chép thư mục cache sang, rồi đặt `HF_HUB_OFFLINE=1`.

## 3. Đặt dữ liệu

```
data/EarVN2.0/001/*.jpg      (chưa chia)        hoặc
data/EarVN2.0/001/train|val|test/*.jpg          (đã chia bằng script cũ 45/15/còn lại)
```

Chi tiết và quy ước: `data/README.md`. **Nên** tạo `data/gender_labels.csv` (cột `subject,gender`, giá trị
`M`/`F`); nếu không có, giới tính lấy theo thứ tự thư mục với `gender.male_count` thư mục đầu là nam.
`gender.male_count` không có giá trị mặc định: phải khai báo, nếu không script 01 sẽ dừng.

Dataset ở chỗ khác thì không cần chép: thêm `--set paths.dataset_root=D:/EarVN2.0` vào mọi lệnh.

## 4. Thứ tự chạy

Cách nhanh nhất là chạy theo **giai đoạn** bằng `run_all.py`; mỗi giai đoạn in ra việc cần làm tiếp theo.

| Giai đoạn | Lệnh | Gồm các script | Việc bạn làm sau đó |
|---|---|---|---|
| 1. Chuẩn bị | `python run_all.py prepare` | 00, 01, 02, 03 | Làm đủ 4 việc ở mục "Bắt buộc trước khi chạy" (ngưỡng QC, ảnh nghi vấn, ranh giới nam/nữ, kích thước E0) |
| 2. Chia dữ liệu | `python run_all.py splits` | 03 `--apply`, 04 | Xem `outputs/splits/splits_summary.md` |
| 3. Chạy cơ bản | `python run_all.py basic` | 05, 06, 07, 08 (5 mô hình cơ bản) | **Gửi thầy** kết quả E0, E1, E2 |
| 4. Đặc trưng | `python run_all.py features` | 10, 11, 12 | E3, E3b, E4, E5/E6 cho 30 mô hình |
| 5. Đầy đủ | `python run_all.py full` | 08 (30 mô hình), 09, 10, 13, 14, 15 | Bảng cuối cùng |

Sau mỗi giai đoạn, `16_make_tables.py` tự chạy: kết quả cuối cùng nằm trong `outputs/results/*.csv`
(giải thích cột ở `outputs/results/README.md`), bảng để dán vào báo cáo ở `outputs/tables/all_tables.md`.

### Danh sách script

| # | Script | Làm gì | Kết quả chính | Mục trong kế hoạch |
|---|---|---|---|---|
| 00 | `00_check_env.py [--download]` | Kiểm tra thư viện, GPU, thư mục data, tải trọng số | log | – |
| 01 | `01_scan_dataset.py` | Quét subject/ảnh, đọc kích thước, phát hiện file hỏng, gán giới tính, thống kê (tỉ lệ cao/rộng → kích thước E0) | `metadata/images.csv`, `subjects.csv`, `dataset_stats.md`, `preprocess_preview.jpg` | Bảng thông tin dataset |
| 02 | `02_qc_duplicates.py [--no-dino]` | Q1: pHash + DINOv2 → nhóm ảnh gần trùng; đo rò rỉ của split cũ, quyết định giữ/chia lại; ảnh soát ngưỡng | `qc/near_duplicate_pairs.csv`, `dup_groups.csv`, `leakage_report.json`, `threshold_review_*.jpg` | Q1 |
| 03 | `03_qc_labels.py` rồi `--apply` | Q2: liệt kê ảnh lạ trong subject, ảnh trùng giữa 2 người, ảnh quá nhỏ để soát tay; `--apply` tạo danh sách loại | `qc/outliers_for_review.csv` → `qc/removed.csv` | Q2 |
| 04 | `04_make_splits.py` | S1 (giữ split cũ hoặc chia lại theo nhóm), S2 5-fold, S3 | `splits/*.csv`, `splits_summary.md` | Protocol chia dữ liệu |
| 05 | `05_model_info.py [--part info\|memory]` | Số tham số, GMACs, độ trễ; **đo bộ nhớ GPU của từng mô hình** (kế hoạch micro-batch, mục 8.1) | `results/per_run/model_info.csv`, `memory_plan.csv` | Cột Params/GMACs của E1, bảng HP |
| 06 | `06_lr_sweep.py` | Quét LR {3e-4, 1e-4, 3e-5} cho mô hình đại diện mỗi họ | `results/per_run/lr_sweep.csv`, `chosen_lr.json` | Bảng LR theo họ |
| 07 | `07_e0_preprocessing.py` | E0: kéo dãn vs letterbox vs không vuông | `results/per_run/E0.csv` | E0 |
| 08 | `08_train_supervised.py --exp E1\|E2` | Fine-tune identity (S1, 3 seed) / gender (S2 5 fold + S1) | `results/per_run/E1.csv`, `E2.csv`, `runs/.../predictions.csv` | E1, E2 |
| 09 | `09_train_s3.py` | Fine-tune trên người "train" của S3 (cho E7/E8) | `runs/S3/<model>/seed0/best.pt` | E7, E8 |
| 10 | `10_extract_features.py --what frozen\|s3ft\|external` | Trích đặc trưng (có cache) | `features/<loại>/<model>.npz` | E3–E8 |
| 11 | `11_fewshot.py` | E3 (K = 1/2/5, 5 val, còn lại test), E3b (1 ảnh mẫu), E4 (gender K = 1/2/5/10) | `results/per_run/E3.csv`, `E3b.csv`, `E4.csv` | E3, E3b, E4 |
| 12 | `12_unsupervised.py` | K-means k=2 (gender), k=số người (identity) | `results/per_run/E56.csv` | E5 + E6 |
| 13 | `13_new_identity.py` | E7: gallery 1/5 ảnh, Rank-k, EER, TAR@FAR=1% | `results/per_run/E7.csv`, `E7_verification.csv` | E7 |
| 14 | `14_cross_dataset.py` | Q3 trùng người với EarVN1.0, rồi E8 trên EarVN1.0/AWE | `qc/overlap_earvn1.csv`, `results/per_run/E8.csv` | Q3, E8 |
| 15 | `15_resolution_analysis.py` | Top-1 theo nhóm độ phân giải (từ dự đoán E1) | `results/per_run/resolution.csv` | Phân tích độ phân giải |
| 16 | `16_make_tables.py` | **Xuất kết quả cuối cùng** (1 CSV / thí nghiệm, đủ mọi chỉ số) + sinh toàn bộ bảng | `results/*.csv`, `results/README.md`, `tables/all_tables.md`, `tables/*.tex` | Tất cả bảng |

Chạy từng script riêng cũng được, ví dụ:

```bash
python scripts/08_train_supervised.py --exp E1 --models resnet50,vit_s16 --seeds 0
python scripts/08_train_supervised.py --exp E2 --models all --split S2 --folds 0,1
python scripts/11_fewshot.py --exp E3b --models dinov2_s
```

`--models` nhận `all`, `basic` (5 mô hình cơ bản), `lr_rep` (đại diện mỗi họ) hoặc danh sách key trong
`configs/models.yaml`. Mọi script có `--help`.

## 5. Tham số và cách ghi đè

Tất cả nằm trong `configs/default.yaml` (có chú thích từng dòng). Ghi đè tạm thời bằng `--set`:

```bash
python run_all.py basic --set train.batch_size=32 --set num_workers=8
python scripts/07_e0_preprocessing.py --set preprocess.nonsquare_hw=[224,112]
```

Các tham số hay cần chỉnh:

| Tham số | Mặc định | Khi nào chỉnh |
|---|---|---|
| `gender.male_count` | *(không có — bắt buộc)* | Số thư mục nam đầu tiên (nếu không có `gender_labels.csv`) |
| `preprocess.nonsquare_hw` | [224, 128] | Lấy từ `dataset_stats.md` (script 01 gợi ý) trước khi chạy E0 |
| `qc.phash_hamming_max`, `qc.dino_cosine_min` | 6, 0.95 | Sau khi xem `threshold_review_*.jpg` (mục "Bắt buộc trước khi chạy") |
| `train.batch_size` | 64 | Giảm nếu hết bộ nhớ GPU |
| `num_workers` | 4 | Đặt 0 nếu Windows báo lỗi DataLoader |
| `fewshot.episodes_linear` | 20 | Giảm nếu logistic regression của E3 quá chậm |

### 5.1 Công thức huấn luyện (fine-tune) — tất cả trong mục `train:` của `configs/default.yaml`

| Mục | Mặc định | Khoá trong config |
|---|---|---|
| Số epoch tối đa | 100 (dừng sớm nếu val không tăng sau 15 epoch; lịch cosine trải trên cả 100 epoch) | `train.epochs`, `train.patience` |
| Optimizer | **AdamW** β = (0.9, 0.999), ε = 1e-8, weight decay 0.05 (không decay bias, lớp norm và các tham số mô hình tự liệt kê như `pos_embed`, `cls_token` của ViT — giống timm) | `train.optimizer.*` |
| Đổi sang SGDM | `--set train.optimizer.name=sgd` (momentum 0.9, `nesterov` tuỳ chọn) — nhớ quét lại LR (SGD thường cần LR ~0.01) | `train.optimizer.momentum`, `.nesterov` |
| Lịch LR | warm-up tuyến tính 2 epoch rồi cosine về 0 (có `step` / `constant`) | `train.scheduler.*` |
| Learning rate | Quét {3e-4, 1e-4, 3e-5} trong 10 epoch trên mô hình nhỏ nhất mỗi họ (script 06), dùng cho cả họ | `train.learning_rate.sweep_grid`, `.sweep_epochs` |
| Batch size | 64 train / 128 eval | `train.batch_size`, `train.eval_batch_size` |
| Loss | cross-entropy, label smoothing 0; gender có trọng số lớp cân bằng | `train.label_smoothing`, `train.gender_balanced_loss` |
| Khác | grad clip 1.0, AMP (bf16 nếu GPU hỗ trợ, không thì fp16), 3 seed {0, 1, 2} | `train.grad_clip`, `train.amp*`, `train.seeds` |
| Tái lập tuyệt đối | `train.deterministic: true` (chậm hơn ~10–20%) | `train.deterministic` |

Thứ tự ưu tiên learning rate của một họ: `train.learning_rate.by_family` (cố định trong config)
→ kết quả quét `outputs/results/per_run/chosen_lr.json` → `train.learning_rate.default`.

**Khi release project**: sau khi chạy script 06, file `outputs/results/per_run/chosen_lr_for_config.yaml` chứa sẵn đoạn
`by_family` — dán vào `configs/default.yaml` để người khác dùng đúng LR của mình mà không phải quét lại
(script 06 tự bỏ qua các họ đã cố định). Mỗi run còn lưu `outputs/runs/.../hparams.json` (toàn bộ tham số thực
tế) và script 16 sinh bảng `tables/HP.md` "Siêu tham số huấn luyện" để đưa vào mục *Implementation details*.

## 6. Kết quả nằm ở đâu

```
outputs/
├── metadata/   images.csv, subjects.csv, scan_problems.csv, dataset_stats.md
├── qc/         near_duplicate_pairs.csv, dup_groups.csv, leakage_report.json,
│               outliers_for_review.csv, removed.csv, overlap_earvn1.csv
├── splits/     s1_identity.csv, s2_gender_folds.csv, s3_new_identity.csv, split_warnings.csv, splits_summary.md
├── runs/       <thí nghiệm>/<mô hình>/<run>/ hparams.json, log.csv, predictions.csv, metrics.json (+ best.pt cho S3)
├── features/   frozen/, s3ft/, frozen_earvn1/, ... (.npz, float16)
├── results/    <- KẾT QUẢ CUỐI CÙNG: 1 file CSV / thí nghiệm, mỗi dòng = 1 mô hình (+ split / shot / ...)
│   ├── E0_preprocessing.csv, LR_sweep.csv, E1_identity_supervised.csv, E2_gender_supervised.csv,
│   │   E3_identity_fewshot.csv, E3b_single_gallery.csv, E3b_cmc_curves.csv, E4_gender_fewshot.csv,
│   │   E5_E6_unsupervised.csv, E7_new_identity_retrieval.csv, E7_new_identity_verification.csv,
│   │   E8_cross_dataset.csv, resolution_analysis.csv, S3_training.csv, model_info.csv, gpu_memory_plan.csv
│   ├── README.md  <- giải thích từng cột
│   └── per_run/   kết quả thô: mỗi dòng = 1 seed / 1 fold / 1 episode (E1.csv, E2.csv, ..., chosen_lr.json)
├── tables/     all_tables.md + từng bảng .md/.tex  <- dán thẳng vào report
└── logs/       log + cấu hình đầy đủ của mỗi lần chạy script (để tái lập)
```

**Chạy lại an toàn.** Mỗi dòng kết quả mang một `fingerprint` = toàn bộ công thức huấn luyện + đúng danh sách
ảnh/nhãn train/val/test (với thí nghiệm few-shot/phân cụm/truy hồi: file đặc trưng + split + tham số). Chạy lại
một script sẽ **bỏ qua** các run có fingerprint trùng và **tự chạy lại** các run mà config, split hay đặc trưng đã
đổi — không bao giờ trộn kết quả cũ và mới trong một bảng. Muốn ép chạy lại: thêm `--overwrite`.

Trong bảng, `(n=2/3)` nghĩa là giá trị trung bình mới có 2 trên 3 seed/fold/episode (chưa chạy xong hoặc có run
phân kỳ); `–` là chưa có kết quả. Đường CMC của E3b nằm ở `results/E3b_cmc_curves.csv` (để vẽ hình).

**Các chỉ số trong `outputs/results/*.csv`** (đều tính bằng %, mỗi chỉ số có cột `_mean`, `_std`, thêm `_ci95`
với thí nghiệm nhiều episode ngẫu nhiên, và `n_runs` = số lần chạy được lấy trung bình):

| Thí nghiệm | Chỉ số |
|---|---|
| Identity có giám sát (E0, E1) và few-shot (E3) | Top-1 (= accuracy), Top-1 trung bình theo người, Top-5, Top-10, precision / recall / F1 (macro và weighted) |
| Gender có giám sát (E2) và few-shot (E4) | Accuracy, balanced accuracy, precision / recall / F1 của **từng giới** và trung bình (macro, weighted), ROC-AUC, MCC, accuracy theo người |
| Truy hồi (E3b, E7, E8) | Rank-1 / 5 / 10, AUCMC, MRR, precision / recall / F1 của quyết định rank-1 |
| Xác thực (E7) | EER, TAR @ FAR = 1% |
| Phân cụm (E5, E6) | ACC (Hungarian), NMI, ARI, purity, homogeneity, completeness, V-measure, precision / recall / F1 sau ghép cụm |

Với identity, Top-1 và accuracy là **cùng một số** (tỉ lệ ảnh đoán đúng người), nên chỉ có cột `top1`.

**Run phân kỳ** (loss thành NaN/inf, thường do LR quá lớn với mô hình lớn trong họ) không làm dừng cả hàng đợi:
run được ghi `status=diverged`, bảng hiện `–`, và script 16 liệt kê chúng. Cách sửa: cố định LR nhỏ hơn cho họ
đó trong `train.learning_rate.by_family` rồi **chạy lại đúng script đó, không cần `--overwrite`** — LR đổi nên chỉ
các run của họ đó được chạy lại. (`--overwrite` không kèm `--models` sẽ train lại *tất cả*, mất nhiều tuần.)

## 7. Những điểm thiết kế quan trọng

- **Letterbox**: resize cạnh dài về 224 (bicubic), đệm hai bên bằng pixel mép (như VGGFace-Ear), chuẩn hoá theo
  mean/std riêng của từng mô hình (CLIP, MobileViT khác ImageNet). Augmentation chỉ khi train và không làm méo tỉ lệ.
  E0 cho số liệu so sánh với kéo dãn và đầu vào không vuông. Khi xoay ảnh (augmentation), góc trống được tô
  màu trung bình của mô hình.
- **Ảnh gần trùng (Q1)**: hai ảnh gần trùng nếu pHash cách ≤ `phash_hamming_max` (chỉ với ảnh ≥ 24 px) *hoặc*
  DINOv2 cosine ≥ `dino_cosine_min` (cùng người). Nếu > 1% ảnh test của split cũ có bản sao trong train thì S1
  được chia lại, mỗi nhóm nằm trọn một phía, vẫn đúng 45/15/còn lại. **Trước khi tin ngưỡng DINOv2**, mở
  `near_duplicate_pairs.csv`, xem ~50 cặp có `cos` gần ngưỡng; nếu script báo nhóm quá lớn (> 20% ảnh một
  người) thì ngưỡng đang quá lỏng → siết lại hoặc dùng `qc.pair_rule=and`.
- **Split**: S2 xoay 5 lần (fold k test, fold k+1 val, còn lại train), mỗi người được test đúng 1 lần.
  S3 để riêng 30% người làm "người mới" cho E7/E8.
- **Learning rate theo họ**: CNN và Transformer cần LR khác nhau; script 06 chọn LR tốt nhất trên val cho mô
  hình nhỏ nhất mỗi họ rồi dùng cho cả họ, cả hai bài toán. Nếu chạy lại quét LR và LR thay đổi (hoặc đổi
  `preprocess.mode`), các run cũ tương ứng sẽ tự chạy lại, không giữ kết quả cũ.
- **Gender mất cân bằng** (~300 nam / ~200 nữ): loss có trọng số nghịch với tần suất lớp
  (`train.gender_balanced_loss`), chọn checkpoint theo balanced accuracy trên val.
- **Few-shot công bằng**: episode ngẫu nhiên chỉ phụ thuộc seed, mọi mô hình dùng *cùng một bộ* episode.
  Ảnh gần trùng với ảnh support bị loại khỏi test. Few-shot gender lấy K ảnh từ K người khác nhau.
- **Độ đo**: identity báo cả Top-1 theo ảnh và Top-1 trung bình theo người (vì "còn lại là test" nên số ảnh test
  mỗi người khác nhau); gender báo thêm Balanced accuracy, AUC và accuracy mức người (bỏ phiếu). Fine-tune:
  trung bình ± độ lệch chuẩn qua seed/fold; few-shot: trung bình ± khoảng tin cậy 95%.
- **GMACs** là số phép nhân-cộng cho 1 ảnh 224×224; nhiều bài gọi số này là "GFLOPs".

## 8. GPU hết bộ nhớ: tự chuyển sang CPU rồi quay lại GPU

Mọi bước huấn luyện, dự đoán và trích đặc trưng đều **ưu tiên GPU** và không crash khi GPU hết bộ nhớ
(ví dụ có tiến trình khác chiếm GPU). Cấu hình ở mục `gpu_fallback` trong `configs/default.yaml`:

1. Gặp lỗi *CUDA out of memory* → dọn bộ nhớ đệm và **chia batch thành micro-batch** (tích luỹ gradient,
   cho đúng cùng loss/gradient như batch đầy đủ), giảm dần tới `min_micro_batch` ảnh.
2. Vẫn thiếu bộ nhớ → theo `mode`:
   - `cpu` (mặc định): chuyển mô hình + optimizer sang **CPU** và chạy tiếp (chậm hơn nhiều, nhưng không dừng);
   - `wait`: tạm dừng rồi thử lại trên GPU; nếu GPU trông còn trống mà vẫn không vừa (chương trình khác giữ một
     phần bộ nhớ) thì khoảng chờ tăng dần (`check_every_s`, ×2, ×4, … tối đa 30 phút); chỉ dừng hẳn khi đã chờ
     quá `max_wait_minutes`;
   - `off`: báo lỗi và dừng như bình thường.
3. Khi đang ở CPU, cứ `check_every_s` giây kiểm tra GPU; khi còn trống ≥ `min_free_mb_to_return` MB thì
   **tự quay lại GPU**. Quay lại thất bại thì thời gian chờ lần sau tăng gấp đôi (tránh nhảy qua lại liên tục).

Mỗi lần chuyển đều ghi log. `log.csv` của từng run có cột `device` và `micro_batches` theo từng epoch;
`results/per_run/*.csv` có các cột `gpu_oom_events`, `gpu_switches_to_fallback`, `gpu_switches_back`,
`gpu_fallback_batches`, `gpu_step_oom` (bằng 0 = run chạy trọn trên GPU) và `gpu_max_micro_batches` (bằng
`planned_micro_batches` = đúng kế hoạch mục 8.1). Nếu một bước cập nhật optimizer bị gián đoạn vì hết bộ nhớ
(`gpu_step_oom` > 0), bước đó có thể chỉ được áp dụng một phần: run vẫn chạy tiếp nhưng bị đánh dấu.

Lưu ý khi viết bài: phần chạy trên CPU không dùng mixed precision, và micro-batch làm BatchNorm thấy batch nhỏ
hơn, nên kết quả có thể lệch rất nhẹ so với run chạy trọn trên GPU. Script 16 liệt kê các run đã rời GPU hoặc
phải chia batch nhiều hơn kế hoạch (mục 8.1) — chạy lại các run đó khi GPU rảnh (`--overwrite --models <key>`,
kèm `--seeds`/`--folds` nếu cần, để không train lại mọi thứ) để mọi số
liệu trong bảng cùng điều kiện. CPU chậm hơn
GPU hàng chục lần với ViT-B, CLIP, DINOv2-B; nếu máy chỉ dùng cho benchmark thì `mode: wait` thường hợp lý hơn.

### 8.1 Công bằng giữa các mô hình (đặc biệt với GPU 10–12 GB như RTX 3080)

Mục tiêu: 30 mô hình khác nhau **chỉ ở kiến trúc và trọng số pretrained**, mọi thứ khác giống hệt nhau.

| Yếu tố | Cách project giữ công bằng |
|---|---|
| Dữ liệu, split, episode few-shot | Cùng file split, cùng seed; episode chỉ phụ thuộc seed nên mọi mô hình gặp đúng cùng ảnh |
| Tiền xử lý | Cùng letterbox 224×224, cùng augmentation; chỉ mean/std theo đúng mô hình pretrained |
| Công thức huấn luyện | Cùng epoch, patience, optimizer, lịch LR, batch size, loss, 3 seed (mục 5.1) |
| Learning rate | Mỗi họ được chọn LR tốt nhất trên **cùng một lưới** {3e-4, 1e-4, 3e-5}, theo val (không nhìn test) |
| **Batch size hiệu dụng** | Luôn là 64. Mô hình lớn không vừa bộ nhớ thì dùng **tích luỹ gradient** (vd. 2 × 32) — gradient giống hệt batch 64. Số micro-batch của từng mô hình được **đo một lần** bằng script 05 trên chính GPU của bạn và **cố định từ bước đầu tiên** của mọi run, không phụ thuộc lúc chạy GPU có đang bị chiếm hay không |
| BatchNorm | Tích luỹ gradient chính xác với mô hình LayerNorm (ViT, Swin, ConvNeXt, DINOv2, CLIP) nhưng không hoàn toàn với BatchNorm (thống kê BN tính trên micro-batch). Script 05 và 16 **cảnh báo** nếu mô hình BN nào phải chia batch; bảng HP ghi lại để nêu trong bài |
| Độ chính xác số | Cùng một kiểu mixed precision cho mọi mô hình (RTX 3080 hỗ trợ **bf16** → `auto` chọn bf16), dùng cho cả train, val/test và trích đặc trưng |
| Tăng tốc | Chỉ dùng các cách **không đổi kết quả**: cuDNN autotune, `channels_last` cho các họ CNN (chỉ đổi bố cục bộ nhớ). **Không** dùng `torch.compile` (lợi cho kiến trúc này hơn kiến trúc khác, có thể lỗi với vài mô hình timm, phụ thuộc phiên bản) |
| Tốc độ/kích thước báo cáo | Params, GMACs, độ trễ bs=1 đo ở fp32, bố cục mặc định, **cùng điều kiện cho mọi mô hình**. Tốc độ train (ảnh/s) đo với đúng thiết lập train, ghi trong `memory_plan.csv` |
| GPU | Mọi run nên chạy trên cùng một GPU. `hparams.json` và `results/per_run/*.csv` ghi `gpu_name`, `amp_dtype_used`, `planned_micro_batches`; script 16 cảnh báo nếu kết quả đến từ nhiều GPU / nhiều độ chính xác khác nhau |

Việc cần làm trên RTX 3080:

1. Chạy `python scripts/05_model_info.py` **trên chính máy train**, tắt các chương trình khác dùng GPU (game,
   trình duyệt nặng) trong lúc đo. Log in ra mô hình nào cần tích luỹ gradient. Dự kiến trên 10 GB: các mô hình
   Base (ViT-B/16, DINOv2-B/14, CLIP-B/16) và có thể Swin-S / ConvNeXt-S cần 2 × 32 — đều là LayerNorm nên
   không ảnh hưởng kết quả. Nếu log báo một mô hình **BatchNorm** phải chia batch, cân nhắc giảm
   `train.batch_size` cho **tất cả** mô hình (vd. 48) rồi chạy lại script 05 với `--overwrite`.
2. Không train hai mô hình cùng lúc trên một GPU (bộ nhớ và thời gian đo sẽ không còn cùng điều kiện).
3. Đổi GPU, batch size, kích thước ảnh hay độ chính xác → kế hoạch cũ tự bị bỏ qua (có cảnh báo); chạy lại
   script 05 `--overwrite`.

## 9. Thời gian và dung lượng ước tính (1 × RTX 3090/4090)

| Việc | Thời gian | Dung lượng |
|---|---|---|
| 01–04 (quét, QC, split) | 15–40 phút | < 1 GB |
| 06 quét LR (33 lượt ngắn) | 4–8 giờ | – |
| 07 + 08 cơ bản (5 mô hình) | 1–2 ngày | – |
| 10 đặc trưng 30 mô hình | 1–2 giờ | ~3 GB |
| 11 few-shot + 12 phân cụm | 6–15 giờ (chủ yếu logistic regression) | – |
| 08 đầy đủ (~330 lượt) | 10–20 ngày | vài GB log |
| 09 S3 (30 mô hình) | 2–4 ngày | ~5 GB checkpoint |

Ước tính với `train.epochs: 100`, `patience: 15` (đa số lượt dừng sớm quanh epoch 30–60; nếu không dừng sớm thì
lâu hơn nữa). Nếu thiếu thời gian: chạy `08 --seeds 0` cho tất cả, rồi thêm seed 1, 2 cho các mô hình tốt nhất;
hoặc chạy nhanh bằng `--set train.epochs=30 --set train.patience=5` (nhớ ghi rõ trong bài nếu làm vậy).

## 10. Chạy thử trên dữ liệu giả (không cần GPU, không cần internet)

```bash
python tools/make_synthetic_dataset.py --out data/synthetic --subjects 12 --images 100 --edge-cases
python run_all.py prepare --no-download --set paths.dataset_root=data/synthetic --set paths.output_root=outputs_synth \
    --set gender.male_count=6 --set pretrained=false --set qc.pair_rule=phash --set splits.s2.n_folds=3
```

Dữ liệu giả cố tình chứa file hỏng, file rỗng, ảnh xám/RGBA/CMYK/16-bit, ảnh xoay EXIF, ảnh 3×5 px, ảnh
5×200 px, tên file tiếng Việt, ảnh trùng (cả giữa 2 người), subject quá ít ảnh, thư mục rỗng.

## 11. Kiểm thử

```bash
pip install -r requirements-dev.txt
pytest -m "not slow"      # ~1 phút: 167 test đơn vị (split, metric, tiền xử lý, QC, đọc ảnh, GPU→CPU, công bằng giữa mô hình...)
pytest -m slow            # ~5 phút CPU: chạy TOÀN BỘ 17 script trên 2 dataset giả (có/không split sẵn,
                          # có rò rỉ, có trùng với EarVN1.0), kiểm tra kết quả và khả năng chạy tiếp
```

## 12. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `CUDA out of memory` | Code đã tự xử lý (mục 8). Nếu log báo chuyển sang CPU quá thường xuyên: đóng tiến trình khác đang dùng GPU, hoặc `--set train.batch_size=32` |
| **Windows**: train chậm bất thường, không bao giờ thấy lỗi hết bộ nhớ | Driver NVIDIA ≥ 536.40 mặc định "tràn" sang RAM thay vì báo hết bộ nhớ GPU → chậm hàng chục lần và cơ chế mục 8 không kích hoạt. Mở *NVIDIA Control Panel → Manage 3D Settings → Program Settings*, chọn `python.exe` của môi trường, đặt **CUDA – Sysmem Fallback Policy = Prefer No Sysmem Fallback**, rồi chạy lại script 05 |
| Lỗi DataLoader trên Windows | `--set num_workers=0` |
| Không tải được trọng số | Kiểm tra internet / proxy; hoặc chép cache Hugging Face và đặt `HF_HUB_OFFLINE=1` |
| `non-finite loss` / run `diverged` | LR quá lớn với họ đó: cố định LR nhỏ hơn cho họ đó trong `train.learning_rate.by_family` rồi chạy lại script (không cần `--overwrite`) |
| `images are not in s1_identity.csv`, `no longer usable`, `near-duplicate groups ... differ` | QC đã đổi sau khi chia (loại thêm ảnh, chạy lại 02 với ngưỡng khác): chạy lại `04_make_splits.py` rồi các thí nghiệm (kết quả cũ tự được chạy lại nhờ fingerprint) |
| Nhiều ảnh `truncated` trong `scan_problems.csv` | File JPEG bị cắt cụt (thiếu byte, phần dưới ảnh xám). Mặc định bị loại; nếu thật sự nhiều và nhìn vẫn dùng được, đặt `dataset.keep_truncated: true` và chạy lại từ script 01 |
| Rò rỉ sau khi chia vẫn > 1% | Ngưỡng ảnh gần trùng quá lỏng (xem mục 7) rồi chạy lại 02–04 |
| Muốn làm lại một thí nghiệm | Thêm `--overwrite` **kèm `--models <key>`** (không kèm thì chạy lại tất cả mô hình), hoặc xoá dòng tương ứng trong `outputs/results/per_run/<exp>.csv` |
