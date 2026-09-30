# Kết quả benchmark EarVN2.0 (tự sinh bởi scripts/16_make_tables.py)

Mỗi file CSV = một thí nghiệm, mỗi dòng = một mô hình (và một split / shot / ... nếu có). Mọi chỉ số tính bằng %. Kết quả thô từng lần chạy (seed / fold / episode) nằm trong `per_run/`.

| File | Nội dung | Số dòng |
|---|---|---|

## Ý nghĩa các cột

| Cột | Ý nghĩa |
|---|---|
| `top1 / acc` | Accuracy (Top-1): tỉ lệ ảnh có dự đoán hạng 1 đúng. Với identity, top1 chính là accuracy |
| `top5, top10` | Tỉ lệ ảnh có người đúng nằm trong 5 / 10 dự đoán cao nhất |
| `top1_subject_mean` | Top-1 tính riêng từng người rồi lấy trung bình (mỗi người nặng như nhau) |
| `macro_precision / macro_recall / macro_f1` | Precision, recall, F1 tính cho từng lớp rồi lấy trung bình không trọng số (chỉ số chính khi lớp không cân bằng). macro_recall = balanced accuracy |
| `weighted_precision / weighted_recall / weighted_f1` | Như trên nhưng trọng số theo số ảnh của lớp |
| `bal_acc` | Balanced accuracy (gender): trung bình recall của nam và nữ |
| `f1` | Gender: macro-F1 của 2 lớp |
| `precision_male / recall_male / f1_male, ..._female` | Chỉ số của từng giới; recall_female = độ nhạy (sensitivity) với lớp nữ, recall_male = độ đặc hiệu (specificity) |
| `auc` | Diện tích dưới đường ROC (gender) |
| `mcc` | Matthews correlation coefficient × 100 (gender), bền với lớp lệch |
| `acc_subject` | Gender theo người: đa số phiếu của các ảnh của một người |
| `rank1 / rank5 / rank10` | Truy hồi: người đúng nằm trong k người giống nhất (CMC tại k) |
| `aucmc` | Diện tích dưới đường CMC (chuẩn hoá về %) |
| `mrr` | Mean reciprocal rank × 100 |
| `eer` | Equal error rate (xác thực 1:1), càng thấp càng tốt |
| `tar_at_far` | True accept rate tại FAR = retrieval.far |
| `nmi / ari` | Normalized mutual information / adjusted Rand index (phân cụm) |
| `purity, homogeneity, completeness, v_measure` | Các chỉ số phân cụm khác |
| `test_* / val_*` | Chỉ số trên tập test / tập val (val chỉ dùng để chọn epoch) |
| `best_epoch, train_seconds` | Epoch được giữ lại (theo val) và thời gian huấn luyện |
| `*_mean / *_std / *_ci95` | Trung bình / độ lệch chuẩn / nửa khoảng tin cậy 95% qua n_runs lần chạy |
| `n_runs / n_diverged` | Số lần chạy được tính / số lần chạy phân kỳ (loss NaN) bị bỏ ra |
