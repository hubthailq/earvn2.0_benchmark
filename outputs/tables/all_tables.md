# Kết quả benchmark EarVN2.0 (tự sinh bởi scripts/16_make_tables.py)

## Siêu tham số huấn luyện (tự sinh từ configs/default.yaml)

| Tham số | Giá trị |
| --- | --- |
| Khởi tạo | trọng số pretrained timm (tag trong configs/models.yaml) |
| Tiền xử lý | letterbox, 224×224, đệm 'edge', nội suy bicubic, chuẩn hoá mean/std riêng từng mô hình |
| Augmentation (train) | crop giữ tỉ lệ [0.8, 1.0], xoay ±15°, dịch ±0.1, color jitter [0.3, 0.3, 0.3], lật ngang = True |
| Optimizer | AdamW (β = (0.9, 0.999), ε = 1e-08), weight decay 0.05 (không áp dụng cho bias/norm) |
| Lịch learning rate | cosine, warm-up tuyến tính 2 epoch, LR cuối = 0 × LR |
| Số epoch tối đa / early stopping | 100 / patience 15 (theo Top-1 hoặc balanced acc trên val) |
| Batch size (train / eval) | 64 / 128 |
| Loss | cross-entropy, label smoothing 0, trọng số lớp nghịch tần suất (gender) |
| Gradient clipping | 1 |
| Mixed precision | có (auto; train, val/test và trích đặc trưng dùng cùng một độ chính xác cho mọi mô hình) |
| channels_last (chỉ đổi bố cục bộ nhớ) | VGG, ResNet, DenseNet, RegNet, MobileNet, EfficientNet, ConvNeXt |
| Seed | [0, 1, 2] (seed chia dữ liệu và episode: 2026) |
| Quét learning rate | [0.0003, 0.0001, 3e-05], 10 epoch, mô hình nhỏ nhất mỗi họ, chọn theo Top-1 val S1 |
| Few-shot | 100 episode (prototype), 20 (logistic regression, C ∈ [0.01, 0.1, 1.0, 10.0]) |
| Learning rate – VGG | 0.0001 (mặc định, chưa quét) |
| Learning rate – ResNet | 0.0001 (mặc định, chưa quét) |
| Learning rate – DenseNet | 0.0001 (mặc định, chưa quét) |
| Learning rate – RegNet | 0.0001 (mặc định, chưa quét) |
| Learning rate – MobileNet | 0.0001 (mặc định, chưa quét) |
| Learning rate – EfficientNet | 0.0001 (mặc định, chưa quét) |
| Learning rate – ConvNeXt | 0.0001 (mặc định, chưa quét) |
| Learning rate – ViT / DeiT | 0.0001 (mặc định, chưa quét) |
| Learning rate – Swin | 0.0001 (mặc định, chưa quét) |
| Learning rate – MobileViT | 0.0001 (mặc định, chưa quét) |
| Learning rate – Pretrain nền tảng | 0.0001 (mặc định, chưa quét) |

## E0 – So sánh cách tiền xử lý (Top-1 identity, %)

| Mô hình | Kéo dãn | Letterbox | Không vuông |
| --- | --- | --- | --- |
| ResNet-18 | – | – | – |
| ResNet-50 | – | – | – |
| MobileNetV2 | – | – | – |
| EfficientNet-B0 | – | – | – |
| ViT-S/16 | – | – | (không áp dụng) |

## Quét learning rate theo họ

| Họ | LR đã chọn | Val Top-1 với 0.0003 / 0.0001 / 3e-05 |
| --- | --- | --- |
| VGG | – | – / – / – |
| ResNet | – | – / – / – |
| DenseNet | – | – / – / – |
| RegNet | – | – / – / – |
| MobileNet | – | – / – / – |
| EfficientNet | – | – / – / – |
| ConvNeXt | – | – / – / – |
| ViT / DeiT | – | – / – / – |
| Swin | – | – / – / – |
| MobileViT | – | – / – / – |
| Pretrain nền tảng | – | – / – / – |

## E1 – Identity, có giám sát (%)

| Họ | Mô hình | Params (M) | GMACs | Top-1 theo ảnh | Top-1 TB theo người | Top-5 | Macro-P | Macro-R | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VGG | VGG-11 | – | – | – | – | – | – | – | – |
| VGG | VGG-16 | – | – | – | – | – | – | – | – |
| ResNet | ResNet-18 | – | – | – | – | – | – | – | – |
| ResNet | ResNet-50 | – | – | – | – | – | – | – | – |
| ResNet | ResNet-101 | – | – | – | – | – | – | – | – |
| DenseNet | DenseNet-121 | – | – | – | – | – | – | – | – |
| DenseNet | DenseNet-169 | – | – | – | – | – | – | – | – |
| RegNet | RegNetY-800MF | – | – | – | – | – | – | – | – |
| RegNet | RegNetY-1.6GF | – | – | – | – | – | – | – | – |
| MobileNet | MobileNetV2 | – | – | – | – | – | – | – | – |
| MobileNet | MobileNetV3-S | – | – | – | – | – | – | – | – |
| MobileNet | MobileNetV3-L | – | – | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B0 | – | – | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B2 | – | – | – | – | – | – | – | – |
| EfficientNet | EfficientNetV2-S | – | – | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-N | – | – | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-T | – | – | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-S | – | – | – | – | – | – | – | – |
| ViT / DeiT | ViT-S/16 | – | – | – | – | – | – | – | – |
| ViT / DeiT | ViT-B/16 | – | – | – | – | – | – | – | – |
| ViT / DeiT | DeiT-S | – | – | – | – | – | – | – | – |
| Swin | Swin-T | – | – | – | – | – | – | – | – |
| Swin | Swin-S | – | – | – | – | – | – | – | – |
| MobileViT | MobileViT-XS | – | – | – | – | – | – | – | – |
| MobileViT | MobileViT-S | – | – | – | – | – | – | – | – |
| MobileViT | MobileViTv2-1.0 | – | – | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-S/14 | – | – | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-B/14 | – | – | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/16 | – | – | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/32 | – | – | – | – | – | – | – | – |

## E2 – Gender, có giám sát (%)

| Họ | Mô hình | Acc (S2) | Bal. Acc (S2) | F1 (S2) | AUC (S2) | Acc mức người (S2) | Acc (S1) | Chênh S1 − S2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VGG | VGG-11 | – | – | – | – | – | – | – |
| VGG | VGG-16 | – | – | – | – | – | – | – |
| ResNet | ResNet-18 | – | – | – | – | – | – | – |
| ResNet | ResNet-50 | – | – | – | – | – | – | – |
| ResNet | ResNet-101 | – | – | – | – | – | – | – |
| DenseNet | DenseNet-121 | – | – | – | – | – | – | – |
| DenseNet | DenseNet-169 | – | – | – | – | – | – | – |
| RegNet | RegNetY-800MF | – | – | – | – | – | – | – |
| RegNet | RegNetY-1.6GF | – | – | – | – | – | – | – |
| MobileNet | MobileNetV2 | – | – | – | – | – | – | – |
| MobileNet | MobileNetV3-S | – | – | – | – | – | – | – |
| MobileNet | MobileNetV3-L | – | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B0 | – | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B2 | – | – | – | – | – | – | – |
| EfficientNet | EfficientNetV2-S | – | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-N | – | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-T | – | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-S | – | – | – | – | – | – | – |
| ViT / DeiT | ViT-S/16 | – | – | – | – | – | – | – |
| ViT / DeiT | ViT-B/16 | – | – | – | – | – | – | – |
| ViT / DeiT | DeiT-S | – | – | – | – | – | – | – |
| Swin | Swin-T | – | – | – | – | – | – | – |
| Swin | Swin-S | – | – | – | – | – | – | – |
| MobileViT | MobileViT-XS | – | – | – | – | – | – | – |
| MobileViT | MobileViT-S | – | – | – | – | – | – | – |
| MobileViT | MobileViTv2-1.0 | – | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-S/14 | – | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-B/14 | – | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/16 | – | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/32 | – | – | – | – | – | – | – |

## E3 – Identity, few-shot (Top-1 %, trung bình ± CI95)

| Họ | Mô hình | Proto 1-shot | Proto 2-shot | Proto 5-shot | LP 1-shot | LP 2-shot | LP 5-shot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VGG | VGG-11 | – | – | – | – | – | – |
| VGG | VGG-16 | – | – | – | – | – | – |
| ResNet | ResNet-18 | – | – | – | – | – | – |
| ResNet | ResNet-50 | – | – | – | – | – | – |
| ResNet | ResNet-101 | – | – | – | – | – | – |
| DenseNet | DenseNet-121 | – | – | – | – | – | – |
| DenseNet | DenseNet-169 | – | – | – | – | – | – |
| RegNet | RegNetY-800MF | – | – | – | – | – | – |
| RegNet | RegNetY-1.6GF | – | – | – | – | – | – |
| MobileNet | MobileNetV2 | – | – | – | – | – | – |
| MobileNet | MobileNetV3-S | – | – | – | – | – | – |
| MobileNet | MobileNetV3-L | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B0 | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B2 | – | – | – | – | – | – |
| EfficientNet | EfficientNetV2-S | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-N | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-T | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-S | – | – | – | – | – | – |
| ViT / DeiT | ViT-S/16 | – | – | – | – | – | – |
| ViT / DeiT | ViT-B/16 | – | – | – | – | – | – |
| ViT / DeiT | DeiT-S | – | – | – | – | – | – |
| Swin | Swin-T | – | – | – | – | – | – |
| Swin | Swin-S | – | – | – | – | – | – |
| MobileViT | MobileViT-XS | – | – | – | – | – | – |
| MobileViT | MobileViT-S | – | – | – | – | – | – |
| MobileViT | MobileViTv2-1.0 | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-S/14 | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-B/14 | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/16 | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/32 | – | – | – | – | – | – |

## E3 – Identity, few-shot (Top-1 trung bình theo người %, trung bình ± CI95)

| Họ | Mô hình | Proto 1-shot | Proto 2-shot | Proto 5-shot | LP 1-shot | LP 2-shot | LP 5-shot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VGG | VGG-11 | – | – | – | – | – | – |
| VGG | VGG-16 | – | – | – | – | – | – |
| ResNet | ResNet-18 | – | – | – | – | – | – |
| ResNet | ResNet-50 | – | – | – | – | – | – |
| ResNet | ResNet-101 | – | – | – | – | – | – |
| DenseNet | DenseNet-121 | – | – | – | – | – | – |
| DenseNet | DenseNet-169 | – | – | – | – | – | – |
| RegNet | RegNetY-800MF | – | – | – | – | – | – |
| RegNet | RegNetY-1.6GF | – | – | – | – | – | – |
| MobileNet | MobileNetV2 | – | – | – | – | – | – |
| MobileNet | MobileNetV3-S | – | – | – | – | – | – |
| MobileNet | MobileNetV3-L | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B0 | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B2 | – | – | – | – | – | – |
| EfficientNet | EfficientNetV2-S | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-N | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-T | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-S | – | – | – | – | – | – |
| ViT / DeiT | ViT-S/16 | – | – | – | – | – | – |
| ViT / DeiT | ViT-B/16 | – | – | – | – | – | – |
| ViT / DeiT | DeiT-S | – | – | – | – | – | – |
| Swin | Swin-T | – | – | – | – | – | – |
| Swin | Swin-S | – | – | – | – | – | – |
| MobileViT | MobileViT-XS | – | – | – | – | – | – |
| MobileViT | MobileViT-S | – | – | – | – | – | – |
| MobileViT | MobileViTv2-1.0 | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-S/14 | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-B/14 | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/16 | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/32 | – | – | – | – | – | – |

## E3b – Một ảnh mẫu mỗi người (%, trung bình ± CI95; đường CMC trong results/E3b_cmc.csv)

| Họ | Mô hình | Rank-1 | Rank-5 | Rank-10 | AUCMC |
| --- | --- | --- | --- | --- | --- |
| VGG | VGG-11 | – | – | – | – |
| VGG | VGG-16 | – | – | – | – |
| ResNet | ResNet-18 | – | – | – | – |
| ResNet | ResNet-50 | – | – | – | – |
| ResNet | ResNet-101 | – | – | – | – |
| DenseNet | DenseNet-121 | – | – | – | – |
| DenseNet | DenseNet-169 | – | – | – | – |
| RegNet | RegNetY-800MF | – | – | – | – |
| RegNet | RegNetY-1.6GF | – | – | – | – |
| MobileNet | MobileNetV2 | – | – | – | – |
| MobileNet | MobileNetV3-S | – | – | – | – |
| MobileNet | MobileNetV3-L | – | – | – | – |
| EfficientNet | EfficientNet-B0 | – | – | – | – |
| EfficientNet | EfficientNet-B2 | – | – | – | – |
| EfficientNet | EfficientNetV2-S | – | – | – | – |
| ConvNeXt | ConvNeXt-N | – | – | – | – |
| ConvNeXt | ConvNeXt-T | – | – | – | – |
| ConvNeXt | ConvNeXt-S | – | – | – | – |
| ViT / DeiT | ViT-S/16 | – | – | – | – |
| ViT / DeiT | ViT-B/16 | – | – | – | – |
| ViT / DeiT | DeiT-S | – | – | – | – |
| Swin | Swin-T | – | – | – | – |
| Swin | Swin-S | – | – | – | – |
| MobileViT | MobileViT-XS | – | – | – | – |
| MobileViT | MobileViT-S | – | – | – | – |
| MobileViT | MobileViTv2-1.0 | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-S/14 | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-B/14 | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/16 | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/32 | – | – | – | – |

## E4 – Gender, few-shot (Balanced accuracy %, trung bình ± CI95)

| Họ | Mô hình | Proto K=1 | Proto K=2 | Proto K=5 | Proto K=10 | LP K=5 | LP K=10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VGG | VGG-11 | – | – | – | – | – | – |
| VGG | VGG-16 | – | – | – | – | – | – |
| ResNet | ResNet-18 | – | – | – | – | – | – |
| ResNet | ResNet-50 | – | – | – | – | – | – |
| ResNet | ResNet-101 | – | – | – | – | – | – |
| DenseNet | DenseNet-121 | – | – | – | – | – | – |
| DenseNet | DenseNet-169 | – | – | – | – | – | – |
| RegNet | RegNetY-800MF | – | – | – | – | – | – |
| RegNet | RegNetY-1.6GF | – | – | – | – | – | – |
| MobileNet | MobileNetV2 | – | – | – | – | – | – |
| MobileNet | MobileNetV3-S | – | – | – | – | – | – |
| MobileNet | MobileNetV3-L | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B0 | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B2 | – | – | – | – | – | – |
| EfficientNet | EfficientNetV2-S | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-N | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-T | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-S | – | – | – | – | – | – |
| ViT / DeiT | ViT-S/16 | – | – | – | – | – | – |
| ViT / DeiT | ViT-B/16 | – | – | – | – | – | – |
| ViT / DeiT | DeiT-S | – | – | – | – | – | – |
| Swin | Swin-T | – | – | – | – | – | – |
| Swin | Swin-S | – | – | – | – | – | – |
| MobileViT | MobileViT-XS | – | – | – | – | – | – |
| MobileViT | MobileViT-S | – | – | – | – | – | – |
| MobileViT | MobileViTv2-1.0 | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-S/14 | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-B/14 | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/16 | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/32 | – | – | – | – | – | – |

## E5 + E6 – Không giám sát (%)

| Họ | Mô hình | Gender ACC | Gender NMI | Identity ACC | Identity NMI | Identity ARI |
| --- | --- | --- | --- | --- | --- | --- |
| VGG | VGG-11 | – | – | – | – | – |
| VGG | VGG-16 | – | – | – | – | – |
| ResNet | ResNet-18 | – | – | – | – | – |
| ResNet | ResNet-50 | – | – | – | – | – |
| ResNet | ResNet-101 | – | – | – | – | – |
| DenseNet | DenseNet-121 | – | – | – | – | – |
| DenseNet | DenseNet-169 | – | – | – | – | – |
| RegNet | RegNetY-800MF | – | – | – | – | – |
| RegNet | RegNetY-1.6GF | – | – | – | – | – |
| MobileNet | MobileNetV2 | – | – | – | – | – |
| MobileNet | MobileNetV3-S | – | – | – | – | – |
| MobileNet | MobileNetV3-L | – | – | – | – | – |
| EfficientNet | EfficientNet-B0 | – | – | – | – | – |
| EfficientNet | EfficientNet-B2 | – | – | – | – | – |
| EfficientNet | EfficientNetV2-S | – | – | – | – | – |
| ConvNeXt | ConvNeXt-N | – | – | – | – | – |
| ConvNeXt | ConvNeXt-T | – | – | – | – | – |
| ConvNeXt | ConvNeXt-S | – | – | – | – | – |
| ViT / DeiT | ViT-S/16 | – | – | – | – | – |
| ViT / DeiT | ViT-B/16 | – | – | – | – | – |
| ViT / DeiT | DeiT-S | – | – | – | – | – |
| Swin | Swin-T | – | – | – | – | – |
| Swin | Swin-S | – | – | – | – | – |
| MobileViT | MobileViT-XS | – | – | – | – | – |
| MobileViT | MobileViT-S | – | – | – | – | – |
| MobileViT | MobileViTv2-1.0 | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-S/14 | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-B/14 | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/16 | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/32 | – | – | – | – | – |

## E7 – Nhận dạng người mới (%)

| Họ | Mô hình | 1-shot Top-1 (tinh chỉnh) | 5-shot Top-1 (tinh chỉnh) | EER (tinh chỉnh) | TAR@FAR=1% (tinh chỉnh) | 1-shot Top-1 (đóng băng) | 5-shot Top-1 (đóng băng) | EER (đóng băng) | TAR@FAR=1% (đóng băng) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VGG | VGG-11 | – | – | – | – | – | – | – | – |
| VGG | VGG-16 | – | – | – | – | – | – | – | – |
| ResNet | ResNet-18 | – | – | – | – | – | – | – | – |
| ResNet | ResNet-50 | – | – | – | – | – | – | – | – |
| ResNet | ResNet-101 | – | – | – | – | – | – | – | – |
| DenseNet | DenseNet-121 | – | – | – | – | – | – | – | – |
| DenseNet | DenseNet-169 | – | – | – | – | – | – | – | – |
| RegNet | RegNetY-800MF | – | – | – | – | – | – | – | – |
| RegNet | RegNetY-1.6GF | – | – | – | – | – | – | – | – |
| MobileNet | MobileNetV2 | – | – | – | – | – | – | – | – |
| MobileNet | MobileNetV3-S | – | – | – | – | – | – | – | – |
| MobileNet | MobileNetV3-L | – | – | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B0 | – | – | – | – | – | – | – | – |
| EfficientNet | EfficientNet-B2 | – | – | – | – | – | – | – | – |
| EfficientNet | EfficientNetV2-S | – | – | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-N | – | – | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-T | – | – | – | – | – | – | – | – |
| ConvNeXt | ConvNeXt-S | – | – | – | – | – | – | – | – |
| ViT / DeiT | ViT-S/16 | – | – | – | – | – | – | – | – |
| ViT / DeiT | ViT-B/16 | – | – | – | – | – | – | – | – |
| ViT / DeiT | DeiT-S | – | – | – | – | – | – | – | – |
| Swin | Swin-T | – | – | – | – | – | – | – | – |
| Swin | Swin-S | – | – | – | – | – | – | – | – |
| MobileViT | MobileViT-XS | – | – | – | – | – | – | – | – |
| MobileViT | MobileViT-S | – | – | – | – | – | – | – | – |
| MobileViT | MobileViTv2-1.0 | – | – | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-S/14 | – | – | – | – | – | – | – | – |
| Pretrain nền tảng | DINOv2 ViT-B/14 | – | – | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/16 | – | – | – | – | – | – | – | – |
| Pretrain nền tảng | CLIP ViT-B/32 | – | – | – | – | – | – | – | – |

## E8 – Thử chéo dataset (%)

_(chưa có kết quả)_
