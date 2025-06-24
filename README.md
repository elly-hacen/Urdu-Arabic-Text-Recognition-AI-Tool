# Urdu/Arabic - Text Recognition AI Tool

UA-TRAIT is a deep learning–based neural network OCR tool built specifically for recognizing **Urdu & Arabic** scripts. 
It is built using a modern architecture: **HRNet** for feature extraction, **DBiLSTM** for sequence modeling, and **CTC Loss** for transcription.

---

## Requirements

- Python 3.11.5
- [`uv`](https://github.com/astral-sh/uv) for dependency and environment management
- PyTorch, torchvision, lmdb, numpy, and related libraries

---

## Dataset

UA-TRAIT is trained on a hybrid Urdu + Arabic dataset:

| Language | Source     | Fonts/Samples |
|----------|------------|---------------|
| Arabic   | Synthetic  | 18 fonts      |
| Urdu     | Real-world | UTRSet Real   |

### Setup Dataset

```bash
unzip Dataset.zip -d UA-TRAIT/
````

Expected structure:

```
UA-TRAIT/
└── dataset/
    ├── train/
    │   ├── test/
    │   └── gt.txt
    └── test/
        ├── test/
        └── gt.txt
```

---
### Install Dependencies

1.  Install `uv`:
```bash
pip install uv
```
2. Inside `UA-TRAIT` dir run this cmd:

```bash
uv sync
```
## Creating LMDB Datasets

### For Training

```bash
uv run create_lmdb_dataset.py \
  --inputPath dataset/train/ \
  --gtFile dataset/train/gt.txt \
  --outputPath ./lmdb_train
```

### For Testing

```bash
uv run create_lmdb_dataset.py \
  --inputPath dataset/test/ \
  --gtFile dataset/test/gt.txt \
  --outputPath ./lmdb_test
```

---

## Training the Model

```bash
uv run train.py \
  --train_data ./lmdb_train \
  --valid_data ./lmdb_test \
  --FeatureExtraction HRNet \
  --SequenceModeling DBiLSTM \
  --Prediction CTC \
  --exp_name UA-TRAIT \
  --num_epochs 100 \
  --batch_size 8
```

---

## Model Architecture

| Component         | Description                                      |
| ----------------- | ------------------------------------------------ |
| Feature Extractor | HRNet (High-Resolution Network)                  |
| Sequence Model    | DBiLSTM (Bidirectional LSTM ×2)                  |
| Prediction        | CTC Loss (Connectionist Temporal Classification) |

---


## References

* Python EOL: [https://devguide.python.org/versions/](https://devguide.python.org/versions/)
* HRNet Paper: [https://arxiv.org/abs/1904.04514](https://arxiv.org/abs/1904.04514)
* CTC Explanation: [https://distill.pub/2017/ctc/](https://distill.pub/2017/ctc/)
* UTRNet Repo: [https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition.git](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition.git)
* uv Packaging Tool: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

---

## Project Motivation & Modernization

This project is reimplementation of [**UTRNet**](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition.git), an early Urdu OCR tool built in Python 3.7 with older PyTorch dependencies.

### Why a Migration Was Needed?

- **Python 3.7 reached EOL** on *June 27, 2023* → [Reference](https://devguide.python.org/versions/)
- The original code was incompatible with Python 3.11+ and recent PyTorch versions
- Training crashed due to:
  - In-place operations affecting `autograd`
  - Unicode errors from unregistered characters
  - Dependency conflicts with updated libraries

### Migration Highlights

- Migrated from **Python 3.7 → 3.11.5**
- Upgraded all major dependencies (Torch, torchvision, etc.)
- Patched **Unicode charset issues**
- Fixed in-place ops causing autograd failures
- Integrated [`uv`](https://github.com/astral-sh/uv) for faster environment syncing
- Refactored dataset processing and improved training stability

---

## Unicode Crash Fixes

Some characters in the original dataset triggered runtime crashes during training:

- `U+0649` — Arabic Alef Maksura
- `U+06CC` — Farsi Yeh / Bari Yeh (Urdu)
- `U+FE87` — Alef with Hamza Below (Isolated Form)

These characters were missing from the original charset
