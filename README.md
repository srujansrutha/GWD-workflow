# Global Wheat Detection Workflow

Reproducible YOLO11m workflow for wheat-head detection using the Kaggle Global Wheat Detection dataset.

The project measures more than in-domain validation accuracy. It uses three splits:

| Split | Contents | Used for |
| --- | --- | --- |
| `train` | 90% of the remaining five farms | Learning |
| `val` | Random 10% of those five farms | Early stopping and comparing every experiment |
| `test` | `usask_1` and `ethz_1`, held out entirely | Scored for the baseline and the final model only |

Experiments are chosen on `val` alone, so `test` remains an honest estimate of performance on unseen farms.

## YOLO11s Baseline

| Evaluation split | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| In-domain validation | 0.9448 | 0.5248 |
| Test (unseen farms) | 0.8900 | 0.3720 |
| Generalization gap | 0.0548 | 0.1528 |

## YOLO11m Combined Run

YOLO11m at `imgsz=1280`, batch 2, 100 epochs, source-balanced sampling and extra blur/noise/shadow augmentation. All rows below are scored with the same protocol (no test-time augmentation).

| Evaluation split | YOLO11s baseline mAP50-95 | YOLO11m combined mAP50-95 |
| --- | ---: | ---: |
| Validation (seen farms) | 0.5248 | 0.5340 |
| Test (unseen: `usask_1`, `ethz_1`) | 0.3720 | 0.3791 |
| Test2 (unseen: official GWHD 2021 test farms) | 0.1487 | 0.1662 |

Gains on unseen farms were small (+0.007 and +0.018) and the generalization gap did not shrink. Test-time augmentation helped the baseline on `val` but lowered the combined model's `test` score, so it is not enabled by default. The full analysis, every setting, and interview notes are in [docs/TRAINING.md](docs/TRAINING.md).

## More Training Farms (Global Wheat Head Dataset 2021)

`notebooks/train_gwhd2021.ipynb` adds 1,707 images from new farms and countries to the training set, keeps `val` and `test` unchanged, and holds out the official 2021 test farms as `test2` (1,380 images, never trained on). All rows use the same protocol (mAP50-95, no test-time augmentation).

| Model | Trained on | Val | Test | Test2 |
| --- | --- | ---: | ---: | ---: |
| YOLO11s | old data | 0.5248 | 0.3720 | 0.1487 |
| YOLO11m combined | old data | 0.5340 | 0.3791 | 0.1662 |
| **YOLO11s** | old + new farms | 0.5243 | 0.3855 | **0.3005** |
| YOLO11m | old + new farms | 0.5289 | **0.3916** | 0.2799 |

More diverse training farms doubled the `test2` score with the same model, while a bigger model added nothing reliable on top. Details and the leakage checks are in [docs/TRAINING.md](docs/TRAINING.md) section 8.3.

## Layout

- `docs/TRAINING.md`: detailed training guide, experiment history, and interview Q&A.
- `notebooks/train.ipynb`: data preparation, training, evaluation, and qualitative inspection.
- `notebooks/train_gwhd2021.ipynb`: adds the 2021 dataset, checks for leakage, trains and compares models.
- `data/raw/`: Kaggle download containing `train.csv` and the source images. Ignored by Git.
- `data/yolo/`: generated YOLO images, labels, and dataset definitions. Ignored by Git.
- `weights/`: downloaded pretrained weights and locally trained checkpoints. Ignored by Git.
- `runs/detect/`: generated Ultralytics training and evaluation artifacts. Ignored by Git.

## Setup

Create a Python environment, install the dependencies, download the competition data, then open `notebooks/train.ipynb` from the repository root.

```powershell
pip install -r requirements.txt
pip install kaggle
kaggle competitions download -c global-wheat-detection -p data/raw
```

Extract the downloaded archive so `data/raw/train.csv` and `data/raw/train/` exist. The notebook finds the repository root automatically and writes all generated files under this repository.

GPU training is expected. The recorded YOLO11s baseline used an NVIDIA GeForce RTX 5050 Laptop GPU with 8 GB VRAM, `imgsz=1024`, and `batch=4`.

## Running The Workflow

Run the notebook cells in order:

1. Prepare YOLO labels and the train, val, and test splits.
2. Run the three-epoch smoke test.
3. Run the 60-epoch YOLO11m training.
4. Evaluate the best checkpoint on `val` and `test`.

The repository does not version raw images, generated run artifacts, or model weights. Keep those locally or store them in dedicated dataset/model storage.