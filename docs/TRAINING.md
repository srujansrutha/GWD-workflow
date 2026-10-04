# Wheat Head Detection: Training Guide, Revision Notes and Interview Prep

This document explains everything done in this project: the data, the pipeline, every training setting and why it was chosen, all experiments with their real results, the mistakes made along the way, and a question-and-answer section for interviews.

All numbers below were measured in this repository. Where something is an inference and not a measured fact, it is marked as such.

---

## 1. The 60-second summary

- **Task:** detect wheat heads (one class) in overhead field photos, using the Kaggle *Global Wheat Detection* dataset.
- **Model:** Ultralytics YOLO11 (small, then medium), fine-tuned from COCO-pretrained weights.
- **The key idea:** most projects report accuracy on a random validation split. Here, two whole farms (`usask_1`, `ethz_1`) are **held out entirely**, so the `test` score measures performance on farms the model has never seen. That is what matters in the real world, because the competition's hidden test set came from other countries.
- **Result:** on familiar farms the model scores mAP50 about 0.95 and mAP50-95 about 0.53. On unseen farms, mAP50 is 0.89 and mAP50-95 is 0.37. The drop is the **domain shift**.
- **An honest finding:** a combined run with five changes (bigger model, higher resolution, longer training, balanced sampling, extra augmentation) took about 9 hours and gave only **small** gains on unseen farms when scored under identical settings: `test` mAP50-95 0.372 → 0.379 (+0.007, near the noise level) and `test2` (a second, harder unseen set) 0.149 → 0.166 (+0.018). That is modest for the compute, which points to data diversity and label quality as the bigger lever.
- **The biggest win came from data, not the model.** Adding 1,707 training images from new farms (Japan, Norway, Belgium, Australia, China, more of France) with the same YOLO11s recipe took the second unseen set (`test2`) from mAP50-95 **0.149 to 0.300** and mAP50 from 0.509 to 0.729, with no change on familiar farms. A bigger model on top of that data added nothing reliable. See section 8.3.
- **A correction worth remembering:** an earlier version of this guide said the combined run did not improve at all. That compared a run scored *with* test-time augmentation against a baseline scored *without* it. Test-time augmentation lowered the combined model's `test` score by 0.007, which hid its gain. Always score models under one identical protocol.

---

## 2. Numbers to remember

| Item | Value |
|---|---|
| Images (all labelled) | 3,422 (3,373 with boxes, 49 background-only) |
| Image size | 1024 × 1024 (all) |
| Annotations | 147,793 rows, 147,786 after cleaning |
| Heads per image | mean 43.8, median 43, max 116 |
| Head size | median about 74 px (5th to 95th percentile: 43 to 127 px) |
| Farms (sources) | 7 |
| Split | train 2,228 / val 247 / test 947 images |
| Hardware | RTX 5050 Laptop GPU, 8 GB VRAM |
| Software | Python 3.13, PyTorch 2.14 (CUDA 13), Ultralytics 8.4.164 |
| YOLO11m size | 20.05 M parameters, 68.3 GFLOPs (at 640 px) |

### Results

All numbers below were scored with the **same protocol**: no test-time augmentation, each model at its own training image size.

| Model | `val` mAP50 | `val` mAP50-95 | `test` mAP50 | `test` mAP50-95 | `test2` mAP50 | `test2` mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11s baseline (1024, batch 4, 60 epochs) | 0.945 | 0.525 | 0.890 | 0.372 | 0.509 | 0.149 |
| YOLO11m combined run (1280, batch 2, 100 epochs) | 0.950 | 0.534 | 0.896 | 0.379 | 0.543 | 0.166 |

`test2` is the 1,380 official Global Wheat Head Dataset 2021 competition-test images (18 farm groups from the US, Mexico, China, Australia, Japan and Sudan). Neither model was trained on them. They are much harder than `test`, which shows how much the "unseen farm" score depends on which farms you pick.

The generalization gap (`val` minus `test`, mAP50-95) is 0.153 for the baseline and 0.155 for the combined run, essentially the same because `val` rose too.

**Test-time augmentation (TTA)**, measured separately: on the baseline checkpoint, `val` only, mAP50-95 rose 0.525 → 0.531 (+0.006) with inference about 2× slower. On the combined model, TTA *lowered* `test` mAP50-95 from 0.379 to 0.372 and left `val` at 0.534. So TTA is not guaranteed to help on unseen farms; measure it per evaluation set.

---

## 3. The dataset

**Source:** Kaggle competition *Global Wheat Detection* (`train.csv` plus images). Each row of `train.csv` is one box:

| Column | Meaning |
|---|---|
| `image_id` | Image file name, without extension |
| `width`, `height` | Image size (always 1024) |
| `bbox` | **A string** like `[834.0, 222.0, 56.0, 36.0]`, meaning `[x_min, y_min, w, h]` in pixels |
| `source` | Which farm or dataset the image came from |

**Farms in the data:**

| Source | Images | Boxes | Boxes/image | Median head size (px) | Role |
|---|---:|---:|---:|---:|---|
| arvalis_1 | 1,055 | 45,716 | 43.3 | 76 | train/val |
| arvalis_2 | 204 | 4,179 | 20.5 | 91 | train/val |
| arvalis_3 | 559 | 16,661 | 29.8 | 87 | train/val |
| inrae_1 | 176 | 3,701 | 21.0 | 110 | train/val |
| rres_1 | 432 | 20,235 | 46.8 | 74 | train/val |
| **ethz_1** | 747 | 51,489 | 68.9 | 64 | **test (held out)** |
| **usask_1** | 200 | 5,805 | 29.0 | 93 | **test (held out)** |

The farms differ a lot: head density ranges from about 20 to about 69 per image, and head size from about 64 to 110 px. That variety is why generalization to new farms is hard.

**Known weaknesses of the data:**
- Boxes were drawn by hand and are inconsistent, especially for overlapping or partly hidden heads. This caps mAP50-95, which punishes loose boxes.
- Only seven farms (five for training), and the three arvalis farms are about 75% of the training-farm images (arvalis_1 alone is about 43%).

---

## 4. The pipeline, step by step

Everything lives in `notebooks/train.ipynb`, run top to bottom.

### 4.1 Environment check
Confirms Python, PyTorch, CUDA, mixed-precision (AMP) and Ultralytics all work before spending hours on training.

### 4.2 Parse the `bbox` strings
`bbox` is text, not a list. The notebook splits it with vectorized string operations (much faster than `ast.literal_eval` on 148k rows) into numeric `x, y, w, h`.

### 4.3 Clean the boxes
1. **Clip** boxes to the image (some run past the edge).
2. **Drop** boxes with a side under 2 px (2 found) and boxes with area above 200,000 px² (5 found, annotation slips).
3. Result: 147,793 → 147,786 boxes. The data was already fairly clean.

### 4.4 Convert to YOLO format
YOLO wants `class x_center y_center width height`, all **divided by the image size** so values are between 0 and 1:

```
xc = ((x1 + x2) / 2) / W        yc = ((y1 + y2) / 2) / H
w  = (x2 - x1) / W              h  = (y2 - y1) / H
```

This is where silent bugs happen: the CSV gives a *corner plus size*, YOLO wants a *center plus size*. Nothing errors if you mix them up, the model just learns garbage. Hence the verification steps below.

### 4.5 Build the splits (the most important design decision)

| Split | Images | Boxes | What it is | Used for |
|---|---:|---:|---|---|
| `train` | 2,228 | 81,505 | 90% of images from the five training farms | Learning |
| `val` | 247 | 8,987 | Random 10% of images from the **same five farms** (seed 42) | Early stopping, choosing the best checkpoint, comparing experiments |
| `test` | 947 | 57,294 | `usask_1` and `ethz_1`, held out **entirely** | The honest unseen-farm score, scored for the baseline and the final model only |

Why not a random 80/20 split? Photos from the same farm look alike (same camera, soil, lighting, often neighboring spots in one field). A random split puts near-duplicates in both train and validation, which gives an inflated score. A farm-level split for `test` removes that leakage.

**Caveat:** `val` still comes from the same farms as `train`, so it is optimistic as an in-domain estimate. It was also used for early stopping and picking `best.pt`, which tunes to it slightly.

### 4.6 Verify before training
- Every label: class is 0, all values in [0, 1], boxes fit inside the image.
- Round trip: convert one box back to pixels and compare with the original CSV row.
- `assert bad == 0`, so the notebook refuses to continue if the conversion is wrong.

### 4.7 Dataset config (`data/yolo/wheat.yaml`)
Points Ultralytics at `images/train`, `images/val`, `images/test`, one class named `wheat_head`.

### 4.8 Source-balanced sampling (added for the combined run)
`arvalis_1` has about 945 training images and `inrae_1` about 159, so the model would mostly tune itself to arvalis. The fix: write the training images to a text file, repeating images from smaller farms `round(sqrt(largest / its_size))` times. Square-root weighting evens things out without filling each epoch with copies of 160 images.

| Farm | Training images | Repeat | Seen per epoch |
|---|---:|---:|---:|
| arvalis_1 | 945 | 1 | 945 |
| arvalis_2 | 184 | 2 | 368 |
| arvalis_3 | 507 | 1 | 507 |
| inrae_1 | 159 | 2 | 318 |
| rres_1 | 387 | 2 | 774 |

2,228 unique images become 2,958 per epoch. `val` and `test` are never repeated.

### 4.9 Extra augmentation (added for the combined run)
Passed via `augmentations=` to simulate camera and environment differences: motion, Gaussian and defocus blur (p=0.25), Gaussian and ISO noise (p=0.2 and 0.1), random shadows (p=0.2), JPEG compression artifacts (p=0.2), CLAHE (p=0.05). This replaces Ultralytics' default Albumentations list, which only fires at p=0.01.

---

## 5. The model: YOLO11

YOLO ("You Only Look Once") is a **one-stage detector**: one network pass predicts all boxes, with no separate region-proposal step. That makes it fast.

**Architecture (three parts):**
1. **Backbone:** extracts features at several scales (convolutions, C3k2 blocks, SPPF pooling, a C2PSA attention block).
2. **Neck:** mixes features across scales (FPN/PAN-style), so small and large objects both get good features.
3. **Head:** predicts, at each location on three feature maps, a box and a class score. It is **anchor-free** (it predicts box distances directly, not offsets from preset anchor shapes) and **decoupled** (separate branches for box and class).

**Training signals (loss):**
- **Classification:** binary cross-entropy.
- **Box:** CIoU loss (overlap, center distance and aspect ratio).
- **DFL (Distribution Focal Loss):** predicts each box edge as a probability distribution, which sharpens localization.
- **Label assignment:** a task-aligned assigner decides which predictions count as positives for each ground-truth box, based on both classification and box quality.

**At inference:** predictions are filtered by confidence, then **NMS (non-maximum suppression)** removes duplicate overlapping boxes.

**Sizes used:**

| Model | Parameters | Role |
|---|---|---|
| YOLO11s (small) | about 9 M | Fast baseline |
| YOLO11m (medium) | 20.05 M | Tested whether more capacity helps |

Both start from **COCO-pretrained weights** (transfer learning): the network already knows edges, textures and objects, and only needs to adapt to wheat. This beats training from scratch on about 2,000 images.

---

## 6. Metrics, explained properly

- **IoU (Intersection over Union):** overlap area divided by union area of a predicted and a true box. 1 = perfect, 0 = no overlap.
- **True positive:** a predicted box whose IoU with an unmatched true box is above a threshold. **False positive:** a prediction with no match. **False negative:** a true head with no matching prediction.
- **Precision** = TP / (TP + FP): of the boxes predicted, how many were right.
- **Recall** = TP / (TP + FN): of the real heads, how many were found.
- **AP (average precision):** area under the precision-recall curve, made by sweeping the confidence threshold. With one class, mAP equals AP.
- **mAP50:** AP when a match needs IoU of at least 0.50 (a loose match). It answers "did the model find the head?".
- **mAP50-95:** AP averaged over IoU thresholds 0.50, 0.55, …, 0.95. It answers "did the model find it **and** box it tightly?". It punishes loose boxes heavily, which is why it is much lower than mAP50.
- **Precision and recall in the summary** are reported at the confidence that maximizes F1.
- **Fitness** (used to pick `best.pt` and for early stopping) = 0.1 × mAP50 + 0.9 × mAP50-95.

Ultralytics uses COCO-style mAP, which differs from the Kaggle leaderboard metric, so our numbers are not comparable to leaderboard scores.

**Why the gap is bigger for mAP50-95 than mAP50** (0.153 vs 0.055 for the baseline): on unseen farms the model usually still finds the heads (recall holds up) but boxes are less precise, since head size, density and appearance differ from what it learned.

---

## 7. Training configuration and the reason for every setting

### 7.1 Settings that were the same in both runs

| Setting | Value | Why |
|---|---|---|
| Pretrained weights | COCO (`yolo11s.pt`, `yolo11m.pt`) | Transfer learning |
| Optimizer | `auto`, which chose **AdamW**, lr 0.002, momentum 0.9 | With short training (fewer than about 10,000 iterations) Ultralytics picks AdamW; on long runs it picks SGD with lr 0.01. This is visible in the logs. |
| `lrf` | 0.01 | Learning rate decays linearly to 1% of the start value |
| `weight_decay` | 0.0005 | Mild regularization (not applied to biases or norm layers) |
| `warmup_epochs` | 3 | Ramp the learning rate up to avoid early instability |
| Loss gains | box 7.5, cls 0.5, dfl 1.5 | Ultralytics defaults. Box quality is weighted most because localization is the hard part. |
| `nbs` | 64 | Nominal batch size: gradients accumulate to about 64 images per optimizer step regardless of the real batch |
| AMP | on | Mixed precision (float16): faster and uses less memory |
| `seed` | 42, deterministic | Reproducibility |
| `cache` | off | Saves RAM |
| `workers` | 2 | Windows and Jupyter dataloaders hang with more |

### 7.2 Augmentation (identical in both runs unless noted)

| Setting | Value | Why |
|---|---|---|
| `mosaic` | 1.0 | Stitches 4 images into one. Shows wheat at mismatched scales and backgrounds, which gives cheap domain robustness. A big win in the Kaggle competition. |
| `close_mosaic` | 10 | Mosaic turned off for the last 10 epochs so the model finishes on realistic images |
| `mixup` | 0.1 | Blends two images lightly |
| `hsv_h / hsv_s / hsv_v` | 0.015 / 0.7 / 0.4 | Strong color jitter: unseen farms differ in lighting, soil tone and crop maturity |
| `fliplr` / `flipud` | 0.5 / 0.5 | Overhead photos have no "up", so vertical flips are physically valid |
| `degrees` | 15 | Same reasoning (no preferred orientation) |
| `scale` / `translate` | 0.5 / 0.1 | Simulate camera distance and framing |
| Extra Albumentations (combined run only) | see 4.9 | Camera quality, noise, shadow, compression |

The rule behind every choice: an augmentation should produce images that **could really occur** in deployment. Flipping text would be wrong; flipping overhead wheat is fine.

### 7.3 What changed between the two runs

| Setting | YOLO11s baseline | YOLO11m combined run |
|---|---|---|
| Model | yolo11s | yolo11m |
| `imgsz` | 1024 | **1280** (upscaled; the native size is 1024) |
| `batch` | 4 | **2** (see section 9) |
| `epochs` / `patience` | 60 / 20 | **100 / 30** |
| Training data | plain train folder | **source-balanced list** |
| Extra augmentation | none | **blur, noise, shadow, compression** |
| Evaluation | no TTA | no TTA (TTA measured separately, see section 2) |

### 7.4 Why `imgsz=1280` when the images are 1024
Upscaling adds no new information. The reasoning was that heads cover more feature-map cells (the detector's finest grid is 8 px per cell), which can help box precision. It costs about 1.56× the pixels and much more memory. It did not give a measurable gain here.

### 7.5 Why batch size 2 is not a problem
Because of `nbs=64`, Ultralytics accumulates gradients over about 32 steps before updating, so the optimizer sees an effective batch near 64 either way. The real batch mainly affects **batch-norm statistics** (noisy with very small batches) and speed. Batch 1 would be unreliable, so 2 is the minimum sensible value.

---

## 8. Experiments, in order

| # | Run | What | Result |
|---|---|---|---|
| 1 | `smoke` | YOLO11s, 3 epochs, 1024, batch 4 | `val` mAP50-95 0.494. Proved the pipeline and memory fit (peak about 3.1 GB). |
| 2 | `baseline` | YOLO11s, 60 epochs | `val` 0.945 / 0.525, `test` 0.890 / 0.372. About 1.5 hours. |
| 3 | TTA check | Baseline checkpoint, `val`, with and without TTA | mAP50-95 0.525 → 0.531 (+0.006), inference 2× slower |
| 4 | `yolo11m_smoke` | YOLO11m, 3 epochs, 1024, batch 2 | 0.463 (a bigger model starts slower, so this says little) |
| 5 | Batch probe | YOLO11m at 1280, 10% of an epoch, batch 2 vs 4 | Batch 2: 4.5 GB, about 4 it/s. Batch 4: 8.9 GB (over the 8.15 GB card, spilling into system RAM), about 2.7× slower. |
| 6 | `yolo11m_combo_smoke` | The full combined setup, 3 epochs | 0.482 |
| 7 | `yolo11m_combo` | Combined run, 100 epochs | No TTA: `val` 0.950 / 0.534, `test` 0.896 / 0.379, `test2` 0.543 / 0.166. With TTA: `val` 0.948 / 0.534, `test` 0.890 / 0.372. About 9 hours. Best epoch: 85. |

### 8.1 Reading the training curves

`val` mAP50-95 for the combined run: 0.490 at epoch 20, 0.505 at 40, 0.520 at 60, 0.530 at 80, **0.534 at 85**, then flat around 0.53 to epoch 100. The baseline was still rising at epoch 60 (0.523), when it stopped.

### 8.2 What the results mean

1. **On familiar farms the combined run gained 0.009 mAP50-95.** At the same epoch (20, 60) it was slightly *behind* the baseline, so its lead comes from epochs 61 to 85, which the baseline never trained. This suggests the gain is mostly from training longer. It is an inference: the learning-rate schedule also differs between a 60 and a 100 epoch run, so the comparison is not perfect.
2. **On unseen farms there was a small, consistent gain.** `test` mAP50-95 rose from 0.372 to 0.379 (+0.007, near the ±0.005 noise band) and `test2` from 0.149 to 0.166 (+0.018, measured over 1,380 images from 18 farms, so more convincing). mAP50 rose by 0.006 and 0.035. So the combined changes helped a little where they were aimed.
3. **The gap did not shrink** (0.153 vs 0.155), because familiar-farm accuracy rose by a similar amount.
4. **TTA is not free accuracy.** It helped the baseline on `val` (+0.006) but lowered the combined model's `test` score (-0.007). An earlier analysis compared TTA numbers against no-TTA numbers and wrongly concluded the combined run had not improved.
5. **Conclusion:** model size, resolution and augmentation each give at most modest gains for 9 hours of compute. The much larger difference between `test` (0.37) and `test2` (0.15) suggests that *which farms* you are tested on matters more than any of these changes, which points toward training-farm diversity as the bigger lever. We changed everything at once, so we cannot say which single change helped.

---

### 8.3 Experiment: more training farms (Global Wheat Head Dataset 2021)

**Question:** is the limit the number of farms the model has seen? **Notebook:** `notebooks/train_gwhd2021.ipynb`.

**Design (change one thing).** Same YOLO11s recipe as the baseline (1024 px, batch 4, 60 epochs, same augmentation and seed). Only the training images change:

| Set | Images | Source |
|---|---:|---|
| `train` | 2,228 old + **1,707 new** = 3,935 | New farms from the official GWHD 2021 train/val CSVs |
| `val` | 247 | Unchanged |
| `test` | 947 | Unchanged (`usask_1` + `ethz_1`) |
| `test2` | 1,380 | Official GWHD 2021 competition-test farms (18 groups: US, Mexico, China, Australia, Japan, Sudan), **never trained on by any model** |

**Leakage defence (the interesting engineering).** The 2021 archive contains all 3,422 original images, saved as PNG under new hash names, and it **renamed and re-split the original farms** (the Kaggle `arvalis_1` with 1,055 images is spread across several `Arvalis_*` groups). So excluding farms by name was not safe. Instead every image was fingerprinted with a 16 × 16 grayscale thumbnail and matched to the nearest old image. The distances split cleanly into two groups (3,422 near-identical images vs 3,087 clearly new, with nothing in between), and the method was validated: the archive's `ETHZ_1` + `Usask_1` images mapped one-to-one onto our 947 test images. A final assertion confirms no `val` or `test` image is in the training list.

**Label-format trap:** the 2021 CSV stores corner boxes (`x_min y_min x_max y_max`, joined by `;`, `no_box` for empty), unlike the original `x_min y_min width height`. A round-trip check proved the conversion.

**Results (mAP50-95, same protocol, no TTA):**

| Model | Trained on | `val` | `test` | `test2` |
|---|---|---:|---:|---:|
| YOLO11s baseline | old data | 0.525 | 0.372 | 0.149 |
| YOLO11m combined (9 h) | old data | 0.534 | 0.379 | 0.166 |
| **YOLO11s + GWHD 2021** | old + new | 0.524 | 0.386 | **0.300** |
| YOLO11m + GWHD 2021 | old + new | 0.529 | **0.392** | 0.280 |

mAP50 on `test2`: 0.509 → 0.729 (YOLO11s + new data). Generalization gap (`val` − `test2`): 0.376 → 0.224.

**What it shows:**
1. **Data diversity was the main lever.** The same small model doubled its `test2` score and improved `test` by 0.014, with `val` unchanged. That is the signature of better generalization and not just more fitting.
2. **A bigger model added nothing reliable on top.** YOLO11m vs YOLO11s on the same data: `val` +0.005 (noise), `test` +0.006 (barely above noise), `test2` −0.021. By farm it was mixed: better on Australian, Japanese and Sudanese farms, worse on US (KSU, Terraref) and Chinese (NAU) farms. YOLO11s costs about half as much, so it is the better choice here.
3. **The gain is uneven, and part of it is "same region, different farm".** The largest gains were on farms with related new training data (NAU +0.34, UQ +0.14 to +0.23, Ukyoto +0.15). Farms with no related training data improved much less (KSU +0.06 to +0.11, CIMMYT 0 to +0.08) and Terraref stayed very poor (about 0.04 to 0.08).
4. **Caveat:** one run per configuration, so differences under about 0.005 are noise, and `test2` farms are not all independent of the new training regions.

### 8.4 What to try next
1. Retrain on **all** farms except `usask_1` and `ethz_1` (set `HOLDOUT_OFFICIAL_TEST = False`). It adds the `test2` regions (US, Mexico, China, Australia), and `test` stays an honest unseen check. The cost is losing `test2` as a measuring set.
2. Collect or label images for the weakest regions (Terraref-style gantry images, US and Mexican farms).
3. Repeat the best run with a second seed to measure the noise.
4. Pseudo-label unlabeled field photos from other sources, never from the test farms.

## 9. Problems hit along the way and how they were solved

| Problem | Cause | Fix |
|---|---|---|
| Baseline in the README was trained from another folder | `runs/detect/baseline/args.yaml` pointed at `FineTune_RAG` | Noted. The split code and seed (42) were unchanged, so the same `val` and `test` images were used and the numbers remain comparable. |
| Three splits needed | Two held-out farms were used both to choose experiments and to report results, which would leak into the final score | Renamed `ood` to `test`. Rule: **choose experiments on `val` only**, score `test` once. |
| Hard-coded absolute paths in the yaml | Generated by the notebook for one machine | The notebook regenerates the yaml at run time. |
| Batch 4 at 1280 was 2.7× slower | VRAM 8.9 GB over the 8.15 GB card, so Windows spilled to shared memory | Measured with a probe and stayed at batch 2. |
| Probe script hung on Windows | Dataloader workers re-import the script | Put the code under `if __name__ == "__main__":`. |
| A stray test process kept using the GPU | A leftover from the first hung probe | Found it with `nvidia-smi` and process inspection, killed it. |
| Long run inside Jupyter is fragile | Kernel or window closing kills it | Ran it headless as a background process and relied on `last.pt` checkpoints to **resume** (`resume=True`). |
| Notebook overwritten by VS Code | The editor re-saved its own copy while a script edited the file | Edited JSON in place and asked to reload the notebook before saving. |
| Renamed farms in the 2021 dataset | Re-grouped under new names with different sizes, so name matching could leak test images | Matched images by thumbnail fingerprints and validated against the known test set |
| Unfair TTA comparison | Scored the new model with TTA and the baseline without it | Rescored every model under one protocol; TTA turned out to hurt on unseen farms |
| Slow epochs late in the run | Epochs 88 to 93 took 470 to 1,185 s instead of about 315 s | Cause not determined (the machine was not dedicated to training). |

---

## 10. Reproducing the project

```powershell
pip install -r requirements.txt
pip install kaggle
kaggle competitions download -c global-wheat-detection -p data/raw
# extract so data/raw/train.csv and data/raw/train/ exist
```

Open `notebooks/train.ipynb` from the repository root and run in order: data preparation → balanced list and augmentation → smoke test → full training → evaluation. To run unattended:

```powershell
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute notebooks\train.ipynb --output-dir runs\notebook --output train_executed --ExecutePreprocessor.timeout=-1
```

To resume an interrupted run:

```python
from ultralytics import YOLO
YOLO(r"runs\detect\yolo11m_combo\weights\last.pt").train(resume=True)
```

**Where things are:**

| Path | Contents |
|---|---|
| `notebooks/train.ipynb` | The whole pipeline |
| `data/yolo/wheat.yaml` | Dataset config (train / val / test) |
| `runs/detect/<run>/results.csv` | Metrics per epoch |
| `runs/detect/<run>/weights/best.pt` | Best checkpoint by fitness |
| `runs/detect/yolo11m_combo/final_scores.txt` | Final `val` and `test` scores |
| `weights/` | Pretrained starting weights |

Raw data, generated images and labels, weights and run folders are not versioned (see `.gitignore`).

---

## 11. Interview questions and answers

### Project and framing

**Q: Describe the project in one minute.**
I trained YOLO11 to detect wheat heads in overhead field images from the Global Wheat Detection dataset. The distinctive part is the evaluation: I held out two entire farms so the test score measures generalization to farms the model never saw, not just a random validation split. A YOLO11s baseline gets about 0.95 mAP50 on familiar farms and 0.89 on unseen ones, and the tight-box metric mAP50-95 drops from 0.52 to 0.37. I then tried a bigger model, higher resolution, longer training, balanced sampling and extra augmentation. It improved the unseen-farm score only slightly (+0.007 on one set, +0.018 on a second, harder one) for about nine hours of training, which suggests the bigger lever is training-farm diversity, not model size.

**Q: Why this problem?**
Because it has a built-in domain-shift problem. Wheat looks very different across countries and cameras, and the original competition's hidden test set came from continents not in the training data. That makes it a good test of whether a model generalizes or just memorizes.

**Q: Why YOLO and not Faster R-CNN or a transformer detector?**
YOLO is a one-stage detector: fast to train and run, easy to fine-tune with a mature toolchain, and strong on dense small objects like wheat heads. Two-stage detectors can be slightly more accurate on some tasks but are slower and more complex. With one class, about 2,000 images and an 8 GB laptop GPU, YOLO was the practical choice.

**Q: Why one class?**
Every box is a wheat head; there are no other categories in the labels. So `nc=1`, and mAP equals the AP of that one class.

### Data

**Q: What is the most important decision in the project?**
The split. I held out two whole farms as the test set, because images from the same farm share camera, lighting and soil, so a random split leaks near-duplicates into validation and inflates the score.

**Q: Why three splits and not two?**
With only two (train and held-out farms) I would be using the held-out farms both to choose between experiments and to report the final number. After many experiments the number would be partly tuned to those farms. So `val` (random 10% of the training farms) is for decisions and early stopping, and `test` is scored for the baseline and the final model only.

**Q: Is your `val` score a fair in-domain estimate?**
It is optimistic. Validation images come from the same farms as training, and photos from one farm can be very similar. I also picked the best checkpoint and early-stopped on it. It is good for comparing experiments, not an unbiased in-domain number.

**Q: What data problems did you find?**
Boxes extending past the image edge (clipped), 2 degenerate boxes under 2 px, 5 oversized boxes from annotation slips (dropped), 49 background-only images (kept with empty label files, which is correct), strong imbalance between farms (the three arvalis farms are about 75% of training-farm images), and generally inconsistent hand-drawn boxes.

**Q: What bug is most likely when converting the labels?**
The CSV stores `[x_min, y_min, w, h]` in pixels, YOLO wants `[x_center, y_center, w, h]` normalized by image size. Mixing up corner and center produces no error but trains garbage. I guarded against it with a range check on every label and a round trip of one box back to pixels.

**Q: How did you handle class imbalance across farms?**
By repeating images from smaller farms in the training list, with a square-root weighting so a small farm is not repeated so much that it dominates. Heavily repeating 160 images mostly teaches the model those exact images.

**Q: Why not add the test farms to training?**
That would remove the only honest estimate of how the model behaves on unseen farms. The same applies to pseudo-labelling on them.

### Metrics

**Q: Explain mAP50 vs mAP50-95.**
mAP50 counts a detection as correct if it overlaps the true box by at least 50% IoU, so it measures "did you find it". mAP50-95 averages over IoU thresholds from 0.50 to 0.95, so it also rewards tight boxes. A model can have high mAP50 and low mAP50-95 if it finds objects but boxes them loosely, which is exactly the pattern on unseen farms.

**Q: Which metric would you pick for this task?**
It depends on the use. For counting heads or estimating yield, finding them (mAP50, recall) matters more than box tightness. For measuring size, mAP50-95 matters. I reported both and was explicit about which one the gap is in.

**Q: What are precision and recall here, and how did they change?**
Precision is the fraction of predicted boxes that are real heads, recall is the fraction of real heads found. For the combined model on `val`, precision is 0.92 and recall 0.89. On `test`, precision is 0.90 and recall 0.86, so it both misses more heads and makes slightly more false detections on unseen farms.

### Training

**Q: Why transfer learning?**
The pretrained network already encodes edges, textures and object structure from COCO. With about 2,000 images, fine-tuning converges faster and generalizes better than training from scratch.

**Q: What optimizer and learning rate did you use?**
`optimizer=auto`. With short training Ultralytics selects AdamW at lr 0.002 with momentum 0.9 (the logs confirm it). The learning rate warms up for 3 epochs and then decays linearly to 1% of its starting value.

**Q: Why was the batch size so small, and does it hurt?**
8 GB of VRAM at high resolution. Ultralytics accumulates gradients to a nominal batch of 64, so the optimizer's effective batch is similar. The main effect of a small real batch is noisier batch-norm statistics. I measured that batch 4 at 1280 needed 8.9 GB, spilled into system memory, and ran 2.7× slower, so batch 2 was both necessary and sensible.

**Q: Explain your augmentation choices.**
Each one targets a real source of variation between farms. Mosaic mixes scales and backgrounds. Strong HSV jitter covers lighting, soil tone and crop maturity. Vertical flips and rotation are valid because the photos are overhead with no up. Scale and translation simulate camera distance. In the combined run I added blur, noise, shadow and JPEG artifacts for camera quality. I turn mosaic off for the last 10 epochs so the model finishes on realistic images.

**Q: What does `close_mosaic` do and why?**
It disables mosaic for the final epochs. Mosaic images are artificial composites, so finishing on natural single images aligns training with what the model sees at inference.

**Q: How do you know when to stop training?**
Early stopping (`patience`) on a validation fitness score, 0.1·mAP50 + 0.9·mAP50-95. Also read the curves: if validation is still rising at the end, you stopped too early, as with the 60-epoch baseline. If train loss falls while validation stalls, you are overfitting.

**Q: What does "best.pt" mean?**
The checkpoint with the best validation fitness during training. Ultralytics also keeps `last.pt` for resuming, and the saved weights are an exponential moving average of the training weights, which tend to be smoother and slightly better.

**Q: What is test-time augmentation?**
Running inference on several transformed versions of each image (different scales and a flip) and merging the predictions. It gave +0.006 mAP50-95 on `val` for the baseline at twice the inference time, so it can be worth it for offline evaluation. But it is not guaranteed: on the combined model it lowered the unseen-farm `test` score by 0.007, so measure it per dataset.

**Q: How did you make a long training run robust?**
Checkpoint every epoch, run it as a detached background process instead of inside the notebook kernel, and resume with `resume=True`, which restores the model, optimizer state, schedule and all arguments. I paused and resumed twice this way.

### Results and judgment

**Q: Did the improvements work?**
Modestly. Scored under one protocol, familiar-farm mAP50-95 rose from 0.525 to 0.534, unseen-farm `test` from 0.372 to 0.379, and a second unseen set (`test2`) from 0.149 to 0.166. The gap between familiar and unseen farms did not shrink (0.153 vs 0.155). So about 9 hours bought roughly +0.01 on unseen farms.

I also made a comparison mistake worth admitting: I first scored the new model with test-time augmentation and the baseline without, which made the new model look no better. Rescoring both the same way showed the small gain, because TTA actually hurts on unseen farms for that model.

**Q: Why not just say the 0.009 is an improvement?**
It is about twice the run-to-run noise of ±0.005 at best, and the new run was behind the baseline at equal epochs, so most of the familiar-farm gain appears to come from simply training longer. A single run per configuration cannot separate a real gain from luck at this size.

**Q: Why did you change five things at once?**
Because each full run is hours on a laptop GPU and I wanted to know whether any combination could move the unseen-farm score. The cost is that I cannot attribute the result. Since the total gain was small, that was an acceptable trade, but to know which change mattered I would need one-change-at-a-time ablations.

**Q: What do you think limits unseen-farm performance?**
My hypothesis, supported but not proven: the number and diversity of training farms (only five), and inconsistent labels. Model capacity, resolution and photometric augmentation together gave only small gains, which points away from them as the main limit. Direct evidence would come from a per-farm error analysis and from training with more farms.

**Q: What would you do next?**
1. Per-farm error analysis: which farm fails and how (missed heads, doubled boxes, loose boxes).
2. Add more training farms from a larger public wheat dataset.
3. Audit the labels for the worst-scoring images.
4. Controlled ablations at lower cost: one change at a time on YOLO11s at 1024.
5. Pseudo-label unlabelled images from other sources, never from the test farms.

**Q: How would you deploy this?**
Export to ONNX or TensorRT, choose the confidence threshold on `val` for the target metric (F1 for counting, higher recall for safety), measure latency at the real input size, and consider tiling for large images. Then monitor in production, because performance drifts as new farms and cameras appear.

### The extra-data experiment

**Q: What did adding the 2021 dataset do?**
With exactly the same model and training recipe, adding 1,707 images from new farms roughly doubled the score on a second unseen set (mAP50-95 0.149 to 0.300, mAP50 0.509 to 0.729) and improved the original unseen test by 0.014, while familiar-farm accuracy stayed the same. It was the biggest gain of the whole project, bigger than a larger model, higher resolution, balanced sampling and extra augmentation combined.

**Q: How did you make sure the new dataset did not leak your test set?**
The new archive contains our original images under new file names, and it had renamed the farms, so I could not trust names. I fingerprinted every image with a 16 × 16 grayscale thumbnail and matched each to its nearest old image. The distances formed two clean groups, and I validated the method: the archive's two test-farm groups mapped one-to-one onto my 947 test images. I also asserted that no validation or test file appears in the training list.

**Q: Did the bigger model help once you had more data?**
Not reliably. On the same data, YOLO11m beat YOLO11s by 0.006 on `test` (barely above noise) and was worse by 0.021 on `test2`, with mixed results by farm. It costs about twice the compute, so I would use the smaller model.

**Q: Why did some farms improve a lot and others barely?**
The big gains were on farms with related new training data: other farms in the same country or from the same collection, such as China (NAU) and Australia (UQ). Farms with no related data, such as US gantry images (Terraref), improved little. So the benefit is partly regional coverage, which tells you what data to collect next.

**Q: Why did you hold out a second test set?**
Two farms can be lucky or unlucky. A second set from different continents that no model trains on gives independent evidence, and it let me score the earlier models fairly too.

### Debugging and engineering

**Q: What went wrong during the project?**
The baseline run in the README had been trained from a different folder, which I caught by reading its `args.yaml` and checked that the split was unchanged. Batch 4 at 1280 ran 2.7× slower because it exceeded VRAM. A test script hung on Windows because dataloader workers re-import the script without a `main` guard, and a leftover process kept using the GPU until I found it. And the notebook file was overwritten by the editor while I was changing it.

**Q: How would you detect data leakage?**
Check that no image, and no near-duplicate or same-field sequence, appears across splits; split by the entity that matters (farm, patient, session) and not by image; and be suspicious of any validation score far above training or far above the held-out score.

**Q: How do you debug a training run that does not improve?**
Verify the labels by drawing them; overfit a tiny subset to prove the pipeline can learn; check learning rate and warmup; look at the loss curves for NaN or spikes; check augmentation is not destroying the objects; check the data yaml points to the right folders.

**Q: How do you ensure reproducibility?**
Fixed seed and deterministic mode, saved arguments in each run folder (`args.yaml`), pinned library versions in `requirements.txt`, a single notebook that regenerates all data from the raw CSV, and logged metrics per epoch.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Domain shift** | Training and deployment data differ (new farms, cameras, lighting), so performance drops |
| **Transfer learning** | Starting from weights pretrained on another dataset |
| **Anchor-free** | The head predicts boxes directly, with no predefined box shapes |
| **NMS** | Removes duplicate overlapping predictions, keeping the most confident |
| **IoU** | Overlap area divided by union area of two boxes |
| **mAP** | Mean average precision, the area under the precision-recall curve averaged over classes (and IoU thresholds) |
| **Epoch** | One pass over the training list |
| **Warmup** | A short period at the start where the learning rate ramps up |
| **AMP** | Automatic mixed precision, using float16 where safe |
| **EMA** | Exponential moving average of weights, giving smoother final weights |
| **TTA** | Test-time augmentation: merging predictions over several transformed inputs |
| **Mosaic** | Augmentation that stitches 4 images into one |
| **Smoke test** | A very short run to prove nothing crashes and memory fits |
| **Ablation** | Removing or changing one component to measure its effect |
| **Leakage** | Information from the evaluation set influencing training or model selection |

---

## 13. Tuning cheat sheet: what each knob does and when to use it

Use this as a checklist. First find the symptom, then change only the matching knob.

### 13.1 The knobs

| Knob | What it does | Change it when | Cost or catch |
|---|---|---|---|
| **Training data** (more farms, cleaner labels) | Gives the model more situations to learn from | Scores drop on new places | Effort to collect. It was the biggest win here (`test2` 0.149 to 0.300). |
| **Model size** (s, m, l) | A bigger model that can learn finer detail | Training score is also low (underfitting) | About 2x slower, more memory. Gave at most +0.009 here. |
| **Image size** (`imgsz`) | More pixels per object | Small objects are missed | Memory grows fast. Past the photo's real size it adds no new detail. |
| **Epochs** | How long it trains | The validation curve is still rising at the end | More time. `patience` stops it when it stops improving. |
| **Learning rate** (`lr0`) | Size of each learning step | Loss jumps or becomes NaN: lower it. Learning is painfully slow: raise it. | Leave on `auto` until everything else is done. |
| **Batch size** | Images processed at once | Never for accuracy, only for speed | Ultralytics accumulates to a nominal batch of 64, so scores barely change. |
| **Augmentation** | Makes copies of images look different | The model overfits, or deployment looks different from training | Only use changes that could really happen. Too strong hides small objects. |
| **Weight decay, dropout** | Stops the model memorizing | Training score is much higher than validation | Too much causes underfitting. |
| **Pretrained weights** | Starts from knowledge learned elsewhere | Almost always, especially with small data | None. Never train from scratch on about 2,000 images. |
| **Loss gains** (`box`, `cls`, `dfl`) | Which mistake the model cares about most | Boxes are loose (`box`, `dfl`) or classes get confused (`cls`) | Rarely needed. |
| **Confidence threshold** (`conf`) | Minimum certainty to show a box | Too many false boxes: raise. Missing heads: lower. | A trade-off; choose it on `val` for your goal. |
| **Test-time augmentation** | Predicts on flipped or resized copies and merges them | Last, to squeeze out a little more | It hurt the YOLO11m model on unseen farms. Always test it first. |

### 13.2 Symptom to action

| Symptom | Likely cause | Try, in order |
|---|---|---|
| Training and validation scores both low | Underfitting | Train longer, then a bigger model, then a higher learning rate |
| Training high, validation low | Overfitting | More data, stronger augmentation, weight decay |
| Good on familiar farms, bad on new ones | Not enough variety | **More diverse data**, then colour and blur augmentation |
| Finds heads but boxes are loose (high mAP50, low mAP50-95) | Localization or noisy labels | Higher `box` and `dfl`, check labels, higher `imgsz` |
| Misses small heads | Resolution | Higher `imgsz`, or tile large images |
| Score is jumpy | Learning rate or small validation set | Lower learning rate, larger validation set |
| Loss becomes NaN | Learning rate too high or bad labels | Lower learning rate, check labels |

### 13.3 Choose by what you are short of

- **Little GPU memory:** lower the batch first, then the image size, then the model size. Avoid batch 1.
- **Little time:** smaller model, fewer epochs, `fraction=0.5` for experiments. Run the long version only for the best idea.
- **Little data:** pretrained weights, strong realistic augmentation, and pseudo-labels on unlabeled photos (never on the test farms).
- **Little labelling effort:** fix the worst-scoring training images first. A few wrong labels cost more than many correct ones add.

### 13.4 Order of attack

1. Data and split first. No model change fixes a leaky split or bad labels.
2. Read the curves, then change **one** knob.
3. Compare on `val` only, and look at `test` once at the end.
4. Ignore differences under about 0.005; that is run-to-run noise.
5. Learning rate, hyperparameter search and test-time augmentation come last.

### 13.5 What our own experiments showed

| Change | Result |
|---|---|
| More farms and countries (data) | **Big gain** on unseen farms |
| Bigger model, higher resolution, balanced sampling, extra augmentation (9 h) | Small gain (+0.007 to +0.018) |
| Bigger model on the new data | No reliable gain over the small model |
| Batch size | No effect expected, so not tested |
| Test-time augmentation | Helped one model, hurt another |

When scores are stuck, the answer is usually better or more varied data before fancier tricks.

## 14. Honest limitations (say these in an interview)

- One training run per configuration, so differences under about 0.005 mAP are not distinguishable from noise.
- The combined run changed five things together, so no single change can be credited or blamed.
- `val` is optimistic as an in-domain estimate; `test` covers just two farms, so the unseen-farm score is itself an estimate with its own uncertainty.
- The baseline was trained in a previous folder (same split and seed). An earlier comparison mixed TTA and no-TTA scores; all numbers here use one protocol (no TTA).
- In the extra-data experiment, `test2` shares regions with some new training farms, so its gain overstates how well the model handles completely unrelated farms. Each configuration was run once.
- The explanation that data diversity and label quality limit the score is a hypothesis supported by the evidence, not something proven here.
