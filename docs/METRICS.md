# Metrics Reference: Detection Scores, COCO Metrics and Count Scores

A reference for every score used to judge an object detector, written around this project (wheat-head detection and counting). Part 1 to 6 are general. Part 7 is specific to this repository and was **checked against the installed Ultralytics 8.4.164 source**. Part 8 gives our measured numbers.

There is no universal list of every possible metric (specialized tasks add others), but this covers the practical foundation: the main detection scores, all 12 standard COCO metrics, and the count scores that matter for a head-counting app.

## 0. One-line cheat sheet

| Score | The question it answers | Better |
|---|---|---|
| IoU | How well does a predicted box overlap the true box? | Higher |
| Precision | Of the boxes drawn, how many were right? | Higher |
| Recall | Of the real heads, how many were found? | Higher |
| F1 | Are precision and recall both good? | Higher |
| AP / mAP | How good is detection across all confidence thresholds? | Higher |
| mAP50 | Did it find the objects (loose box matching)? | Higher |
| mAP75 | Did it find them with well-placed boxes? | Higher |
| mAP50-95 | Did it find them across loose to very strict box matching? | Higher |
| AR@k | How many real objects does it find when it may output at most k boxes? | Higher |
| Signed error / bias | Does it count too many or too few? | Closer to 0 |
| MAE | By how many heads is a photo's count off on average? | Lower |
| RMSE | The same, but punishing big misses more. | Lower |
| MAPE | By what percentage is the count off on average? | Lower |
| Latency, FPS, GPU memory | How fast and how heavy is it? | Lower / higher / lower |

---

## 1. Basic detection measurements

Before any score is calculated, the evaluator **matches** predicted boxes to the labelled boxes (the ground truth).

### 1.1 IoU: Intersection over Union

**Question:** "How well does this predicted box overlap the correct box?"

$$\text{IoU}=\frac{\text{area shared by both boxes}}{\text{area covered by either box}}$$

- IoU = 1: the boxes overlap perfectly. IoU = 0: they do not overlap. IoU = 0.50: the shared area is half the combined area.
- Higher is better.
- IoU is a **matching** measurement, not a complete detector score. A model can draw excellent boxes around some heads while missing many others.

### 1.2 TP, FP and FN

These counts feed precision and recall.

| Measurement | Simple meaning | Wheat example |
|---|---|---|
| True positive (TP) | A prediction correctly matches a real object | A box matches a labelled head closely enough |
| False positive (FP) | A prediction does not correctly match a real object | A box is drawn on a leaf, or an extra box duplicates a head that already has one |
| False negative (FN) | A real object has no matching prediction | A labelled head is missed |

Matching normally needs the correct class **and** enough IoU. A badly placed box can cause both an FP (the box matches nothing) and an FN (the real head stays unmatched).

### 1.3 Confidence score

**Question:** "How sure is the model about this prediction?"

A box might have confidence 0.90. The app uses a **confidence threshold** to decide which predictions to keep.

This is a prediction output, not a quality score. A confidence of 0.90 does not automatically mean a 90% chance of being correct: that depends on how well the model's confidence is *calibrated*.

---

## 2. Main detection scores

### 2.1 Precision

**Question:** "Of the boxes the model drew, how many were correct?"

$$\text{Precision}=\frac{TP}{TP+FP}$$

Example: 100 boxes drawn, 90 correctly match heads, so precision = 90%. High precision means few false detections. It does **not** mean all heads were found.

### 2.2 Recall

**Question:** "Of all the real heads, how many did the model find?"

$$\text{Recall}=\frac{TP}{TP+FN}$$

Example: the photo has 100 heads and the model finds 80, so recall = 80%. High recall means few missed heads. It does **not** mean every box is correct.

### 2.3 F1-score

**Question:** "How good is the balance between precision and recall?"

$$F1=\frac{2\times\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}$$

- Higher is better. F1 is high only when **both** precision and recall are reasonably high.
- It gives precision and recall equal weight.
- It is useful for choosing a confidence threshold (the threshold with the highest F1 is the best balance).

Precision, recall and F1 all depend on the confidence threshold and the matching rules, so always record those settings when comparing models.

### 2.4 AP: Average Precision

**Question:** "How well does the model balance correct detections and missed objects across confidence thresholds?"

Changing the confidence threshold produces different (precision, recall) pairs. **AP summarizes the resulting precision-recall curve for one class**, using the evaluator's interpolation rules.

- Higher is better.
- AP is *not* simply the average of precision and recall.
- It judges the model across all thresholds, not at one chosen threshold.

### 2.5 mAP: Mean Average Precision

**Question:** "What is the average AP across the object classes?"

For a model detecting heads, helmets and people:

$$\text{mAP}=\frac{AP_{\text{head}}+AP_{\text{helmet}}+AP_{\text{person}}}{3}$$

With only one class ("wheat head"), **AP and mAP are the same** under the same settings. Always state the IoU convention: "mAP" alone is incomplete.

### 2.6 mAP50, mAP75 and mAP50-95

| Name | IoU needed to count as a match | In simple words | Notes |
|---|---|---|---|
| **mAP50** | at least 0.50 | "Did the model find the objects with reasonably overlapping boxes?" | Forgiving of box placement |
| **mAP75** | at least 0.75 | "Did it find them with accurately placed boxes?" | A big gap between mAP50 and mAP75 means boxes overlap enough at 0.50 but often fail the tighter test |
| **mAP50-95** | average over 0.50, 0.55, ..., 0.95 (10 thresholds) | "How good is detection from forgiving to very strict matching?" | The main COCO metric; the stricter overall score shown in YOLO validation results |

A mAP50-95 of 0.413 means **41.3% on that metric**, not "41.3% of heads were found".

---

## 3. All 12 COCO metrics

COCO reports **six AP metrics and six average-recall (AR) metrics**. Its summary calls the class-averaged result "AP", which is also commonly called mAP.

| Metric | Simple explanation |
|---|---|
| AP | Overall AP averaged over IoU 0.50 to 0.95 |
| AP50 | AP using IoU 0.50: more forgiving of box placement |
| AP75 | AP using IoU 0.75: stricter about box placement |
| AP-small | AP (0.50-0.95) for small objects |
| AP-medium | AP (0.50-0.95) for medium objects |
| AP-large | AP (0.50-0.95) for large objects |
| AR@1 | Average recall when at most 1 detection per image per category is kept |
| AR@10 | Average recall with at most 10 detections per image per category |
| AR@100 | Average recall with at most 100 detections per image per category |
| AR-small | Average recall for small objects (standard 100-detection limit) |
| AR-medium | Average recall for medium objects (standard 100-detection limit) |
| AR-large | Average recall for large objects (standard 100-detection limit) |

- All are **higher-is-better**.
- AR averages recall across IoU thresholds under the stated detection cap. It is **not** the same as recall at one confidence threshold.
- COCO's size groups use the **box area** in pixels, with boundaries at 32² and 96² pixels (small is under 32², large is over 96²). For detection this is the box area, not just width or height. These are benchmark definitions, not universal definitions of a "small" or "large" head.
- **Detection cap:** if evaluation keeps only 100 predictions per image, a photo with more than 100 real heads cannot reach full recall. (This project's evaluator uses a higher cap; see part 7.)
- For a counting app, the size-specific scores help expose missed distant or tiny heads.

---

## 4. Count scores

These judge the **number** of heads in a photo, not where the boxes are.

Let the count error of photo *i* be:

$$e_i=\text{predicted count}_i-\text{actual count}_i$$

A **positive** error means overcounting; a **negative** error means undercounting.

### 4.1 The standard count metrics

| Metric | Simple meaning | Better |
|---|---|---|
| Signed error | Predicted minus actual for one photo. Actual 100, predicted 94 gives -6 | Closer to 0 |
| Absolute error | The size of the mistake, ignoring direction. The same example gives 6 | Lower |
| **MAE** (Mean Absolute Error) | Average absolute error over photos. MAE = 6 means off by 6 heads per photo on average | Lower |
| **MSE** (Mean Squared Error) | Average squared error. Large misses get much more weight. Units are heads squared | Lower |
| **RMSE** (Root Mean Squared Error) | Square root of MSE. Highlights large misses but is back in units of heads | Lower |
| **Bias** (mean signed error) | Average signed error. Bias = -4 means undercounting by 4 heads per photo on average | Closer to 0 |
| **MAPE** (Mean Absolute Percentage Error) | Average absolute error as a percentage of each photo's actual count | Lower |

MAE, MSE and RMSE measure the **size** of errors; bias measures their **direction**. MAPE allows relative comparisons but is undefined when the actual count is zero and can mislead for very small counts.

### 4.2 Formulas

For *N* photos:

$$\text{MAE}=\frac1N\sum_i|e_i| \qquad \text{MSE}=\frac1N\sum_i e_i^2 \qquad \text{RMSE}=\sqrt{\frac1N\sum_i e_i^2}$$

$$\text{Bias}=\frac1N\sum_i e_i \qquad \text{MAPE}=\frac{100}{N}\sum_i\frac{|e_i|}{\text{actual count}_i}$$

The MAPE formula needs every actual count to be non-zero.

### 4.3 Why bias alone is not enough

Suppose two photos have count errors of +10 and -10:

- Bias = 0 heads (the errors cancel)
- MAE = 10 heads
- RMSE = 10 heads

There is no average directional bias, but the counts are still wrong. Likewise, false detections and missed heads can cancel within **one** photo, giving a correct total even though individual detections are wrong.

### 4.4 Custom count measures

Application-specific measures are common, but they are reporting choices, not part of standard COCO. **State their definitions explicitly.**

- **Exact-count rate:** share of photos where predicted count equals actual count.
- **Within-tolerance rate:** share of photos within a fixed allowance, such as ±5 heads.
- **Percentage-tolerance rate:** share within a relative allowance, such as ±10%, with a separate rule for empty photos.
- **95th-percentile absolute error:** the error that 95% of photos stay at or below.
- **Median absolute percentage error:** like MAPE but using the median, so a few tiny-count photos do not dominate it.
- **Relative total bias:** predicted total divided by actual total, minus 1, over a whole group of photos.

---

## 5. Diagnostics and practical measurements

### 5.1 Diagnostic charts

| Report | What it helps you understand |
|---|---|
| Precision-recall curve | How precision changes as recall increases across confidence thresholds |
| Precision-confidence curve | How reliable the kept detections are at each threshold |
| Recall-confidence curve | How many real objects stay detected as the threshold changes |
| F1-confidence curve | Which threshold gives the best precision-recall balance |
| Confusion matrix | Class mix-ups, missed objects and false detections (for a one-class model, mainly misses and false detections) |

These explain performance; they are not independent extra scores. Ultralytics saves all of them in each run folder (for example `BoxF1_curve.png`).

### 5.2 Speed and resource measurements

- **Latency:** time to process one image, in milliseconds.
- **FPS / throughput:** images processed per second.
- **GPU memory:** memory needed to run the model.
- **Parameters and model size:** storage and capacity, not accuracy.

Distinguish model-only inference time from the whole app pipeline (loading, resizing, post-processing, counting), and compare speed on the same hardware and input size.

---

## 6. Interpretation rules

1. **Confidence is not IoU.** Confidence filters predictions; IoU judges box overlap.
2. **Two different IoU thresholds exist.** The *NMS* IoU compares predictions with *other predictions* to remove duplicates. The *evaluation* IoU compares predictions with *ground truth*. Do not mix them up.
3. **Training losses** (box, classification, DFL) are optimization signals, not substitutes for validation scores.
4. **Good counts do not guarantee good boxes, and good mAP does not guarantee accurate counts** at the app's chosen confidence threshold. Evaluate both.
5. **Report the settings** (confidence, IoU, image size, test-time augmentation) next to every number, and compare models under one identical protocol.

---

## 7. How this project measures things (verified in the installed code)

Checked against Ultralytics **8.4.164**, the version used here.

| Setting | What this project's evaluation does |
|---|---|
| mAP IoU thresholds | 10 thresholds, 0.50 to 0.95 in steps of 0.05 (mAP50-95), plus mAP50 and mAP75 available |
| AP interpolation | 101-point interpolation, as in COCO |
| Confidence during `val` | Default **0.001** (almost everything is kept so the full curve can be drawn) |
| NMS IoU | **0.7** (used to remove duplicate boxes) |
| Detection cap per image | **300** (`max_det`), higher than COCO's 100. The busiest photo has 190 heads (in `test2`) and 75 `test2` photos have more than 100, so a cap of 100 would have limited recall there; 300 does not |
| Reported precision and recall | Taken at the confidence that **maximizes F1** (on a smoothed F1 curve) |
| Checkpoint selection and early stopping ("fitness") | **mAP50-95 only.** In this version the weights are `[P, R, mAP50, mAP50-95] = [0, 0, 0, 1]`. Older versions used `0.1 x mAP50 + 0.9 x mAP50-95` |
| COCO size splits and AR | Not reported by Ultralytics (no AP-small/medium/large, no AR@k) |
| Leaderboard comparability | Ultralytics uses COCO-style mAP, which differs from the Kaggle Global Wheat Detection leaderboard metric, so our numbers are not comparable to leaderboard scores |

---

## 8. Our results, in these terms

Final model: **YOLO11s trained on the original + GWHD 2021 farms** (the model inside the app). Evaluation sets: `val` (random 10% of the training farms), `test` (usask_1 + ethz_1, never trained on) and `test2` (official GWHD 2021 test farms from other countries, never trained on).

### 8.1 Detection scores

| Set | mAP50 | mAP50-95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| val | 0.949 | 0.524 | 0.919 | 0.894 | 0.906 |
| test | 0.896 | 0.386 | 0.892 | 0.859 | 0.875 |
| test2 | 0.729 | 0.301 | 0.808 | 0.666 | 0.730 |

Precision and recall are at the F1-maximizing confidence; F1 was calculated from them as 2PR/(P+R).

### 8.2 Count scores at confidence 0.25 (the app's default)

Predicted count vs. hand-labelled count per photo.

| Measure | val | test | test2 |
|---|---:|---:|---:|
| Photos | 247 | 947 | 1,380 |
| Average actual heads per photo | 36.4 | 60.5 | 48.8 |
| **MAE** (heads) | 2.65 | 3.39 | 7.97 |
| **RMSE** (heads) | 3.51 | 4.55 | 11.08 |
| **Bias** (mean signed error, heads) | +1.10 | +1.04 | -6.84 |
| Relative total bias | +3.0% | +1.7% | -14.0% |
| **MAPE** (mean absolute % error) | 7.9% | 6.7% | 22.7% |
| Median absolute % error | 6.0% | 4.6% | 13.2% |
| 95th-percentile absolute error (heads) | 7.0 | 9.0 | 23.0 |
| Exact-count rate | 12.6% | 11.7% | 8.2% |
| Within ±5 heads | 89.5% | 81.3% | 49.1% |
| Within ±10% | 71.3% | 79.9% | 39.9% |
| Worst overcount / undercount (heads) | +11 / -9 | +25 / -23 | +26 / -54 |

Notes:
- MAPE and the percentage rates skip photos with zero real heads (3 in `val`, 47 in `test2`), because the percentage is undefined for them.
- Mean MAPE is larger than the median because photos with few heads give large percentages from small absolute errors. The median is the steadier "typical" figure.
- **The app's accuracy table** reports the *median absolute % error* as "Typical count error" and the *relative total bias* as "Average bias".
- Exact-count rates look low (about 10%) but errors are small: on `val` and `test`, 80 to 90% of photos are within ±5 heads.
- Count scores are more forgiving than mAP: a missed head and a false box in different places cancel in the total.

### 8.3 Choosing the confidence threshold with count scores

MAE (heads) / relative total bias, by threshold:

| Confidence | val | test | test2 |
|---:|---|---|---|
| 0.15 | 7.29 / +19.9% | 11.24 / +18.4% | 11.25 / +13.9% |
| 0.20 | 3.89 / +9.5% | 5.47 / +7.9% | 6.90 / -3.8% |
| **0.25** | **2.65 / +3.0%** | **3.39 / +1.7%** | 7.97 / -14.0% |
| 0.30 | 2.60 / -1.2% | 3.67 / -2.8% | 10.45 / -20.8% |
| 0.40 | 4.07 / -9.0% | 6.98 / -10.7% | 15.45 / -31.5% |
| 0.50 | 6.60 / -17.5% | 11.98 / -19.6% | 21.13 / -43.3% |

Lower thresholds count more heads (and more false ones). The best setting is where the bias crosses zero: about 0.25 to 0.30 on familiar farms, about 0.20 on unfamiliar ones. The app defaults to 0.25 and offers a slider.

---

## 9. Which metric for which question

| If you care about... | Look at |
|---|---|
| The number of heads per photo | MAE, bias, median absolute % error (and RMSE for big misses) |
| Whether the app over- or under-counts | Bias, relative total bias |
| Finding heads (few misses) | Recall, mAP50 |
| Few false alarms | Precision |
| One balanced detection number | F1, mAP50 |
| Tight, well-placed boxes | mAP50-95, mAP75 |
| Choosing the confidence threshold | F1-confidence curve, count error vs. threshold |
| Whether small or distant heads are missed | AP-small, AR-small (COCO), or a per-size breakdown |
| Speed and cost | Latency, FPS, GPU memory |

---

## 10. Sources

The general definitions (parts 1 to 6) are adapted from your notes, which cite these pages. I did not independently re-check the web pages; I did check the Ultralytics-specific facts in part 7 against the installed source code.

- COCO evaluation metrics: https://deepwiki.com/cocodataset/cocoapi/6.1-evaluation-metrics
- Object-detection metrics and mAP: https://condados.ai/blog/object-detection-metrics-map
- F1-score: https://www.ultralytics.com/glossary/f1-score
- Ultralytics metrics discussion: https://github.com/ultralytics/ultralytics/issues/7307
- Confidence calibration: https://link.springer.com/article/10.1007/s11263-024-02219-z
- Error metrics (MAE, RMSE, MAPE): https://www.jedox.com/en/blog/error-metrics-how-to-evaluate-forecasts/
- YOLO performance metrics guide: https://github.com/cobank1633/ultralytics_yolov8/blob/main/docs/en/guides/yolo-performance-metrics.md
- Ultralytics validation system: https://deepwiki.com/ultralytics/ultralytics/4.2-validation-system-and-metrics
