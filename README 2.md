# Robustness of Object Detection under Image Blur and Sharpening

**A controlled computer-vision experiment measuring how Gaussian blur degrades YOLO11n detection accuracy on COCO128 — and whether classical sharpening can recover it.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-0A0A0A?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-scientific--computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Real-world vision systems rarely receive clean images. Cameras shake, lenses lose focus, video is re-encoded, and frames arrive compressed — yet detection models are almost always benchmarked on pristine data. This project measures part of that gap using a controlled synthetic blur experiment.

Using a **frozen, pretrained YOLO11n detector** and the **COCO128** benchmark, the pipeline evaluates detection performance across three controlled image conditions:

1. **Baseline** — original, unmodified images
2. **Degraded** — the same images after a 9×9 Gaussian blur
3. **Restored** — the blurred images after a classical sharpening operation

With the detector weights, labels, and validation settings held constant, this study isolates the effects of the image transformations under the selected experimental setup. The result is a clean ablation study of preprocessing effects on detector robustness, rather than a training exercise.

The headline question the project answers: *does putting a sharpening filter in front of a detector actually recover lost accuracy, or does it just make images look better to humans while adding noise and false positives for the model?*

---

## Key Features

- **Reproducible three-stage evaluation pipeline** — baseline, blur, and sharpening stages run through an identical `model.val()` protocol for a fair comparison.
- **Zero-training, inference-only design** — isolates preprocessing as the single independent variable; no weight updates, no data leakage.
- **Programmatic dataset generation** — blurred and sharpened COCO128 variants are created on disk with labels preserved, each with an auto-generated dataset YAML.
- **Label-preserving transforms** — image dimensions and annotation geometry are left untouched, so ground-truth bounding boxes stay valid across all stages.
- **Standard detection metrics** — Precision, Recall, mAP@0.5, and mAP@0.5:0.95 reported to four decimal places for every stage.
- **Qualitative evidence alongside quantitative** — at least three saved prediction visualisations per stage, allowing failure modes (missed small objects, duplicate boxes, low-confidence detections) to be inspected directly.
- **Environment-agnostic paths** — the notebook auto-detects Google Colab vs. a local/Jupyter/GPU-server environment and resolves dataset and output directories accordingly.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Object detection | Ultralytics YOLO11n (pretrained COCO weights) |
| Image processing | OpenCV (`cv2`), NumPy |
| Evaluation | Ultralytics validation API (mAP, Precision, Recall) |
| Dataset | COCO128 (128-image COCO subset, YOLO format) |
| Config | PyYAML |
| Visualisation | Matplotlib, Ultralytics plotting utilities |
| Environment | Google Colab / Jupyter Notebook |
| Tooling | Git, Markdown |

**Domain keywords:** Computer Vision · Object Detection · YOLO · Convolutional Neural Networks · Image Preprocessing · Gaussian Blur · Unsharp Masking · Convolution Kernels · Model Evaluation · mAP · Precision-Recall · Robustness Testing · Ablation Study · Benchmarking

---

## System Architecture / Workflow

```
                      ┌──────────────────────────┐
                      │   COCO128 (128 images)   │
                      │   images/ + labels/      │
                      └────────────┬─────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
  ┌───────────────┐      ┌──────────────────┐     ┌────────────────────┐
  │   STAGE 1     │      │     STAGE 2      │     │      STAGE 3       │
  │   Baseline    │      │  Gaussian Blur   │     │    Sharpening      │
  │   (original)  │      │  9×9 kernel      │     │  on blurred images │
  └───────┬───────┘      └────────┬─────────┘     └─────────┬──────────┘
          │                       │                         │
          │              labels copied unchanged ────────────┤
          │                       │                         │
          ▼                       ▼                         ▼
  ┌───────────────┐      ┌──────────────────┐     ┌────────────────────┐
  │ baseline.yaml │      │   blurred.yaml   │     │  sharpened.yaml    │
  └───────┬───────┘      └────────┬─────────┘     └─────────┬──────────┘
          │                       │                         │
          └───────────────────────┼─────────────────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │  Frozen pretrained YOLO11n  │
                   │   model.val()  (identical   │
                   │   settings for all stages)  │
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │  Precision · Recall         │
                   │  mAP@0.5 · mAP@0.5:0.95     │
                   │  + prediction visualisations│
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │  STAGE 4 — Comparative       │
                   │  analysis & conclusions      │
                   └─────────────────────────────┘
```

**Design principle:** the detector is the *constant*; the image pipeline is the *variable*. This is what makes the comparison scientifically meaningful.

---

## Installation and Setup

### Option A — Google Colab (recommended, zero setup)

1. Open the notebook in Colab.
2. Select **Runtime → Change runtime type → GPU** (T4 is sufficient).
3. Run cells top to bottom. Dataset download and dependency installation are handled in the setup cells.

### Option B — Local / Jupyter

```bash
# 1. Clone the repository
git clone https://github.com/g25ait2138-collab/CV-Assignment.git
cd CV-Assignment

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -U ultralytics opencv-python pyyaml matplotlib

# 4. Launch the notebook
jupyter notebook  # Open the executed G25AIT2138 notebook from your local project
```

The COCO128 dataset (~22 MB) is downloaded automatically on first run to `~/fvl_datasets/` locally, or `/content/datasets/` on Colab. Pretrained YOLO11n weights are fetched by Ultralytics on first model load.

**Requirements:** Python 3.10+, ~2 GB free disk space. A GPU is optional; runtime depends on hardware, image size and software versions.

---

## Usage Examples

**Generate a blurred copy of the dataset:**

```python
import cv2
from pathlib import Path

blurred_dir = PROJECT_ROOT / "blurred" / "images" / "train2017"
blurred_dir.mkdir(parents=True, exist_ok=True)

for img_path in sorted(SOURCE_IMAGES.glob("*.jpg")):
    image = cv2.imread(str(img_path))
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    cv2.imwrite(str(blurred_dir / img_path.name), blurred)
```

**Apply sharpening via an unsharp mask:**

```python
import numpy as np

def unsharp_mask(image, kernel_size=(0, 0), sigma=1.0, amount=1.0):
    """Sharpen by amplifying the difference between the image and its blurred copy."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
```

**Evaluate any stage with identical settings:**

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
metrics = model.val(data=str(stage_yaml), imgsz=640, batch=16,
                    conf=0.001, iou=0.7, workers=0, verbose=False)

print(f"Precision     : {metrics.box.mp:.4f}")
print(f"Recall        : {metrics.box.mr:.4f}")
print(f"mAP@0.5       : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")
```

**Save qualitative prediction examples:**

```python
model.predict(
    source=sample_images,
    save=True,
    project=str(PROJECT_ROOT / "predictions"),
    name="stage_blurred",
)
```

---

## Results

The following results were read from the **executed notebook** at `21-10-26/New/G25AIT2138_FoVL_Individual_Project_CV.ipynb` in the supplied assignment ZIP. A separate experimental variant uses different sharpening parameters and yields different scores; avoid mixing runs.

| Stage | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Original / Baseline | **0.6633** | **0.5894** | **0.6706** | **0.5026** |
| Gaussian Blur (9×9) | **0.6802** | **0.5358** | **0.6176** | **0.4434** |
| Sharpened after Blur | **0.6859** | **0.5339** | **0.6224** | **0.4499** |

### Changes in detection quality

| Comparison | Δ Precision | Δ Recall | Δ mAP@0.5 | Δ mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Blur − Baseline | +0.0169 | −0.0536 | −0.0530 | −0.0592 |
| Sharpened − Blur | +0.0057 | −0.0019 | +0.0048 | +0.0065 |
| Sharpened − Baseline | +0.0226 | −0.0555 | −0.0482 | −0.0527 |

**Main findings:**

- **Blur degraded overall detection:** mAP@0.5 dropped by 0.0530 (7.9% relative); mAP@0.5:0.95 dropped by 0.0592 (11.8% relative). Recall fell even though Precision rose. Thus not every metric deteriorated.
- **Sharpening recovered little:** after unsharp masking, mAP@0.5 increased by 0.0048 and mAP@0.5:0.95 by 0.0065 relative to blur, but Recall decreased slightly. The sharpened scores remained below baseline on both mAP measures.
- **Engineering decision:** do not deploy sharpening solely because the images look clearer. On this limited benchmark, the mAP recovery is modest; validate on production-domain images and assess latency, failure cases and precision–recall trade-offs.

**Experimental settings:** fixed pretrained `yolo11n.pt`, `imgsz=640`, validation `conf=0.001`, `iou=0.7`, `batch=16`, `workers=0`, and no training. Stage 2 uses `cv2.GaussianBlur(image, (9,9), 0)`. Stage 3 uses `cv2.addWeighted(blurred, 2.0, cv2.GaussianBlur(blurred, (0,0), sigmaX=1.0), -1.0, 0)` (an unsharp mask). These settings correspond to the selected notebook, not necessarily every alternate version.

**Qualitative samples:** matched image IDs `000000000009.jpg`, `000000000025.jpg`, `000000000030.jpg`, three per stage (nine detections in total).

**Limitations:** COCO128 is a small, nonrepresentative substitute for a held-out benchmark such as COCO val2017. The numbers above are saved outputs, not a fresh independent rerun; further tests and confidence intervals would be needed before claiming general robustness.

---

## Folder Structure

> **Suggested modular structure:** this is a proposed layout for reorganizing the project, not a verified listing of files committed to GitHub. The tested notebook is stored in the uploaded assignment ZIP under `21-10-26/New/`.

```
.
├── notebooks/
│   └── blur_sharpening_object_detection.ipynb   # Main experiment notebook
├── src/
│   ├── transforms.py                            # Blur and sharpening operations
│   ├── dataset_builder.py                       # Dataset variant + YAML generation
│   └── evaluate.py                              # Unified evaluation wrapper
├── configs/
│   ├── baseline.yaml                            # Dataset config — original images
│   ├── blurred.yaml                             # Dataset config — blurred variant
│   └── sharpened.yaml                           # Dataset config — sharpened variant
├── results/
│   ├── metrics/                                 # Per-stage metric exports (CSV/JSON)
│   └── predictions/
│       ├── baseline/                            # ≥3 prediction examples
│       ├── blurred/                             # ≥3 prediction examples
│       └── sharpened/                           # ≥3 prediction examples
├── assets/                                      # Screenshots and comparison figures
├── requirements.txt
├── LICENSE
└── README.md
```

> Datasets and model weights are downloaded at runtime and excluded from version control via `.gitignore`.

---

## Challenges Faced and Solutions Implemented

| Challenge | Solution |
|---|---|
| **Hard-coded paths broke between Colab and local runs.** The original setup assumed `/content`, which does not exist outside Colab. | Added environment detection that checks for a writable `/content` directory and otherwise falls back to a home-directory project root, making the notebook portable across Colab, Jupyter, and GPU servers. |
| **Ultralytics resolves dataset paths relative to its own config root**, causing "image not found" errors for the generated variants. | Generated each dataset YAML with explicit absolute paths, and verified image/label pairing counts before every validation run. |
| **Risk of an unfair comparison** if the model, image size, or split changed between stages. | Centralised evaluation into a single parameterised function so all three stages share identical model weights, `imgsz`, and split — the transformation is the only variable. |
| **Transformed datasets initially had images without matching labels**, silently skewing Recall. | Copied labels programmatically alongside each image write and asserted filename-level parity between the image and label directories. |
| **Naive kernel sharpening amplified JPEG artefacts** and produced clipped, saturated pixels. | Implemented unsharp masking with a tunable `amount` parameter and explicit `np.clip` to `[0, 255]` before `uint8` casting, keeping output in valid intensity range. |
| **Metric differences were easy to over-interpret** on a 128-image subset. | Paired every quantitative claim with saved prediction visualisations so conclusions are supported by observable failure cases, not just decimal shifts. |

---

## Skills Demonstrated

**Computer Vision & Deep Learning**
- Object detection fundamentals: anchor-free detection, bounding-box regression, non-maximum suppression
- Evaluation metrics: Precision, Recall, IoU thresholds, mAP@0.5 vs. mAP@0.5:0.95 and what each measures
- Model robustness and degradation analysis under distribution shift

**Image Processing**
- Convolution kernels, Gaussian filtering, frequency-domain intuition for blur
- Unsharp masking and edge enhancement, including their limitations

**Engineering Practice**
- Reproducible experiment design with a single controlled variable
- Programmatic dataset construction and YAML-driven configuration
- Cross-environment path handling and defensive validation of data integrity
- Clean, modular, documented Python using OpenCV, NumPy, and the Ultralytics API

**Analysis & Communication**
- Translating metric deltas into actionable engineering conclusions
- Pairing quantitative results with qualitative visual evidence
- Technical documentation and structured reporting

---

## Screenshots / Demo

| Baseline | Blurred | Sharpened |
|---|---|---|
| ![Baseline detections](assets/baseline_example.png) | ![Blurred detections](assets/blurred_example.png) | ![Sharpened detections](assets/sharpened_example.png) |

**Metric comparison across stages**

![Metric comparison chart](assets/metrics_comparison.png)

> To render the screenshots in GitHub, first commit the prediction images under `assets/` and update the Markdown paths to the actual committed filenames.

---

## Impact Highlights

- Designed and executed a **three-stage controlled ablation study** quantifying object-detection degradation under image blur across **128 images and 80 COCO classes**.
- Built a **reproducible, zero-training evaluation pipeline** that isolates preprocessing as the sole experimental variable, eliminating confounds from weight updates or threshold drift.
- Benchmarked a pretrained **YOLO11n** detector on four standard metrics across three dataset variants, producing evidence-backed guidance on whether sharpening belongs in a production preprocessing stack.
- Implemented **label-preserving image transformations** with automated integrity checks, preventing silent recall distortion from image/annotation mismatches.
- Delivered **portable, environment-aware code** running unchanged on Google Colab, local Jupyter, and GPU servers.

---

## Future Improvements

- **Blur severity sweep** — vary the kernel from 3×3 to 21×21 and plot metric degradation curves to find the detector's breaking point.
- **Additional degradations** — motion blur, JPEG compression, Gaussian noise, and low-light simulation, for a broader robustness profile.
- **Sharpening method comparison** — benchmark Laplacian kernels, unsharp masking, and a learned deblurring model (e.g. DeblurGAN) against each other.
- **Per-class and per-size breakdown** — quantify whether small objects degrade disproportionately versus large ones.
- **Scale to full COCO val2017** — confirm whether the 128-image findings hold on 5,000 images.
- **Model comparison** — repeat across YOLO11n/s/m and a transformer-based detector such as RT-DETR to test whether robustness is architecture-dependent.
- **Automation** — convert the notebook into a CLI pipeline with a CI workflow that regenerates metrics and plots on every commit.

---

## Contributing

Contributions are welcome.

1. Fork the repository and create a feature branch: `git checkout -b feature/your-feature`
2. Keep changes focused; follow PEP 8 and add docstrings to new functions.
3. Re-run the full notebook end-to-end and confirm metrics still reproduce before opening a PR.
4. Commit with a clear message: `git commit -m "Add motion blur degradation stage"`
5. Push and open a pull request describing the change, the motivation, and any metric impact.

For bugs or ideas, please open an issue with reproduction steps or a short proposal.

---

## License

Released under the MIT License — see [LICENSE](LICENSE) for details.

COCO128 is distributed by Ultralytics under the terms of the original COCO dataset licence. YOLO11 is provided by Ultralytics under AGPL-3.0; review their licensing terms before any commercial use.

---

## Author

**Alok Sahni**
Post Graduate Diploma in Artificial Intelligence — School of AI and Data Science, IIT Jodhpur
Broadcast Engineering, Prasar Bharati (Akashvani)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/your-handle)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/your-username)

---

<sub>If you find this project useful, consider starring the repository.</sub>
