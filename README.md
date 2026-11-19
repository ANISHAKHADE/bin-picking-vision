# Bin-Picking Perception — One-Day Build

**Purpose of this repo:** demonstrate, with real data and measured results, the first perception
sub-step of a robot bin-picking pipeline — detecting and localizing small objects in a cluttered
bin from a single RGB image — before pose estimation and pick planning. Built in ~1 day to show
evidence against a specific job's requirements (CNN/detection fundamentals, data annotation,
train/val/test discipline), starting from zero prior CV/DL experience.

## What this IS
- A real, self-collected image dataset (not stock photos, not synthetic).
- A pretrained-baseline vs. fine-tuned-model comparison on that dataset, with real metrics.
- An honest error analysis of where the fine-tuned model fails.
- A bonus OpenCV camera calibration exercise (intrinsics + undistortion).

## What this IS NOT (on purpose — don't let anyone, including future-me, overclaim this)
- Not pose estimation. Bounding-box centers are **not** 6D pose.
- Not a bin-picking system — there's no robot, no depth camera, no grasp planning.
- Not ROS-integrated.
- Not trained on enough data to generalize beyond this specific bin/lighting setup.

---

## Folder guide

```
bin-picking-vision/
├── data/
│   ├── raw_images/        # phone photos, before annotation (Step 2)
│   └── dataset/           # Roboflow export goes here: train/valid/test + data.yaml (Step 3)
├── calibration/
│   ├── images/             # checkerboard photos (bonus step)
│   └── calibrate.py        # run this after capturing checkerboard images
├── train_and_eval.ipynb    # main notebook: baseline -> fine-tune -> eval -> error analysis
├── results/
│   ├── baseline_predictions/    # pretrained model's predictions on raw images
│   ├── finetuned_predictions/   # fine-tuned model's predictions on test split
│   └── calibration/             # camera matrix + undistorted sample image
├── requirements.txt
└── .gitignore
```

## Step-by-step (time-boxed — see the plan doc for the full breakdown)

1. **Pick 3 small object classes** you own (default: pen, eraser, USB drive — swap in whatever
   you have). Scatter them in a box/tray ("the bin"). Don't overthink it.
2. **Shoot ~90 real photos** into `data/raw_images/`. Vary: angle, lighting (lamp on/off,
   near window), occlusion (objects overlapping), and clutter density (sparse vs. piled).
3. **Annotate in Roboflow** (free tier, web UI): create an Object Detection project, upload
   `data/raw_images/`. Use the **Smart Polygon** tool (SAM-powered, class-agnostic — click once
   inside an object and it draws a tight box/outline around it instantly, works on any object
   even though it's not a COCO class) instead of dragging rectangles by hand: click the object,
   hit the class hotkey, move on. This is the time-saver — roughly 3-5x faster than manual boxing.
   Let Roboflow split train/valid/test (~70/15/15), export in **YOLOv8 format**, unzip into
   `data/dataset/`. You should end up with `data/dataset/data.yaml` plus `train/`, `valid/`,
   `test/` folders.
4. **Open `train_and_eval.ipynb`** (locally in VS Code with the Jupyter extension, or upload to
   Google Colab for a free GPU — recommended for the fine-tune step) and run cells top to bottom:
   - Baseline: pretrained YOLO on your raw images (expect it to mostly fail — your objects
     aren't COCO classes — that's the honest, useful "before").
   - Fine-tune: transfer-learn a YOLO-nano model on `data/dataset/`.
   - Evaluate: precision/recall/mAP@0.5 on the **test** split (never seen during training).
   - Error analysis: pull real failure images into `results/finetuned_predictions/`.
5. **(Bonus, only if time remains)** Print a checkerboard pattern, take 15–20 phone photos of it
   from different angles into `calibration/images/`, run `python calibration/calibrate.py`.
6. **Fill in the Results section below** with your real numbers — not estimates.

---

## Results

**Baseline (pretrained COCO YOLOv8n, no fine-tuning) — run on all 106 raw images:**
- 77/106 images (73%): zero detections at all.
- 29/106 images (27%): detected *something*, but always mislabeled as an
  unrelated COCO class (scissors, airplane, laptop, clock, chair, bed,
  refrigerator, suitcase, toilet, etc.) — never correctly identified as
  pen/eraser/hole-punch, since those aren't COCO classes.
- Takeaway: the pretrained model has no meaningful signal on these objects
  before fine-tuning. This is the expected, honest "before" picture.

**Fine-tuned model (YOLOv8n, transfer-learned):**
- Training set: 73 images. Training auto-stopped at epoch 60 via early
  stopping (patience=20); best checkpoint selected from epoch 40.
- Held-out **test** split (10 images / 29 object instances — never seen
  during training or for checkpoint selection):
  - Precision: 0.990
  - Recall: 1.000
  - mAP@0.5: 0.995
  - mAP@0.5:0.95: 0.890
- Per-class mAP@0.5: Eraser 0.995, Pen 0.995, Punch 0.995 (per-class
  instance counts: Eraser 9, Pen 10, Punch 10 — one test image has no
  eraser in frame, which is why Eraser instance count is 9, not 10).

**Honest caveat on the numbers above:** the test split is only 10 images.
Near-perfect scores at this scale reflect a controlled, low-clutter setup
with 3 easily-distinguishable object classes on a plain background — not a
claim of robustness at scale or under harder conditions.

**Failure modes observed:**
1. No missed or misclassified detections on the 10-image test split itself
   — every object present was correctly detected and classified.
2. During annotation (before training), Roboflow's AI-assisted auto-labeler
   confused the hole-punch and eraser classes on 3 of 106 images — caught
   and corrected manually before training. This suggests the two classes
   are visually similar at a glance (similar size/color against this
   background), a risk that would likely resurface with more classes or
   more cluttered scenes.
3. Detection confidence drops (0.59–0.86, vs. 0.9+ typical) on frames with
   more occlusion/overlap between objects — still correct in this dataset,
   but this is the most likely place performance would degrade on a
   larger/harder dataset.

**What I'd do with more time:**
- Segmentation masks for tighter pick regions
- Depth camera + PnP for actual 6D pose
- Wrap the detector in a ROS node
- Test under real warehouse-style lighting/clutter at scale
- Expand the test set — 10 images is too small to trust the metrics above
  as a general performance claim
- Retake calibration photos with the board deliberately swept across all
  regions of the frame (corners/edges, not just tilted near-center), which
  should meaningfully reduce the reprojection error below

**Camera calibration (bonus):**
- Reprojection error: 9.40 px — notably higher than the ~0.5px that's
  typical for a well-conditioned phone-camera calibration.
- Honest root cause: across the 20 calibration photos, the checkerboard was
  tilted at many angles but stayed roughly centered in the frame rather
  than sweeping across different regions of the image (corners, edges,
  varying depth) — calibration math needs that spatial coverage to solve
  focal length and principal point reliably. A slight curl in the printed
  board likely compounded this. The large distortion coefficients this
  produced (k2=-1.69, k3=4.21) are a symptom of this poor conditioning,
  not a realistic property of the phone's actual lens.
- Camera matrix and distortion coefficients: see
  `results/calibration/camera_matrix.txt` and `dist_coeffs.txt`.
- This is reported as-is rather than re-shot and cherry-picked, in keeping
  with this project's goal of honest, non-overclaimed results.
