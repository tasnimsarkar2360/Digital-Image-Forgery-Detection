# Digital Image Forgery Detection using DSP

EEE 312 (Digital Signal Processing I Laboratory) project, BUET, Group 3.

This MATLAB project checks whether an image has been forged and shows where the edit is likely to be.
A good forgery can't be seen by eye, but it still changes the image as a 2-D signal: its compression history,
sensor noise, frequency content and pixel correlations. The system measures those changes using DSP.

## What it does

1. **Classifies images as authentic or forged.** It extracts a 29-element feature vector from each image
   and trains three classifiers: Logistic Regression, an RBF-kernel SVM and a 300-tree Random Forest.
2. **Localizes suspicious regions without a trained model.** It combines ELA, wavelet-noise and DCT-energy
   maps into a single heatmap and thresholds it with Otsu's method.
3. **Detects copy-move forgery.** It matches the DCT signatures of overlapping 8×8 blocks at full resolution
   and marks both the copied region and its source.

## DSP features (29 in total, computed on the R, G and B channels separately)

| Family | DSP operation | Features |
|---|---|---|
| Error Level Analysis | JPEG (Q = 90) re-compression residual | 6 |
| Block-DCT / Benford | 8×8 2-D DCT, first-digit statistics | 5 |
| 2-D FFT | Band energies, spectral flatness, peaks | 6 |
| Wavelet noise | db8 DWT with LL band removed, local variance | 5 |
| Laplacian high-pass | 3×3 FIR convolution | 3 |
| CFA / demosaicing | Green-channel linear prediction error | 3 |
| JPEG blockiness | Pixel differences on the 8-pixel grid | 1 |

## Results

Tested on the RECOD.ai/LUC Scientific Image Forgery Detection dataset (5,128 images) with a stratified
80/20 split (4,103 training / 1,025 validation images):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | 5-fold CV AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.546 | 0.588 | 0.518 | 0.551 | 0.554 | 0.568 | 0.589 ± 0.018 |
| **RBF-SVM** | **0.649** | **0.648** | **0.756** | **0.698** | **0.649** | **0.620** | 0.570 ± 0.015 |
| Random Forest | 0.326 | 0.363 | 0.338 | 0.350 | 0.312 | 0.475 | 0.386 ± 0.012 |

- **DSP-only localization:** mean IoU 0.106 and mean Dice 0.173 over 550 forged validation images.
- **Copy-move detector:** finds both the pasted region and its source in copy-move examples.

The DSP features carry a real but modest signal. Exact block-DCT matching is much more precise for
copy-move forgeries.

## Files

| File | Purpose |
|---|---|
| `extract_dataset_features_and_preprocess.m` | Extracts the 29 features for every image into `output/dsp_features.xlsx` and saves DSP-stage figures |
| `train_all_models.m` | Trains and compares the three models, evaluates localization (IoU/Dice), saves checkpoints and figures |
| `infer_svm.m`, `infer_logistic_regression.m`, `infer_random_forest.m` | Classify one image, show P(forged) and the localization heatmap |
| `CopyMoveDetectionRGB.m` | Standalone block-DCT copy-move detector |

## Requirements

MATLAB with the Image Processing Toolbox, Wavelet Toolbox, and Statistics and Machine Learning Toolbox.
No Python is needed: `.npy` masks are read by the built-in reader in `train_all_models.m`.

## Usage

```matlab
% 1. Train (dataset folder must contain authentic/ and forged/)
results = train_all_models('path/to/dataset', 'checkpoint', 'path/to/train_masks');

% 2. Classify an image (put test images in a test/ folder next to the scripts)
result = infer_svm();              % choose an image from test/
result = infer_svm('image.png');   % or pass a file directly

% 3. Copy-move detection: set IMAGE_FILE in the script, then run
CopyMoveDetectionRGB
```

Results are screening aids, not proof of manipulation. Always report the model, threshold and limitations with any result.

## Team

| Name | Student ID |
|---|---|
| Safwan Zaman Chowdhury | 2206070 |
| Tasnim Sarkar | 2206084 |
| Md. Naimul Hasan Ayon | 2206085 |
| Md. Shadman Shifat | 2206092 |
| Md. Redwoan Islam | 2206097 |

## Dataset

[RECOD.ai/LUC Scientific Image Forgery Detection (Kaggle)](https://www.kaggle.com/datasets/llkh0a/recod-ailuc-scientific-image-forgery-detection)
