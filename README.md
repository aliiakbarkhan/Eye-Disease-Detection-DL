<div align="center">

<img src="assets/device-front-and-screen.png" alt="Handheld fundus camera concept, front and rear views" width="100%">

# Eye Disease Detection using Deep Learning

Classification of retinal fundus images into four categories (cataract, diabetic retinopathy, glaucoma, normal) using a fine-tuned ResNet18.

![Python](https://img.shields.io/badge/Python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![fastai](https://img.shields.io/badge/fastai-00A3E0?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=flat-square)
![Accuracy](https://img.shields.io/badge/Validation%20accuracy-92.2%25-brightgreen?style=flat-square)

</div>

---

## Overview

Cataract, glaucoma and diabetic retinopathy are major causes of preventable vision loss, and screening typically relies on specialists reviewing retinal photographs. This project trains a convolutional neural network on labelled fundus images to predict the most likely condition and return a probability for each class. A Gradio app (`eye_app.py`) is included for interactive inference.

> **Disclaimer:** This is a research and learning project, not a medical device. Its output must not be used for diagnosis or treatment decisions.

## Dataset

Labelled retinal images organised in one folder per class, included in this repository as `dataset/` and `dataset.zip`.

| Class | Description |
|---|---|
| `cataract` | Clouding of the eye lens |
| `diabetic_retinopathy` | Retinal damage caused by diabetes |
| `glaucoma` | Optic nerve damage |
| `normal` | No disease |

<div align="center">
<img src="assets/dataset-samples.png" alt="Sample batch of labelled retinal images" width="75%">
<br><sub>Random training batch (resized to 192 x 192). Image quality, field of view and colour vary considerably.</sub>
</div>

<!-- TODO: add dataset source and licence. -->

## Method

| Component | Setting |
|---|---|
| Pre-processing | Images resized to max side 400 px; corrupt files removed with `verify_images` |
| Input size | 192 x 192 (squish resize) |
| Split | 80% train / 20% validation (843 images), `RandomSplitter(seed=42)` |
| Batch size | 32 |
| Architecture | ResNet18, ImageNet-pretrained (`vision_learner`) |
| Training | `fine_tune(10)`: 1 frozen epoch followed by 10 unfrozen epochs |
| Metric | Error rate |

The complete pipeline is in [`eye-desease.ipynb`](eye-desease.ipynb).

## Results

The final model reaches a validation error rate of **7.83%** (66 of 843 images misclassified), corresponding to **92.2% accuracy**.

| Epoch | Train loss | Valid loss | Error rate |
|:---:|:---:|:---:|:---:|
| 0 | 0.4514 | 0.3727 | 0.1269 |
| 1 | 0.2516 | 0.3835 | 0.1234 |
| 2 | 0.2383 | 0.4484 | 0.1269 |
| 3 | 0.1532 | 0.3512 | 0.0949 |
| 4 | 0.0928 | 0.3451 | 0.0937 |
| 5 | 0.0763 | 0.4029 | 0.0961 |
| 6 | 0.0565 | 0.3475 | 0.0866 |
| 7 | 0.0294 | 0.3547 | 0.0795 |
| 8 | 0.0170 | 0.3388 | 0.0807 |
| 9 | 0.0100 | 0.3380 | 0.0783 |

Training loss keeps decreasing while validation loss plateaus near 0.34, which suggests mild overfitting. Only overall error rate is reported; per-class metrics have not yet been evaluated.

## Usage

**Install**

```bash
git clone https://github.com/aliiakbarkhan/Eye-Disease-Detection-DL.git
cd Eye-Disease-Detection-DL
pip install -r requirements.txt
```

**Run the Gradio app**

```bash
python eye_app.py
```

**Predict from Python**

```python
from fastai.vision.all import *

learn = load_learner("eye_disease_model.pkl")
pred, idx, probs = learn.predict("path/to/retinal_image.jpg")

for label, p in zip(learn.dls.vocab, probs):
    print(f"{label}: {p:.4f}")
```

**Retrain**: open `eye-desease.ipynb` in Jupyter or Kaggle (GPU recommended) and run all cells. Update the test-image path in the prediction cell to a file on your machine.

## Portable Screening Concept

The project is motivated by point-of-care screening: a handheld fundus camera that captures the retina and runs a classifier on the device. The renders below are concept illustrations only, not an existing product.

<div align="center">
<img src="assets/device-side-views.png" alt="Concept render of a handheld fundus camera, angled and side views" width="100%">
<br><br>
<img src="assets/device-multi-angle.png" alt="Concept render of the handheld device from multiple angles" width="100%">
</div>

## Model Card

Intended use, training data, evaluation and limitations are documented in [MODEL_CARD.md](MODEL_CARD.md).

## Limitations

- Not clinically validated; performance on other cameras, hospitals or populations is unknown.
- Evaluated on a single random split with no external test set.
- Supports only the four listed classes and will always return one of them, even for non-retinal images.
- Reliability may drop on low-quality or unusual captures.

## Repository Structure

```text
Eye-Disease-Detection-DL/
├── assets/                  # README images
├── dataset/                 # Labelled retinal images (one folder per class)
├── dataset.zip
├── eye-desease.ipynb        # Training and evaluation notebook
├── eye_app.py               # Gradio inference app
├── eye_disease_model.pkl    # Exported fastai model
├── MODEL_CARD.md
├── requirements.txt
└── README.md
```

## Author

**Ali Akbar Khan** — [GitHub](https://github.com/aliiakbarkhan)
