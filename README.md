<div align="center">

<img src="images/1.png" alt="Handheld fundus camera concept, front and rear views" width="100%">

# Eye Disease Detection using Deep Learning

Classification of retinal fundus images into four categories (cataract, diabetic retinopathy, glaucoma, normal) using a fine-tuned ResNet34.

![Python](https://img.shields.io/badge/Python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![fastai](https://img.shields.io/badge/fastai-00A3E0?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=flat-square)
![Test accuracy](https://img.shields.io/badge/Test%20accuracy-91.6%25-brightgreen?style=flat-square)
![Macro F1](https://img.shields.io/badge/Macro%20F1-0.92-brightgreen?style=flat-square)

</div>

---

## Overview

Cataract, glaucoma and diabetic retinopathy are major causes of preventable vision loss, and screening typically relies on specialists reviewing retinal photographs. This project trains a convolutional neural network on labelled fundus images to predict the most likely condition and return a probability for each class. A Gradio app (`app.py`) is included for interactive inference.

> **Disclaimer:** This is a research and learning project, not a medical device. Its output must not be used for diagnosis or treatment decisions.

## Impact

Today, screening usually means travelling to a hospital and waiting for a specialist. The goal of this project is to move the first check closer to the patient.

<div align="center">

| Today: the hospital queue | With our device: screening at home |
|:---:|:---:|
| <img src="images/impact-before.png" alt="Crowd of people waiting in a hospital" width="100%"> | <img src="images/impact-after.png" alt="A son screening his mother's eyes using the handheld device" width="100%"> |

</div>

**What's happening in these images**

- **Left:** Large crowds wait for hours in hospital eye departments, where specialists are limited and many people come only after their vision has already worsened.
- **Left:** Early-stage disease often has no obvious symptoms, so it is easy to miss until the damage is advanced.
- **Right:** A family member captures a retinal image with the handheld device, with no appointment or travel needed.
- **Right:** The on-device model returns a likely class and probability within moments, so only people flagged as higher risk need to visit a specialist.
- **Right:** Elderly people, who are most at risk, can be checked comfortably at home with a relative's help.

> These images show the intended use case. The device is a concept and the model has not been clinically validated (see [Limitations](#limitations)), so it supports early awareness and referral, not diagnosis.

## Cause

Eye disease is rising mainly because the world's population is growing and ageing, and because diabetes and other systemic conditions are becoming more common. The figures below come from published studies and articles.

<div align="center">

<img src="images/eye_disease_rise.png" alt="Bar charts of projected rise in vision impairment worldwide and from age-related macular degeneration" width="90%">
<br><sub>Chart built from the published figures cited below. Only the reported endpoints are plotted; nothing is interpolated.</sub>

</div>

**Key findings**

- **Global vision impairment:** The Lancet Global Health Commission reports 1.1 billion people with distance vision impairment or uncorrected presbyopia in 2020, rising to 1.8 billion by 2050. It also projects about 895 million people with distance vision impairment by 2050, of whom 61 million would be blind. Population ageing, growth and urbanisation drive this, and most affected people live in low- and middle-income countries.
- **Preventable in most cases:** The same Commission estimates that over 90% of vision loss has a preventable or treatable cause, which is why early screening matters.
- **Blindness projections:** An earlier Vision Loss Expert Group analysis projected blindness rising from about 36 million to 115 million by 2050. The Commission's later estimate (61 million blind in 2050) is lower, so projections vary with the model and data used.
- **Macular degeneration:** A Global Burden of Disease analysis estimates that about 8 million people had vision impairment from age-related macular degeneration in 2021, rising to roughly 21.34 million by 2050.
- **Share vs. number of people:** The proportion of the world population with visual impairment fell from 4.58% in 1990 to 3.38% in 2015, but the total number of people affected is expected to grow because the population is ageing.
- **Glaucoma:** *The Ophthalmologist* (March 2026) attributes rising glaucoma prevalence to longer lifespans and better detection of previously missed cases, along with diabetes, hypertension and vascular disease.
- **Myopia:** Research reported by ScienceDaily (Feb 2026) from SUNY College of Optometry suggests dim indoor light, not only screen time, may be fuelling the rise in nearsightedness.
- **Diabetic eye disease:** The American Academy of Ophthalmology describes diabetic eye disease as one of the most common causes of preventable blindness.

**Sources**

| # | Source | Link |
|:-:|---|---|
| 1 | Burton et al., *The Lancet Global Health Commission on Global Eye Health: vision beyond 2020* (2021) | [Full text](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7966694/) |
| 2 | Michigan Medicine, news release on the Lancet Commission | [Link](https://www.michiganmedicine.org/news-release/lancet-global-health-vision-loss-could-be-treated-one-billion-people-worldwide) |
| 3 | Bourne et al., *Magnitude, temporal trends, and projections of the global prevalence of blindness and vision impairment*, Lancet Global Health (2017) | [Full text](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(17)30293-0/fulltext) |
| 4 | SEE International, *Global Blindness Projected to Triple by 2050* (summary of the 2017 study) | [Link](https://www.seeintl.org/blog/global-blindness-2050/) |
| 5 | *Global burden of vision impairment due to age-related macular degeneration, 1990–2021, with forecasts to 2050*, Lancet Global Health | [Full text](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(25)00143-3/fulltext) |
| 6 | *Why Glaucoma Prevalence Is Rising*, The Ophthalmologist (March 2026) | [Link](https://theophthalmologist.com/issues/2026/articles/march/why-glaucoma-prevalence-is-rising) |
| 7 | ScienceDaily, Eye Care News (myopia and indoor light, Feb 2026) | [Link](https://www.sciencedaily.com/news/health_medicine/eye_care/) |
| 8 | American Academy of Ophthalmology, *11 Things Gen-Z and Millennials Can Do Now to Avoid Blindness Later* (Aug 2025) | [Link](https://www.aao.org/eye-health/tips-prevention/healthy-lifestyle-now-good-vision-later) |

> The numbers above are model-based projections, not counts, and they depend on population trends and how eye-care services develop.

## Dataset

The model is trained on the [Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) dataset from Kaggle: 4,217 labelled retinal images in one folder per class, with no corrupt files found. It is also included in this repository as `dataset/` and `dataset.zip`. Refer to the Kaggle page for the dataset's licence and provenance.

<div align="center">

| Class | Description | Images |
|---|---|:---:|
| `cataract` | Clouding of the eye lens | 1,038 |
| `diabetic_retinopathy` | Retinal damage caused by diabetes | 1,098 |
| `glaucoma` | Optic nerve damage | 1,007 |
| `normal` | No disease | 1,074 |

<img src="output.png" alt="Sample of labelled retinal images from the dataset" width="75%">
<br><sub>Random sample of labelled images. Image quality, field of view and colour vary considerably. The classes are well balanced, so no re-weighting was applied.</sub>


</div>


## Method

The complete pipeline is in [`eye-desease.ipynb`](eye-desease.ipynb) (written for Kaggle with a GPU).

<div align="center">

| Component | Setting |
|---|---|
| Pre-processing | Images resized to a max side of 460 px on a copy of the dataset |
| Split | Stratified 70 / 15 / 15: **2,951 train, 633 validation, 633 test** (seed 42) |
| Test set | Separated before any training and never used for tuning or model selection |
| Input | `RandomResizedCrop` to 224 x 224 (min scale 0.75) |
| Augmentation | Rotation up to 15°, zoom up to 1.1, lighting 0.2, warp 0.1 |
| Architecture | ResNet34, ImageNet-pretrained (`vision_learner`) |
| Learning rate | `lr_find` valley, 1.4e-3 |
| Training | `fine_tune(15)` with early stopping (patience 3 on validation loss) and best-checkpoint saving |
| Batch size | 32 |
| Metrics | Accuracy, error rate, macro precision / recall / F1 |

</div>

Training stopped early after 14 unfrozen epochs. The best checkpoint (epoch 10, lowest validation loss) was restored and exported as `eye_disease_model.pkl`.

## Results

All headline numbers below are measured on the **held-out test set** (633 images).

**Overall: 91.6% accuracy (580 of 633 correct), macro F1 0.92.**

<div align="center">

| Class | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| cataract | 0.97 | 0.92 | 0.94 | 156 |
| diabetic_retinopathy | 0.97 | 0.99 | 0.98 | 165 |
| glaucoma | 0.85 | 0.87 | 0.86 | 151 |
| normal | 0.88 | 0.89 | 0.88 | 161 |
| **Macro average** | **0.92** | **0.92** | **0.92** | 633 |

</div>

**Confusion matrix** (rows are the true class, columns the predicted class):

<div align="center">

| True \ Predicted | cataract | diabetic_retinopathy | glaucoma | normal |
|---|:---:|:---:|:---:|:---:|
| **cataract** | 143 | 0 | 8 | 5 |
| **diabetic_retinopathy** | 0 | 163 | 2 | 0 |
| **glaucoma** | 4 | 1 | 131 | 15 |
| **normal** | 1 | 4 | 13 | 143 |

</div>

- Diabetic retinopathy is the easiest class to identify (F1 0.98).
- Glaucoma is the hardest (F1 0.86). Most errors are between glaucoma and normal: 15 glaucoma images were predicted as normal and 13 normal images as glaucoma.
- 8 cataract images were predicted as glaucoma.

**Validation and training.** At the selected checkpoint the validation set (633 images) gave 94.8% accuracy and macro F1 0.947. This set was used for early stopping and checkpoint selection, so it is optimistic; the test-set figures above are the reference.

<details>
<summary>Training log (unfrozen epochs)</summary>

<br>

One frozen epoch (classifier head only) preceded these, ending at 80.9% validation accuracy. The best epoch is in bold.

| Epoch | Train loss | Valid loss | Accuracy | Macro F1 |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.7880 | 0.4249 | 0.8310 | 0.8303 |
| 1 | 0.6459 | 0.3310 | 0.8705 | 0.8704 |
| 2 | 0.5667 | 0.3427 | 0.8768 | 0.8718 |
| 3 | 0.4757 | 0.2635 | 0.9084 | 0.9061 |
| 4 | 0.4013 | 0.3444 | 0.8910 | 0.8903 |
| 5 | 0.3668 | 0.2410 | 0.9115 | 0.9113 |
| 6 | 0.3186 | 0.2287 | 0.9194 | 0.9187 |
| 7 | 0.2428 | 0.2120 | 0.9163 | 0.9152 |
| 8 | 0.2117 | 0.2281 | 0.9242 | 0.9234 |
| 9 | 0.1877 | 0.2543 | 0.9273 | 0.9267 |
| **10** | **0.1472** | **0.2058** | **0.9479** | **0.9472** |
| 11 | 0.1142 | 0.2487 | 0.9352 | 0.9333 |
| 12 | 0.1032 | 0.2226 | 0.9415 | 0.9406 |
| 13 | 0.0843 | 0.2168 | 0.9400 | 0.9388 |

</details>

## Usage

**Install**

```bash
git clone https://github.com/aliiakbarkhan/Eye-Disease-Detection-DL.git
cd Eye-Disease-Detection-DL
pip install -r requirements.txt
```

**Run the Gradio app**

```bash
python app.py
```

**Predict from Python**

```python
from fastai.vision.all import *

learn = load_learner("eye_disease_model.pkl")
pred, idx, probs = learn.predict("path/to/retinal_image.jpg")

for label, p in zip(learn.dls.vocab, probs):
    print(f"{label}: {p:.4f}")
```

**Retrain**: the notebook reads the dataset from `/kaggle/input` and writes to `/kaggle/working`, so it runs directly on Kaggle with a GPU and the dataset attached. To run it locally, change the `SRC`, `WORK` and `TEST_DIR` paths. It also writes `model_card.json` with the split sizes and test metrics.

## Portable Screening Concept

The project is motivated by point-of-care screening: a handheld fundus camera that captures the retina and runs a classifier on the device. The renders below are concept illustrations only, not an existing product.

<div align="center">
<img src="images/2.png" alt="Concept render of a handheld fundus camera, angled and side views" width="100%">
<br><br>
<img src="images/3.png" alt="Concept render of the handheld device from multiple angles" width="100%">
</div>

## Limitations

- Not clinically validated; performance on other cameras, hospitals or populations is unknown.
- Evaluated on one random split of a single public dataset, with no external test set. On 633 test images the 95% confidence interval on accuracy is roughly ±2 percentage points.
- The split is per image, not per patient. If the dataset contains several images of the same patient, some leakage between splits is possible and the test score could be optimistic.
- Glaucoma and normal are frequently confused, which matters for screening because a missed disease is costlier than a false alarm.
- Supports only the four listed classes and will always return one of them, even for non-retinal images.

## Repository Structure

```text
Eye-Disease-Detection-DL/
├── images/                  # README images
├── dataset/                 # Labelled retinal images (one folder per class)
├── dataset.zip
├── eye-desease.ipynb        # Training and evaluation notebook
├── app.py                   # Gradio inference app
├── eye_disease_model.pkl    # Exported fastai model (ResNet34)
├── model_card.json          # Split sizes and test-set metrics
├── requirements.txt
├── output.png
└── README.md
```

## Author

**Ali Akbar Khan** — [GitHub](https://github.com/aliiakbarkhan)
