# DermaSense AI — Skin Lesion Classifier

A **Machine Spectra 1.0** Streamlit mini-project that demonstrates CNN-based image classification for skin lesions.

## What it does

- Upload a JPG/PNG skin-lesion image
- Runs a trained **MobileNetV2 transfer-learning CNN**
- Displays:
  - Benign-like / Melanoma-suspicious model class
  - Classification confidence
  - Melanoma-class model score
  - Model-score band
- Stores local analysis history (not the uploaded image)
- Shows model workflow, saved training curves, confusion matrix, and metadata

> **Important:** This is an educational AI prototype, not a medical diagnostic device.

---

## Project structure

```text
Skin_Lesion_Classifier_Streamlit/
│
├── app.py
├── utils.py
├── train_model.py
├── requirements.txt
├── RUN_APP.bat
├── TRAIN_MODEL.bat
├── PROJECT_EXPLANATION.md
├── MODEL_CARD.md
│
├── .streamlit/
│   └── config.toml
│
├── dataset/
│   ├── benign/
│   ├── melanoma/
│   └── DATASET_INSTRUCTIONS.txt
│
├── models/
│   └── (trained files are created here)
│
└── data/
    └── (SQLite history is created here)
```

## 1. Install

Open PowerShell or the VS Code terminal inside this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can skip activation and use your normal Python installation.

## 2. Add dataset images

Place labeled images like this:

```text
dataset/
├── benign/
│   ├── benign_001.jpg
│   ├── benign_002.jpg
│   └── ...
└── melanoma/
    ├── melanoma_001.jpg
    ├── melanoma_002.jpg
    └── ...
```

The folder names must be exactly **benign** and **melanoma**.

For a proper project, use a reputable, labeled skin-lesion dataset and follow its license/usage rules. The ISIC Challenge datasets are a common public research source:
https://challenge.isic-archive.com/data/

Do not train using random Google images.

## 3. Train the CNN

```powershell
python train_model.py
```

Or double-click:

```text
TRAIN_MODEL.bat
```

The training script:
1. Uses an 80/20 train-validation split.
2. Loads MobileNetV2 with ImageNet weights.
3. Trains a new binary classifier head.
4. Fine-tunes the top part of MobileNetV2.
5. Saves the best model according to validation AUC.
6. Creates metrics, training curves, and a confusion matrix.

Generated files:

```text
models/
├── skin_lesion_model.keras
├── metadata.json
├── metrics.json
├── training_curves.png
└── confusion_matrix.png
```

## 4. Run Streamlit

```powershell
python -m streamlit run app.py
```

Or double-click:

```text
RUN_APP.bat
```

Streamlit normally opens the app in your browser automatically.

## Optional training settings

Example:

```powershell
python train_model.py --epochs 15 --fine-tune-epochs 5 --batch-size 32
```

If your laptop has low memory:

```powershell
python train_model.py --batch-size 8
```

## Important project limitations

- A strong validation score does **not** make this a clinical diagnostic system.
- Results depend on dataset quality and labeling.
- Dermoscopic images and ordinary phone-camera images can have very different distributions.
- Class imbalance can distort accuracy.
- Skin-tone representation must be considered when discussing fairness and generalization.
- Do not tell judges or users that the model can “confirm cancer.”
- Present it as an **educational early-screening / image-classification prototype**.

## Recommended presentation wording

> “DermaSense AI demonstrates how CNN-based transfer learning can classify visual patterns in labeled skin-lesion images. It is an educational prototype and not a substitute for professional medical diagnosis.”
