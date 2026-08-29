# Machine Spectra 1.0 — Project Explanation

## Title
**Skin Lesion Classifier: Using CNNs to Detect Early Signs of Melanoma from Images**

## App name
**DermaSense AI**

## Problem statement
Melanoma is a serious type of skin cancer. Visual patterns in skin-lesion images can be analyzed using computer vision. This project demonstrates how a CNN-based model can classify labeled lesion images into benign-like and melanoma-suspicious categories.

## Objective
To build an educational Streamlit application that accepts a skin-lesion image and uses a trained CNN to produce a classification result and confidence score.

## Input
Skin-lesion image in JPG, JPEG, or PNG format.

## Output
- Benign-like or Melanoma-suspicious model class
- Classification confidence
- Melanoma-class model score
- Analysis history

## Algorithm
**Convolutional Neural Network with Transfer Learning — MobileNetV2**

## Why CNN?
CNNs learn visual features such as edges, textures, shapes, and more complex image patterns automatically from image data.

## Why transfer learning?
Instead of learning every visual feature from scratch, the project starts from MobileNetV2 weights learned on ImageNet and trains a new classifier for the two lesion classes. This is useful when a student project has less data and computing power than a large research project.

## Flow
```text
User uploads lesion image
        ↓
Convert to RGB
        ↓
Resize to 224 × 224
        ↓
Data preprocessing
        ↓
MobileNetV2 CNN feature extractor
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Sigmoid classifier
        ↓
Model probability
        ↓
Prediction + confidence + history
```

## Training
- 80% training data
- 20% validation data
- Data augmentation
- Class weighting for class imbalance
- Adam optimizer
- Binary cross-entropy loss
- Early stopping
- Best-model checkpointing
- Fine-tuning of upper CNN layers

## Evaluation
The training script saves:
- Accuracy
- AUC
- Precision
- Recall
- Loss
- Training/validation curves
- Confusion matrix

## What to say during demonstration
“First, I upload a skin-lesion image. The app resizes the image and sends it to the trained CNN. MobileNetV2 extracts visual features, and the final sigmoid layer generates the melanoma-class probability. The interface converts this output into a model class and confidence. The history page stores the previous model results.”

## Important limitation
This is an educational machine-learning prototype. It does not diagnose melanoma, rule out melanoma, or replace a dermatologist.

## Future improvements
- Multi-class lesion classification
- Better dataset balancing
- Cross-validation
- Calibration of probability scores
- Explainability heatmaps such as Grad-CAM
- External validation on a separate dataset
- Improved testing across different skin tones and image-acquisition conditions
