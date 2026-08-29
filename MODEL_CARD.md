# Model Card — DermaSense AI

## Intended use
Educational demonstration of binary skin-lesion image classification for an AIML mini-project.

## Not intended for
Clinical diagnosis, treatment decisions, emergency decisions, or replacing professional medical assessment.

## Classes
- benign
- melanoma

The Streamlit UI displays these as:
- Benign-like
- Melanoma-suspicious

## Architecture
MobileNetV2 transfer learning with a sigmoid binary classifier.

## Input
RGB image resized to 224 × 224.

## Output
A number between 0 and 1 interpreted by this project as the melanoma-class model score.

## Known limitations
Performance can change substantially with:
- Dataset source
- Label quality
- Class imbalance
- Skin-tone representation
- Dermoscopic vs phone-camera images
- Lighting, focus, magnification, and compression
- Data leakage between training and validation sets
- Different hospitals/devices/populations

## Safety wording
A lower model score does not rule out melanoma.
A higher model score does not confirm melanoma.
