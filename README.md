DermaSense AI

Explainable and Confidence-Aware AI Skin Screening

DermaSense AI is an educational AI/ML skin-image screening prototype built with MobileNetV2 transfer learning and Grad-CAM explainability. The application classifies uploaded skin images into user-facing categories, shows model confidence, provides visual attention analysis, and includes an animated structural visualization for supported predictions.

Live Demo

Streamlit App: https://dermasense-ai.streamlit.app/

Key Features

Upload skin images or capture images using the camera

MobileNetV2 transfer-learning model

Benign-like, Melanoma-suspicious, and Other outputs

Actual model confidence scores for supported predictions

Grad-CAM model-attention visualization

Animated AI analysis / prediction process

Structural skin visualization for supported predictions

SQLite-based analysis history

Responsive Streamlit interface

Medical-safety disclaimer and responsible next-step guidance

Model

Architecture: MobileNetV2

Learning: Transfer Learning with ImageNet pretrained weights

Input: 224 × 224 RGB images

Framework: TensorFlow / Keras

Explainability: Grad-CAM

The internal model classes are mapped to the following user-facing outputs:

benign → Benign-like

melanoma → Melanoma-suspicious

non_skin + other_skin → Other

Evaluation

The model was evaluated using a fixed held-out test split.

Test images: 800

Accuracy: 81.38%

Macro Precision: 78.81%

Macro Recall: 78.67%

Macro F1 Score: 78.51%

These values come from the project's saved evaluation results and are not clinical performance claims.

Technology Stack

Python

TensorFlow / Keras

MobileNetV2

Grad-CAM

Streamlit

SQLite

Pandas

Pillow

HTML / CSS

Streamlit Community Cloud

Run Locally

Create/activate your Python environment, install the dependencies, and run:

pip install -r requirements.txt
python -m streamlit run app.py

Then open:

http://localhost:8501

Deployment

The application is deployed on Streamlit Community Cloud.

Live application: https://dermasense-ai.streamlit.app/

Medical Safety

DermaSense AI is an educational screening and decision-support prototype. It does not provide a medical diagnosis and must not replace assessment by a dermatologist or another qualified healthcare professional.

A Melanoma-suspicious output indicates a machine-learning classification response, not confirmation of melanoma. Grad-CAM highlights regions that influenced the model and must not be interpreted as cancer localization.
