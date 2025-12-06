# ATTRIBUTION.md: Project Resources and Dependencies

This document provides detailed attribution for all code, libraries, models, and datasets used in the Emotional Detection project.

---

## 1. Datasets

| Resource Name | Source / Original URL | License | Notes |
| :--- | :--- | :--- | :--- |
| **FER2013 Dataset** | [Kaggle: Challenges in Representation Learning Facial Expression Recognition Challenge (FER2013)](https://www.kaggle.com/datasets/msambare/fer2013) | Competition-specific/Public Domain (Widely used) | Used for training and evaluating the emotional detection models. The images are stored locally in the `labeled_images/` directory. |

---

## 2. External Libraries (Python Packages)

All packages are installed via `pip` (as detailed in `requirements.txt`).

| Library Name | Purpose | Primary Citation/Source |
| :--- | :--- | :--- |
| **PyTorch** (`torch`, `torchvision`) | Core deep learning framework and computer vision utilities. | [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://pytorch.org/) |
| **timm** (`timm`) | Provides the Vision Transformer (ViT) architecture and pre-trained weights. | [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) |
| **CLIP** (Custom Install) | Used for the Zero-Shot Classification baseline experiment. | [OpenAI CLIP GitHub Repository](https://github.com/openai/CLIP) |
| **transformers** | Used for managing dependencies and potentially tokenizers/utilities for CLIP or other models. | [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) |
| **scikit-learn** (`sklearn.metrics`) | Used for computing and displaying quantitative metrics (Confusion Matrices, Classification Report). | [scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/) |
| **Seaborn/Matplotlib** | Used for data visualization and plotting the confusion matrices. | [Seaborn: Statistical data visualization](https://seaborn.pydata.org/) |
| **Tqdm** | Used for displaying progress bars during the training and evaluation loops. | [tqdm: A Fast, Extensible Progress Bar for Python](https://github.com/tqdm/tqdm) |
| **Pillow, NumPy** | Standard Python libraries for image handling and numerical operations. | [Pillow (PIL Fork)](https://python-pillow.org/) / [NumPy](https://numpy.org/) |

---

## 3. Pre-trained Models and Architectures

| Model / Architecture | Source | Notes |
| :--- | :--- | :--- |
| **Vision Transformer (ViT-Base)** | Loaded via `timm` with ImageNet pre-trained weights. | The base model for the fine-tuning experiments (with and without segmentation). |
| **FCN-ResNet50 Segmentation** | Loaded via `torchvision.models.segmentation`. | Used to generate the facial segmentation mask (Class 15: Person) for the image "cleaning" experiment. Uses weights pre-trained on COCO/Pascal VOC. |
| **CLIP** (`ViT-B/32`) | Loaded via the `clip` library. | Used for the zero-shot baseline, requiring no fine-tuning on the FER2013 dataset. |

---

## 4. AI-Generated and Helper Code

| Code Section | Description | Notes |
| :--- | :--- | :--- |
| **`models/emotion_model.py`** | Model configuration and loading script. | The functions `add_lora_to_linear` and `get_model` were consolidated from the notebook code to meet project structure requirements. |
| **`simple_seg_clean` function** | Segmentation processing logic (Cell [14]). | Custom function created to integrate the FCN-ResNet50 model, normalize inputs, apply the 'Person' mask, and return the cleaned image batch. |
| **Evaluation Functions** | Functions for training loop, validation, and generating confusion matrices. | Standard PyTorch practices adapted for the ViT/LoRA model, including early stopping logic and inference time calculation. |
