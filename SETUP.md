# SETUP: Emotional Detection Project

This document provides instructions for setting up the environment and running the Emotional Detection project.

## 1. Prerequisites

You must have the following installed on your system:

* **Git:** For cloning the repository.
* **Conda/Mamba or Python 3.11:** A Python environment manager is strongly recommended for dependency isolation. (The project was developed using Python 3.11).
* **Jupyter Lab/Notebook:** To run the main project notebook.


## 2. Environment Setup

Follow these steps to create a Python environment.

2.1: Clone and enter the repository

2.2: Necessary Python packages are listed in requirements.txt.
Recommended: 
Create and activate a new Conda/Mamba environment:
conda create -n emotion-env python=3.11
conda activate emotion-env

Install Required Packages
pip install -r requirements.txt


## 3. Model Weights and Data Setup

Step 3.1: Download Trained Model Weights

The fine-tuned ViT model weights (`best_emotion_vit.pth`) are **~330 MB** and are not included in the repository due to size limits.

1.  Download the model file from the following external link:
   https://drive.google.com/file/d/1lvQWN8ILPOwyTxVUdi-vj6BKoC6x0rYS/view?usp=drive_link
2.  Place the downloaded file directly into the **`models/`** directory.

Step 3.2: Data Preparation

The dataset is stored in `labeled_images.zip`.

1.  Ensure you have the compressed data file in your project root: `labeled_images.zip`.
2.  Unzip the file to create the necessary image directory structure:
    unzip labeled_images.zip
    *(Note: This creates the `labeled_images` directory that the notebook uses for loading the dataset.)*

