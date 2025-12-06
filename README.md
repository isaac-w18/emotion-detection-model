# Segmentation for Emotional Detection
I used PyTorch to fine-tune a pre-trained ViT to label human faces with one of seven emotions. Then, I used person image segmentation to improve the accuracy rate, beating OpenAI's pre-trained CLIP model by a 37.5% improvement in accuracy rate.

## What it Does

Non-Technical Overview: My project essentially takes images of human faces, "cuts out" the parts of the image not showing a person, and then predicts which emotion the facce is expressing. Then, I compare the results of the predictions of images that "cut out" non-person parts and the predictions of images that did cut out person parts to see if the cutting out of non-person parts was helpful. I also compared this with an existing popular model to see if it helped.

Technical Overview: 
The goal of my project was to evaluate the effectiveness of image segmentation on emotion detection in human faces.

Starting with a pre-trained Vision Transformer and 48x48 pixel faces that are labeled with 7 emotions: happy, angry, sad, surprised, neutral, disgust, fear. It then uses Low-Rank Adaptation to pre-train the Transformer on the dataset. To evaluate the effectiveness of person image segmentation, in the test round, I used the same test dataset on segmented and non-segmented images and displayed the results. I also evaluated CLIP on the data with segmented and non-segmented images, but segmentation reduced the accuracy rate of CLIP.

## Quick Start
The project evaluation pipeline is contained entirely within the main Jupyter Notebook. This section assumes you have already completed the setup steps detailed in the SETUP.md file (cloning the repo, installing requirements.txt, downloading model weights, and unzipping data).

Activate Environment:
conda activate emotion-env

Launch Jupyter:
jupyter lab

Run Notebook:
Open Emotional Detection.ipynb.

To run the entire evaluation (ViT with LoRA, Segmentation experiment, and CLIP Zero-Shot baseline), select Run → Run All Cells from the menu.

View Results:
The model will load the weights from the models/ directory.

The final quantitative results (accuracy scores, confusion matrices, and inference times) are reported in the notebook.

## Video Links:
a Video Links section with direct links to your demo and technical walkthrough videos,

## Evaluation:

an Evaluation section that presents any quantitative results, accuracy metrics, or qualitative outcomes from testing,

