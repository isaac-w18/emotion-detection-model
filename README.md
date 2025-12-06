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
Non-Technical Walkthrough: https://drive.google.com/file/d/13S7kswFVFdNxEfBcxRxpKYf1e1mYQicH/view?usp=sharing
Technical Walkthrough: https://drive.google.com/file/d/1Vy3V0svmH3IAGNZwaJwn8THuP4V9WbX_/view?usp=sharing

## Evaluation:

### Quantitative Results:
ViT Val Accuracy after 1st Epoch: 64.59%
Fine-Tuned ViT Test Accuracy (w/o Segmentation): 68.11%
<img width="652" height="581" alt="Screenshot 2025-12-06 at 3 35 46 PM" src="https://github.com/user-attachments/assets/f7293b5b-995b-4df1-81aa-f09a3a1952da" />

Fine-Tuned ViT Test Accuracy (w/ Segmentation): 68.47%
<img width="654" height="581" alt="Screenshot 2025-12-06 at 3 36 03 PM" src="https://github.com/user-attachments/assets/53696a06-fb26-4ad9-87e0-c405d8b68ffd" />

Pre-Trained CLIP Eval Accuracy (w/o Segmentation): 49.81%
<img width="662" height="617" alt="Screenshot 2025-12-06 at 3 36 23 PM" src="https://github.com/user-attachments/assets/2b39ad1b-5628-4d88-b201-d164b4b71026" />

Pre-Trained CLIP Eval Accuracy (w/ Segmentation): 36.99%
<img width="667" height="612" alt="Screenshot 2025-12-06 at 3 36 40 PM" src="https://github.com/user-attachments/assets/7f2dd98c-9c25-4930-b346-d1b1f3e500a2" />

### Qualitative Assessment:
Image segmentation made a very slight positive difference in evaluation accuracy of the fine-tuned ViT. It made a large negative difference in pre-trained CLIP, likely because the images CLIP was trained on were not segmented.

My next exploration question is to consider fine-tuning a ViT on already segmented images, in addition to segmenting the images at inference time, to see whether this would be beneficial to increasing accuracy rate.

