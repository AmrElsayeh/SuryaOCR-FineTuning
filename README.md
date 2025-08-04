# Arabic OCR Fine-Tuning with SuryaOCR

This repository contains a Jupyter notebook for fine-tuning the **SuryaOCR** model on a custom Arabic dataset. The goal is to improve Arabic text recognition performance using the encoder-decoder architecture provided by SuryaOCR.

## 🚀 Project Overview

SuryaOCR is a modular OCR framework built with PyTorch and Hugging Face Transformers. This notebook demonstrates:

- Loading and preprocessing a custom Arabic dataset
- Tokenizing labels using a processor
- Fine-tuning the encoder-decoder model on text line images
- Evaluating model performance on a validation set
- Saving the best-performing checkpoint

## 📁 Repository Structure


## 🧠 Model Architecture

- **Encoder:** DonutSwinModel (Swin Transformer)
- **Decoder:** Transformer decoder (auto-regressive)
- **Text Tokenizer:** Hugging Face tokenizer embedded in SuryaProcessor

The architecture supports mixed precision training and teacher forcing with shifted label sequences.

## 🧪 Training Details

- **Dataset:** Arabic text line images (manually prepared)
- **Loss Function:** CrossEntropy with ignore_index for padding
- **Metrics:** Accuracy on non-padding tokens
- **Epochs:** Configurable in the notebook
- **Batch Size:** Dynamic (depends on hardware)

> Note: Notebook includes logic for label filtering and early stopping based on accuracy.

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Transformers
- Hugging Face Datasets (optional)
- PIL
- tqdm
- SuryaOCR (custom modules)

Install dependencies:

```bash
pip install torch torchvision transformers pillow tqdm

git clone https://github.com/AmrElsayeh/SuryaOCR-FineTuning.git
cd SuryaOCR-FineTuning

jupyter notebook "surya-ocr-fine-tuning V5.ipynb"

