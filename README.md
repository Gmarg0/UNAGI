# UNAGI
Utilizing NLP for Attribution of Grouped Individualized dialogues

This repository contains code for training and using a character classifier for TV shows like Friends and Seinfeld using transformer-based models.

## Overview

The project includes scripts for:
1. Preprocessing TV show scripts
2. Training character classification models
3. Making predictions using the trained models

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/Gmarg0/UNAGI.git
cd tv-show-character-classifier
pip install -r requirements.txt

## Preparation

Place your TV show scripts in the data_files directory.
Run the script_processor.py to preprocess the scripts.

## Training
Run the main.py script to train the model:
python main.py
This script will:

## Load and preprocess the data
Train a transformer model (DistilBERT, BERT, or RoBERTa)
Save the trained model

## Prediction
Use the predict.py script to make predictions:
python predict.py

## Models
Two fine-tuned BERT models are available:

CentralBERT: Trained on Friends episodes - https://huggingface.co/GalMargo/centralBert
YadaYadaBERT: Trained on Seinfeld episodes - https://huggingface.co/GalMargo/YadaYadaBERT

## Additional Scripts

script_downloader.py: Downloads TV show scripts from a specified website

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
