# Native Language Identification of Indian English Speakers

This repository contains the core working scripts for a Native Language Identification (NLI) system
that predicts the native language (L1) of Indian speakers from their English speech.
The system uses MFCC features and HuBERT embeddings, and includes a simple cuisine
recommendation demo based on the detected accent.

---

## Files in This Repository

- **extract_mfcc.py**  
  Extracts MFCC features from audio files.

- **extract_hubert.py**  
  Extracts HuBERT embeddings (all layers) from audio files.

- **train_classifier.py**  
  Trains SVM classification models using MFCC and HuBERT features.

- **cuisine_app_demo.py**  
  Uses the trained model to predict accent → map to region → recommend cuisine.

- **requirements.txt**  
  Contains all Python dependencies needed to run the scripts.

---

## Dataset
This project uses the **IndicAccentDB** dataset:
https://huggingface.co/datasets/DarshanaS/IndicAccentDb

The dataset is *not included* in this repository.

---

## Notes
The dataset, extracted features, and trained models are **not included** here.

---

## Authors
Abel Jacob
