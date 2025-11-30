#!/usr/bin/env python3
"""
cuisine_app_demo.py
Usage:
    python code/cuisine_app_demo.py --wav sample.wav --hubert ../features/hubert/hubert_layers.pkl --model ../models/hubert_svm_layer9.pkl --layer 9
This demo will:
 - Compute HuBERT embeddings for the input wav (uses processor + model)
 - Mean-pool and pick specified layer
 - Load classifier and predict native language
 - Map predicted language to region and cuisines (simple mapping below)
"""
import argparse
import os
import joblib
import numpy as np
import torchaudio
import torch
from transformers import Wav2Vec2Processor, HubertModel

# Simple mapping (extend as needed)
ACCENT_TO_REGION = {
    "malayalam": ("Kerala", ["Appam", "Puttu", "Avial"]),
    "punjabi": ("Punjab", ["Butter Chicken", "Amritsari Kulcha"]),
    "hindi": ("North India", ["Chole Bhature", "Dal Makhani"]),
    "tamil": ("Tamil Nadu", ["Dosa", "Idli", "Sambar"]),
    "telugu": ("Andhra Pradesh/Telangana", ["Biryani", "Gongura"]),
    "kannada": ("Karnataka", ["Bisi Bele Bath", "Mysore Pak"])
}

def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy(), target_sr

def extract_layer_vector(wav_path, model_name, layer_index, device="cpu"):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()
    audio, sr = load_audio(wav_path, target_sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        outputs = model(input_values)
    hidden_states = outputs.hidden_states
    vec = hidden_states[layer_index].mean(dim=1).squeeze(0).cpu().numpy()
    return vec

def main(args):
    if not os.path.exists(args.wav):
        raise FileNotFoundError(f"{args.wav} not found")
    # If user already precomputed hubert features and classifier trained, we can just load classifier.
    clf = joblib.load(args.model)
    # For simplicity compute layer vector live using model (works but will be slower)
    vec = extract_layer_vector(args.wav, args.model_name, args.layer, device=args.device)
    pred = clf.predict(vec.reshape(1, -1))[0]
    pred_lower = pred.lower()
    region, dishes = ACCENT_TO_REGION.get(pred_lower, ("Unknown", ["No suggestion"]))
    print("Predicted native language:", pred)
    print("Inferred region:", region)
    print("Recommended dishes:", ", ".join(dishes))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, help="Path to input wav file")
    p.add_argument("--model", required=True, help="Path to trained classifier (hubert SVM)")
    p.add_argument("--model_name", default="facebook/hubert-base-ls960", help="HuBERT model name")
    p.add_argument("--layer", type=int, default=9, help="HuBERT layer index used by classifier")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    args = p.parse_args()
    main(args)
