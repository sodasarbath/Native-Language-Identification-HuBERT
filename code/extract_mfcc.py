#!/usr/bin/env python3
"""
extract_mfcc.py
Usage:
    python code/extract_mfcc.py --data ../data/adult --out ../features/mfcc --sr 16000
Produces: ../features/mfcc/mfcc_features.pkl
"""
import os
import argparse
import librosa
import numpy as np
from tqdm import tqdm
import joblib

def extract_mfcc_for_file(path, sr=16000, n_mfcc=13):
    audio, _ = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # shape (n_mfcc,)

def main(args):
    data_path = args.data
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    X, y = [], []
    speakers = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    if not speakers:
        raise FileNotFoundError(f"No subfolders (language classes) found in {data_path}")
    for lang in speakers:
        folder = os.path.join(data_path, lang)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".flac", ".mp3"))]
        for f in tqdm(files, desc=f"Extracting MFCC - {lang}"):
            path = os.path.join(folder, f)
            try:
                mfcc = extract_mfcc_for_file(path, sr=args.sr, n_mfcc=args.n_mfcc)
                X.append(mfcc)
                y.append(lang)
            except Exception as e:
                print(f"Skipping {path} due to {e}")
    save_file = os.path.join(out_dir, "mfcc_features.pkl")
    joblib.dump((X, y), save_file)
    print("Saved MFCC features to", save_file)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to data folder (subfolders per language)")
    p.add_argument("--out", default="../features/mfcc", help="Output folder for features")
    p.add_argument("--sr", type=int, default=16000, help="Sampling rate")
    p.add_argument("--n_mfcc", type=int, default=13, help="Number of MFCCs")
    args = p.parse_args()
    main(args)
