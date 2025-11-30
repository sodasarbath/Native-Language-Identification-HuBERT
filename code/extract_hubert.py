#!/usr/bin/env python3
"""
extract_hubert.py
Usage:
    python code/extract_hubert.py --data ../data/adult --out ../features/hubert --model facebook/hubert-base-ls960 --device cpu
Produces: ../features/hubert/hubert_layers.pkl
"""
import os
import argparse
import joblib
import torch
import torchaudio
from transformers import Wav2Vec2Processor, HubertModel
from tqdm import tqdm

def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.squeeze(0).numpy()
    return waveform, target_sr

def main(args):
    data_path = args.data
    out_dir = args.out
    model_name = args.model
    device = args.device
    os.makedirs(out_dir, exist_ok=True)

    print("Loading processor & model:", model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    X, y = [], []
    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    if not classes:
        raise FileNotFoundError(f"No subfolders (language classes) found in {data_path}")

    for lang in classes:
        folder = os.path.join(data_path, lang)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".flac", ".mp3"))]
        for f in tqdm(files, desc=f"HuBERT extract - {lang}"):
            path = os.path.join(folder, f)
            try:
                audio, sr = load_audio(path, target_sr=args.sr)
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                with torch.no_grad():
                    outputs = model(input_values)
                # outputs.hidden_states is a tuple (layer0, layer1, ...). Each is (batch, seq_len, dim)
                hidden_states = outputs.hidden_states
                # mean pool over time dimension for each layer
                layer_vectors = [hs.mean(dim=1).squeeze(0).cpu().numpy() for hs in hidden_states]
                X.append(layer_vectors)
                y.append(lang)
            except Exception as e:
                print(f"Skip {path}: {e}")
    save_file = os.path.join(out_dir, "hubert_layers.pkl")
    joblib.dump((X, y), save_file)
    print("Saved HuBERT layer features to", save_file)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to data folder (subfolders per language)")
    p.add_argument("--out", default="../features/hubert", help="Output folder")
    p.add_argument("--model", default="facebook/hubert-base-ls960", help="HuBERT model name")
    p.add_argument("--sr", type=int, default=16000, help="Target sampling rate")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    args = p.parse_args()
    main(args)
