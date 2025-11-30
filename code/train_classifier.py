#!/usr/bin/env python3
"""
train_classifier.py
Usage:
    python code/train_classifier.py --mfcc ../features/mfcc/mfcc_features.pkl --hubert ../features/hubert/hubert_layers.pkl --out ../models
Produces: ../models/mfcc_svm.pkl and ../models/hubert_svm.pkl
"""
import os
import argparse
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prepare_hubert_matrix(hubert_X, layer_index=9):
    # hubert_X: list of lists (layers)
    X = []
    for layers in hubert_X:
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError("layer_index out of range")
        X.append(layers[layer_index])
    return np.vstack(X)

def main(args):
    os.makedirs(args.out, exist_ok=True)

    if args.mfcc:
        print("Loading MFCC features from", args.mfcc)
        mfcc_X, mfcc_y = joblib.load(args.mfcc)
        X = np.vstack([x.reshape(1, -1) for x in mfcc_X])
        y = np.array(mfcc_y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = SVC(kernel="rbf")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("MFCC SVM Accuracy:", acc)
        print(classification_report(y_test, preds))
        joblib.dump(clf, os.path.join(args.out, "mfcc_svm.pkl"))

    if args.hubert:
        print("Loading HuBERT features from", args.hubert)
        hubert_X, hubert_y = joblib.load(args.hubert)
        layer = args.hubert_layer
        X = prepare_hubert_matrix(hubert_X, layer_index=layer)
        y = np.array(hubert_y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = SVC(kernel="rbf")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"HuBERT (layer {layer}) SVM Accuracy:", acc)
        print(classification_report(y_test, preds))
        joblib.dump(clf, os.path.join(args.out, f"hubert_svm_layer{layer}.pkl"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mfcc", default="../features/mfcc/mfcc_features.pkl", help="Path to mfcc features")
    p.add_argument("--hubert", default="../features/hubert/hubert_layers.pkl", help="Path to hubert layers file")
    p.add_argument("--hubert_layer", type=int, default=9, help="HuBERT layer index to use (0-based)")
    p.add_argument("--out", default="../models", help="Folder to save models")
    args = p.parse_args()
    main(args)
