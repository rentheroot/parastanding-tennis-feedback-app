# train_model.py
"""
Train a simple MLP swing detector.

Input: FINAL_CSV (a CSV with columns:
    video,timestamp,landmark,x,y,z,visibility,label
    - produces one flattened feature vector per (video,timestamp)
    - label: 1 = Swinging, 0 = Not

Outputs:
 - model_pipeline.pkl  (scaler + MLPClassifier)
 - model_meta.json     (landmarks order used)
"""

import json
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

# ---------------- USER VARIABLES ----------------
FINAL_CSV = "final_training_dataset.csv"   # <-- set this to your CSV
OUT_MODEL = "model_pipeline.pkl"
OUT_META = "model_meta.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# MLP hyperparams (you can tune)
MLP_HIDDEN = (128, 64)
MLP_MAX_ITER = 300
# ------------------------------------------------

def load_and_pivot(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure correct columns
    required = {"video","frame_idx","landmark","x","y","z","visibility","label"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"CSV missing required columns. Found: {df.columns}")

    # Create a unique key per sample (video + timestamp)
    df['key'] = df['video'].astype(str) + "___" + df['frame_idx'].astype(str)

    # Determine landmark set (sorted)
    landmarks = sorted(df['landmark'].unique())

    # Pivot into one row per key
    samples = {}
    labels = {}
    for key, g in df.groupby('key'):
        # start vector with NaNs
        vec = []
        lab = int(g['label'].mode().iloc[0])  # majority label for that timestamp
        # create dict for fast lookup
        m = { row['landmark']: row for _, row in g.iterrows() }
        # For each landmark in canonical order, append x,y,z,visibility
        valid = True
        for L in landmarks:
            if L in m:
                row = m[L]
                vec.extend([float(row['x']), float(row['y']), float(row['z']), float(row['visibility'])])
            else:
                # If missing landmark, we place NaN. We'll handle later.
                vec.extend([np.nan, np.nan, np.nan, np.nan])
        samples[key] = vec
        labels[key] = lab

    X = np.array([samples[k] for k in samples])
    y = np.array([labels[k] for k in samples])
    keys = list(samples.keys())
    return X, y, landmarks, keys

def impute_and_scale(X):
    # simple imputation: replace NaN by column mean
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X

def main():
    if not os.path.exists(FINAL_CSV):
        raise FileNotFoundError(f"{FINAL_CSV} not found.")

    print("Loading and pivoting CSV...")
    X, y, landmarks, keys = load_and_pivot(FINAL_CSV)
    print("Samples:", X.shape[0], "Features:", X.shape[1])

    X = impute_and_scale(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Building pipeline (StandardScaler + MLPClassifier)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER, random_state=RANDOM_STATE))
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    print(f"Saving model pipeline to {OUT_MODEL} and metadata to {OUT_META} ...")
    dump(pipeline, OUT_MODEL)

    meta = {
        "landmarks_order": landmarks,
        "feature_per_landmark": ["x","y","z","visibility"],
        "feature_vector_length": X.shape[1]
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
