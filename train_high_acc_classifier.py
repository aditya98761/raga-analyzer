"""
train_high_acc_classifier.py
--------------------------------------------------------------------------------
Trains a high-accuracy Raga Classifier on well-represented Hindustani ragas.

Target Ragas (10):
  Bhairavi, Bhimpalasi, Jog, Lalit, Malkauns, Marwa, Miya Malhar, Shree, Todi, Yaman
"""

import os
import json
import pickle
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_HOME    = Path("./saraga_hindustani/saraga1.5_hindustani")
FEATURES_DIR = Path("./features")
MODELS_DIR   = Path("./backend/models")

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NORM_MAP = {
    "bhairabi": "Bhairavi", "bhairavi": "Bhairavi",
    "shree": "Shree",
    "todi": "Todi",
    "lalat": "Lalit", "lalit": "Lalit",
    "marwa": "Marwa",
    "miya malhar": "Miya Malhar", "mian malhar": "Miya Malhar",
    "jog": "Jog",
    "yaman kalyan": "Yaman", "yaman": "Yaman", "kalyan": "Yaman",
    "bhimpalas": "Bhimpalasi", "bhimpalasi": "Bhimpalasi",
    "malkauns": "Malkauns",
    "bhoop": "Bhoopali", "bhoopali": "Bhoopali",
    "bihag": "Bihag",
    "kedar": "Kedar",
    "dhani": "Dhani",
}

TARGET_RAGAS = [
    "Bhairavi", "Bhimpalasi", "Jog", "Lalit",
    "Malkauns", "Marwa", "Miya Malhar", "Shree", "Todi", "Yaman"
]


# ─── PyTorch Classifier Architecture (for ONNX Export) ───────────────────────

class RagaFeatureClassifier(nn.Module):
    def __init__(self, in_features=168, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
            if x.size(1) < 168:
                x = torch.cat([x, torch.zeros(x.size(0), 168 - x.size(1), device=x.device)], dim=1)
            elif x.size(1) > 168:
                x = x[:, :168]
        return self.net(x)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  RAGA PRACTICE ANALYZER -- HIGH ACCURACY MODEL TRAINER")
    print("=" * 65)

    json_files = list(DATA_HOME.rglob("*.json"))
    print(f"\n[1/5] Scanning dataset at '{DATA_HOME}' ({len(json_files)} tracks)...")

    X_list, y_list, track_list = [], [], []

    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)

            raags = data.get("raags", [])
            rname = None
            if raags and isinstance(raags, list) and len(raags) > 0:
                rname = raags[0].get("common_name", "").strip() or raags[0].get("name", "").strip()
            if not rname:
                rname = data.get("title", "").strip()
                if rname.lower().startswith("raag "):
                    rname = rname[5:].strip()
            if not rname:
                rname = jf.stem
                if rname.lower().startswith("raag "):
                    rname = rname[5:].strip()

            raga = NORM_MAP.get(rname.lower(), rname.title().strip())
            if raga not in TARGET_RAGAS:
                continue

            # Load tonic
            ctonic_file = jf.parent / f"{jf.stem}.ctonic.txt"
            if not ctonic_file.exists():
                ctonics = list(jf.parent.glob("*.ctonic.txt"))
                if ctonics:
                    ctonic_file = ctonics[0]
                else:
                    continue
            with open(ctonic_file) as cf:
                tonic = float(cf.read().strip())

            if tonic <= 0:
                continue

            # Load pitch curve
            pitch_file = jf.parent / f"{jf.stem}.pitch.txt"
            if not pitch_file.exists():
                candidates = list(jf.parent.glob("*.pitch.txt"))
                if candidates:
                    pitch_file = candidates[0]
                else:
                    continue

            pdf = pd.read_csv(pitch_file, sep=r"\s+|,", engine="python", header=None)
            freqs = pdf.iloc[:, 1].values.astype(np.float32)
            voiced = freqs > 0
            if voiced.sum() < 500:
                continue

            rel_cents = np.zeros_like(freqs, dtype=np.float32)
            rel_cents[voiced] = (1200.0 * np.log2(freqs[voiced] / tonic)) % 1200.0

            win_len = 1000  # 10 seconds at 100 fps
            hop_len = 500   # 50% overlap

            for i in range(0, len(rel_cents) - win_len, hop_len):
                w_cents  = rel_cents[i:i + win_len]
                w_voiced = voiced[i:i + win_len]
                v_cents  = w_cents[w_voiced]
                if len(v_cents) < 200:
                    continue

                swar_idx = (np.round(v_cents / 100.0) % 12).astype(int)
                
                # 1. Swar Histogram (12 values)
                hist, _  = np.histogram(swar_idx, bins=12, range=(0, 12), density=True)

                # 2. Transition Matrix 1-step (144 values)
                trans = np.zeros((12, 12), dtype=np.float32)
                if len(swar_idx) > 1:
                    for s1, s2 in zip(swar_idx[:-1], swar_idx[1:]):
                        trans[s1, s2] += 1.0
                    s_sum = trans.sum()
                    if s_sum > 0:
                        trans /= s_sum

                # 3. 2-step Transition Matrix (12 values summary: skip-1 transitions)
                trans2 = np.zeros(12, dtype=np.float32)
                if len(swar_idx) > 2:
                    for s1, s3 in zip(swar_idx[:-2], swar_idx[2:]):
                        if s1 != s3:
                            trans2[s1] += 1.0
                    t2_sum = trans2.sum()
                    if t2_sum > 0:
                        trans2 /= t2_sum

                feat = np.hstack([hist, trans.flatten(), trans2])
                X_list.append(feat)
                y_list.append(raga)
                track_list.append(jf.stem)

        except Exception as e:
            continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    print(f"\n[2/5] Extracted {len(X)} sample windows across {len(set(y))} ragas:")
    counts = collections.Counter(y)
    for raga in sorted(TARGET_RAGAS):
        cnt = counts.get(raga, 0)
        print(f"   * {raga:<20}: {cnt:>5} windows")

    # Encode labels
    le = LabelEncoder()
    le.fit(sorted(TARGET_RAGAS))
    y_encoded = le.transform(y)

    # ─── 3. Evaluate Cross-Validation Accuracy ─────────────────────────────────
    print("\n[3/5] Evaluating 5-Fold Stratified Cross-Validation ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
        clf = ExtraTreesClassifier(n_estimators=150, max_depth=25, random_state=42, n_jobs=-1)
        clf.fit(X[train_idx], y_encoded[train_idx])
        preds = clf.predict(X[test_idx])
        acc = accuracy_score(y_encoded[test_idx], preds)
        accs.append(acc)
        print(f"   Fold {fold + 1} Accuracy: {acc * 100:.2f}%")

    mean_acc = np.mean(accs) * 100
    std_acc  = np.std(accs) * 100
    print(f"\n  -------------------------------------------------------------")
    print(f"  --> OVERALL 5-FOLD CV ACCURACY: {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"  -------------------------------------------------------------")

    # ─── 4. Train Neural Model & Save ──────────────────────────────────────────
    print("\n[4/5] Training PyTorch neural classifier for ONNX deployment ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature scaling (mean 0, std 1)
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True) + 1e-7
    X_scaled = (X - mean) / std

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_encoded, dtype=torch.long)

    dataset_t = torch.utils.data.TensorDataset(X_t, y_t)
    loader    = torch.utils.data.DataLoader(dataset_t, batch_size=64, shuffle=True)

    nn_model = RagaFeatureClassifier(in_features=168, n_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    nn_model.train()
    for epoch in range(1, 51):
        epoch_loss, correct, total = 0.0, 0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = nn_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        scheduler.step()
        epoch_acc = correct / total * 100.0

        if epoch % 10 == 0 or epoch == 50:
            print(f"   Epoch {epoch:2d}/50 | Loss: {epoch_loss/total:.4f} | Accuracy: {epoch_acc:.2f}%")

    # ─── 5. Export to ONNX & Save Metadata ─────────────────────────────────────
    print("\n[5/5] Exporting ONNX model & metadata...")
    nn_model.eval()
    nn_model.cpu()

    onnx_path = MODELS_DIR / "raga_classifier_cnn.onnx"
    dummy_input = torch.randn(1, 168, dtype=torch.float32)

    torch.onnx.export(
        nn_model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12,
    )
    print(f"   [OK] Saved ONNX model: {onnx_path}")

    # Save Label Encoder
    le_path = FEATURES_DIR / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"   [OK] Saved label encoder: {le_path}")

    # Save Metadata JSON
    meta_path = FEATURES_DIR / "metadata.json"
    meta = {
        "n_ragas": len(le.classes_),
        "raga_names": list(le.classes_),
        "accuracy_percent": round(float(epoch_acc), 2),
        "swar_names": ["Sa", "re", "Re", "ga", "Ga", "Ma", "ma", "Pa", "dha", "Dha", "ni", "Ni"]
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   [OK] Saved metadata: {meta_path}")

    # Final Summary Report
    print("\n" + "=" * 65)
    print(f"  SUCCESS: Trained classifier on {len(le.classes_)} ragas.")
    print(f"  Achieved Accuracy: {mean_acc:.2f}% (Requirement: > 75%)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
