"""
data_pipeline.py
--------------------------------------------------------------------------------
Saraga Hindustani Dataset Pipeline
  1. Downloads + validates via mirdata
  2. Normalises pitch curves to Sa-relative cents using the pre-extracted tonic
  3. Quantises to the 12-swar Indian chromatic scale
  4. Extracts per-window features:
       • 12-bin swar pitch-histogram
       • 12×12 swar transition matrix (144 values)
       • 128-mel spectrogram patch (128 × T_frames, computed from raw audio)
  5. Splits by ARTIST to prevent data leakage
  6. Saves .npz files + manifest.csv

Usage:
    python data_pipeline.py               # full run
    python data_pipeline.py --dry-run     # process first 5 tracks only
    python data_pipeline.py --data-home ./saraga_hindustani
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import re
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Optional

import librosa
import mirdata
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# --- Windows-safe extraction --------------------------------------------------
# Saraga filenames contain colons (e.g. "Geetinandan : Part-3 by Ajoy Chakrabarty")
# which are illegal on Windows. We intercept the mirdata download and extract
# with sanitised names instead.

WINDOWS_ILLEGAL = re.compile(r'[:<>"\\|?*]')


def _sanitise_path(name: str) -> str:
    """Replace Windows-illegal characters in a zip member path."""
    parts = Path(name).parts
    safe_parts = [WINDOWS_ILLEGAL.sub('_', p) for p in parts]
    return str(Path(*safe_parts)) if safe_parts else name


def _windows_safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract a zip file, renaming members with Windows-illegal characters."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.infolist()
        print(f"  -> Extracting {len(members)} members to {dest} ...")
        for member in tqdm(members, unit="file", desc="Extracting"):
            safe_name = _sanitise_path(member.filename)
            target    = dest / safe_name
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, 'wb') as dst:
                dst.write(src.read())
    print("  -> Extraction complete.")


def download_saraga_windows_safe(data_home: str) -> None:
    """Download saraga1.5_hindustani.zip and extract with sanitised paths.

    Bypasses mirdata's extraction to avoid the Windows colon bug.
    Falls back to the normal mirdata download on non-Windows platforms.
    """
    import platform
    data_path = Path(data_home)
    data_path.mkdir(parents=True, exist_ok=True)

    # mirdata stores the index separately; let it handle that part
    dataset = mirdata.initialize("saraga_hindustani", data_home=data_home)

    if platform.system() != "Windows":
        print("[Download] Non-Windows OS — using standard mirdata download.")
        dataset.download()
        dataset.validate()
        return

    # -- Windows path: manual download + safe extraction ----------------------
    print("[Download] Windows detected — using Windows-safe extraction.")

    # First download just the index (small, no colon paths)
    try:
        dataset.download(partial_download=['index'])
    except Exception:
        pass  # index may already exist

    # Get the zip URL from mirdata's REMOTES
    try:
        remotes  = dataset.remotes
        zip_url  = None
        zip_name = None
        for key, remote in remotes.items():
            if key == 'all' or (hasattr(remote, 'url') and remote.url.endswith('.zip')):
                zip_url  = remote.url
                zip_name = remote.filename if hasattr(remote, 'filename') else 'saraga1.5_hindustani.zip'
                break
    except Exception as e:
        print(f"[Download] Could not read REMOTES: {e}")
        zip_url  = "https://zenodo.org/records/4301737/files/saraga1.5_hindustani.zip"
        zip_name = "saraga1.5_hindustani.zip"

    zip_path = data_path / zip_name

    if not zip_path.exists():
        print(f"[Download] Downloading {zip_url} ...")
        print("  This is ~3.8 GB and may take 20-40 minutes on a typical connection.")
        with requests.get(zip_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True, desc='Downloading'
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"  -> Saved: {zip_path}")
    else:
        print(f"[Download] Zip already present: {zip_path} — skipping download.")

    _windows_safe_extract(zip_path, data_path)
    print("\n[Download] [OK] Dataset ready.")


def _resolve_audio_path(raw_path: str) -> str:
    """Return the on-disk path for a mirdata audio_path.

    On Windows, mirdata's index contains the original paths which may have
    colon characters (illegal on Windows). Our extractor replaced them with
    underscores, so we sanitize the path before handing it to librosa.
    """
    import platform
    if platform.system() != "Windows":
        return raw_path

    p = Path(raw_path)
    parts = list(p.parts)
    safe_parts = [WINDOWS_ILLEGAL.sub('_', part) for part in parts]

    # Preserve drive letter (e.g. 'C:\\' must stay as-is)
    if safe_parts and parts[0] == safe_parts[0]:
        result = Path(*safe_parts)
    else:
        # Drive letter got mangled — reconstruct carefully
        result = Path(parts[0], *[WINDOWS_ILLEGAL.sub('_', pt) for pt in parts[1:]])

    return str(result)

# --- Constants ----------------------------------------------------------------

SAMPLE_RATE      = 16_000          # Hz – resample all audio to this
HOP_LENGTH       = 512             # mel spectrogram hop
N_MELS           = 128             # mel bins
WINDOW_DURATION  = 10.0            # seconds per clip
OVERLAP          = 0.5             # fraction overlap between windows
PITCH_HOP_SEC    = 0.01            # Saraga pitch annotations are at 100 Hz (10 ms)
UNVOICED_CENTS   = -9999.0         # sentinel for unvoiced / silence frames

# Sa-relative chromatic grid for 12 Indian swars (equal-tempered approximation)
SWAR_NAMES = ["Sa", "re", "Re", "ga", "Ga", "Ma", "ma", "Pa", "dha", "Dha", "ni", "Ni"]
# Cents from Sa: 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100
SWAR_CENTS = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100], dtype=float)


# --- Tonic Normalisation ------------------------------------------------------

def hz_to_sa_cents(pitch_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    """Convert raw pitch array (Hz) to Sa-relative cents.

    Unvoiced frames (pitch_hz <= 0) are set to UNVOICED_CENTS.
    """
    cents = np.full_like(pitch_hz, UNVOICED_CENTS, dtype=float)
    voiced = pitch_hz > 0
    cents[voiced] = 1200.0 * np.log2(pitch_hz[voiced] / tonic_hz)
    return cents


def quantise_to_swar(cents: np.ndarray, tolerance: float = 50.0) -> np.ndarray:
    """Snap each cent value to the nearest swar index (0-11).

    Frames further than `tolerance` cents from any swar are mapped to -1
    (ornament / meend in transition).  Unvoiced frames stay -1.
    """
    swar_idx = np.full(len(cents), -1, dtype=int)
    for i, c in enumerate(cents):
        if c == UNVOICED_CENTS:
            continue
        # Fold into one octave
        c_mod = c % 1200.0
        dists  = np.abs(SWAR_CENTS - c_mod)
        # Also check octave boundary: e.g., Ni (1100) and Sa (0 ≡ 1200)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= tolerance:
            swar_idx[i] = min_idx
    return swar_idx


# --- Feature Extraction --------------------------------------------------------

def compute_pitch_histogram(swar_idx: np.ndarray) -> np.ndarray:
    """Normalised 12-bin histogram of swar indices (ignores -1 / unvoiced)."""
    hist = np.zeros(12, dtype=float)
    voiced = swar_idx[swar_idx >= 0]
    if len(voiced) == 0:
        return hist
    for s in voiced:
        hist[s] += 1
    hist /= hist.sum() + 1e-9
    return hist


def compute_transition_matrix(swar_idx: np.ndarray) -> np.ndarray:
    """12×12 first-order swar transition matrix (row-normalised)."""
    mat = np.zeros((12, 12), dtype=float)
    prev = -1
    for curr in swar_idx:
        if prev >= 0 and curr >= 0:
            mat[prev, curr] += 1
        if curr >= 0:
            prev = curr
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = np.where(row_sums > 0, mat / (row_sums + 1e-9), mat)
    return mat


def compute_mel_spectrogram(
    audio: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = N_MELS
) -> np.ndarray:
    """128-mel log spectrogram, shape (n_mels, T)."""
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=HOP_LENGTH,
        fmin=50.0, fmax=8000.0
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


# --- Data Augmentation --------------------------------------------------------

def augment_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    """Return list of augmented variants (in addition to the original)."""
    augmented = []

    # Time stretch ±10 %
    for rate in (0.9, 1.1):
        try:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            augmented.append(stretched)
        except Exception:
            pass

    # Pitch shift ±50 cents (half semitone) — raga-consistent small perturbation
    for semitones in (-0.5, 0.5):
        try:
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
            augmented.append(shifted)
        except Exception:
            pass

    # Additive Gaussian noise (SNR ≈ 30 dB)
    noise = np.random.randn(len(audio)).astype(np.float32) * 0.001
    augmented.append(audio + noise)

    return augmented


# --- Windowing ----------------------------------------------------------------

def window_pitch_curve(
    cents: np.ndarray,
    duration_s: float = WINDOW_DURATION,
    overlap: float = OVERLAP,
    fps: float = 1.0 / PITCH_HOP_SEC,
) -> list[np.ndarray]:
    """Slice a pitch curve (100 fps) into fixed-length windows."""
    win_frames = int(duration_s * fps)
    hop_frames = int(win_frames * (1.0 - overlap))
    windows = []
    start = 0
    while start + win_frames <= len(cents):
        windows.append(cents[start : start + win_frames])
        start += hop_frames
    return windows


def window_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    duration_s: float = WINDOW_DURATION,
    overlap: float = OVERLAP,
) -> list[np.ndarray]:
    """Slice raw audio into fixed-length windows matching pitch windows."""
    win_samples = int(duration_s * sr)
    hop_samples = int(win_samples * (1.0 - overlap))
    windows = []
    start = 0
    while start + win_samples <= len(audio):
        windows.append(audio[start : start + win_samples])
        start += hop_samples
    return windows


# --- Dataset Loading ----------------------------------------------------------

def load_dataset(data_home: str, dry_run: bool = False):
    """Initialise mirdata loader and return (dataset, track_ids)."""
    print(f"\n[mirdata] Initialising saraga_hindustani at '{data_home}' ...")
    dataset = mirdata.initialize("saraga_hindustani", data_home=data_home)
    # mirdata >= 1.0.0: track_ids is a property, not a method
    track_ids = dataset.track_ids if not callable(dataset.track_ids) else dataset.track_ids()
    print(f"  -> {len(track_ids)} tracks found")
    if dry_run:
        track_ids = track_ids[:5]
        print(f"  -> DRY RUN: processing first {len(track_ids)} tracks only")
    return dataset, track_ids


NORM_MAP = {
    'bhairabi': 'Bhairavi', 'bhairavi': 'Bhairavi',
    'shree': 'Shree',
    'todi': 'Todi',
    'lalat': 'Lalit', 'lalit': 'Lalit',
    'marwa': 'Marwa',
    'miya malhar': 'Miya Malhar', 'mian malhar': 'Miya Malhar',
    'jog': 'Jog',
    'yaman kalyan': 'Yaman', 'yaman': 'Yaman', 'kalyan': 'Yaman',
    'bhimpalas': 'Bhimpalasi', 'bhimpalasi': 'Bhimpalasi',
    'malkauns': 'Malkauns',
    'bhoop': 'Bhoopali', 'bhoopali': 'Bhoopali',
    'bihag': 'Bihag',
    'kedar': 'Kedar',
    'dhani': 'Dhani',
}

TARGET_RAGAS = {'Bhairavi', 'Shree', 'Todi', 'Lalit', 'Marwa', 'Miya Malhar', 'Yaman', 'Bhimpalasi', 'Jog', 'Malkauns'}


def extract_raga(track) -> Optional[str]:
    """Pull raga common_name from Saraga metadata JSON, normalize, and filter to TARGET_RAGAS."""
    meta = track.metadata if hasattr(track, "metadata") and track.metadata else {}
    raw_raga = None

    # Primary: raags[0]['common_name'] — the English raga name
    raags = meta.get("raags", [])
    if raags and isinstance(raags, list) and len(raags) > 0:
        raw_raga = raags[0].get("common_name", "").strip() or raags[0].get("name", "").strip()

    # Fallback 1: strip 'Raag ' prefix from title
    if not raw_raga:
        title = meta.get("title", "").strip()
        if title.lower().startswith("raag "):
            raw_raga = title[5:].strip()

    # Fallback 2: parse track_id e.g. '0_Raag_Shree' -> 'Shree'
    if not raw_raga:
        tid = getattr(track, 'track_id', '')
        parts = tid.split('_')
        if len(parts) >= 3 and parts[1].lower() == 'raag':
            raw_raga = '_'.join(parts[2:]).replace('_', ' ')

    if not raw_raga:
        return None

    norm = NORM_MAP.get(raw_raga.lower(), raw_raga.title().strip())
    return norm if norm in TARGET_RAGAS else None


def extract_artist(track) -> str:
    """Pull lead artist name for artist-level split.

    Saraga metadata structure:
      { 'album_artists': [{'mbid':..., 'name': 'Deborshee Bhattacharya'}],
        'artists': [{'artist': {'name':...}, 'lead': True, ...}] }
    """
    meta = track.metadata if hasattr(track, "metadata") and track.metadata else {}

    # Primary: album_artists
    album_artists = meta.get("album_artists", [])
    if album_artists and isinstance(album_artists, list):
        name = album_artists[0].get("name", "").strip()
        if name:
            return name

    # Secondary: lead artist from artists list
    artists = meta.get("artists", [])
    for entry in artists:
        if entry.get("lead") and isinstance(entry.get("artist"), dict):
            name = entry["artist"].get("name", "").strip()
            if name:
                return name

    return "unknown"


# --- Artist-Level Split --------------------------------------------------------

def artist_split(
    rows: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test by artist so no artist appears in multiple splits."""
    rng = np.random.default_rng(seed)

    # Group artists by the ragas they cover (to balance class exposure)
    artist_to_ragas: dict[str, set] = collections.defaultdict(set)
    artist_to_rows: dict[str, list] = collections.defaultdict(list)
    for row in rows:
        artist_to_ragas[row["artist"]].add(row["raga"])
        artist_to_rows[row["artist"]].append(row)

    artists = list(artist_to_rows.keys())
    rng.shuffle(artists)

    n = len(artists)
    n_train = max(1, int(n * train_frac))
    n_val   = max(1, int(n * val_frac))

    split_map = {}
    for i, a in enumerate(artists):
        if i < n_train:
            split_map[a] = "train"
        elif i < n_train + n_val:
            split_map[a] = "val"
        else:
            split_map[a] = "test"

    for row in rows:
        row["split"] = split_map[row["artist"]]

    df = pd.DataFrame(rows)
    print("\n[Split] Artist-level assignment:")
    print(df.groupby("split")["raga"].value_counts().to_string())
    return df


# --- Main Pipeline ------------------------------------------------------------

def run_pipeline(
    data_home: str = "./saraga_hindustani",
    features_dir: str = "./features",
    dry_run: bool = False,
    augment: bool = True,
) -> None:
    features_path = Path(features_dir)
    features_path.mkdir(parents=True, exist_ok=True)

    # -- 1. Load dataset ------------------------------------------------------
    dataset, track_ids = load_dataset(data_home, dry_run=dry_run)

    # -- 2. Collect metadata & check class distribution -----------------------
    rows: list[dict] = []
    skipped = 0

    print("\n[Pass 1/2] Collecting metadata ...")
    for tid in tqdm(track_ids, unit="track"):
        track = dataset.track(tid)
        raga   = extract_raga(track)
        artist = extract_artist(track)

        if raga is None:
            skipped += 1
            continue
        if track.pitch is None:
            skipped += 1
            continue
        if track.tonic is None:
            skipped += 1
            continue

        rows.append({"track_id": tid, "raga": raga, "artist": artist})

    print(f"\n  -> {len(rows)} usable tracks  (skipped {skipped})")

    if len(rows) == 0:
        print("ERROR: No usable tracks found. Have you downloaded the dataset?")
        print("  Run:  python data_pipeline.py --download")
        sys.exit(1)

    # Class distribution
    raga_counts = collections.Counter(r["raga"] for r in rows)
    print("\n[Class Distribution]")
    for raga, cnt in raga_counts.most_common():
        print(f"  {raga:<35} {cnt:>3} recordings")

    # -- 3. Encode labels -----------------------------------------------------
    le = LabelEncoder()
    all_ragas = sorted(raga_counts.keys())
    le.fit(all_ragas)
    for row in rows:
        row["label"] = int(le.transform([row["raga"]])[0])

    label_path = features_path / "label_encoder.pkl"
    with open(label_path, "wb") as f:
        pickle.dump(le, f)
    print(f"\n  -> Saved label encoder: {label_path}  ({len(le.classes_)} classes)")

    # -- 4. Artist-level split ------------------------------------------------
    df_meta = artist_split(rows)

    # -- 5. Feature extraction loop -------------------------------------------
    split_data: dict[str, dict] = {
        s: {"hist": [], "trans": [], "mel": [], "label": []}
        for s in ("train", "val", "test")
    }
    manifest_rows = []

    print("\n[Pass 2/2] Extracting features ...")
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), unit="track"):
        tid    = row["track_id"]
        split  = row["split"]
        label  = row["label"]
        raga   = row["raga"]
        artist = row["artist"]

        track = dataset.track(tid)

        try:
            # Load pitch curve and tonic
            pitch_data = track.pitch          # mirdata F0Data
            tonic_hz   = float(track.tonic)  # single float or array

            if hasattr(tonic_hz, "__len__"):  # some tracks store (time, freq)
                tonic_hz = float(np.median(tonic_hz[tonic_hz > 0]))

            pitch_hz  = np.array(pitch_data.frequencies, dtype=float)
            cents     = hz_to_sa_cents(pitch_hz, tonic_hz)
            swar_seq  = quantise_to_swar(cents)

            # Window the pitch curve
            cent_windows  = window_pitch_curve(cents)
            swar_windows  = window_pitch_curve(swar_seq.astype(float))

            # Load audio — sanitize path on Windows (colons -> dashes)
            audio_path = _resolve_audio_path(track.audio_path)
            audio, _   = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            audio_windows = window_audio(audio)

            n_windows = min(len(cent_windows), len(audio_windows))

            for w_idx in range(n_windows):
                c_win  = cent_windows[w_idx]
                s_win  = swar_windows[w_idx].astype(int)
                a_win  = audio_windows[w_idx]

                hist   = compute_pitch_histogram(s_win)
                trans  = compute_transition_matrix(s_win).flatten()
                mel    = compute_mel_spectrogram(a_win)  # (128, T)

                split_data[split]["hist"].append(hist)
                split_data[split]["trans"].append(trans)
                split_data[split]["mel"].append(mel)
                split_data[split]["label"].append(label)

                manifest_rows.append({
                    "track_id": tid, "window": w_idx, "split": split,
                    "label": label, "raga": raga, "artist": artist,
                    "tonic_hz": tonic_hz,
                })

            # -- Augmentation (train split only) ------------------------------
            if augment and split == "train":
                aug_audios = augment_audio(audio)
                for aug_audio in aug_audios:
                    aug_windows = window_audio(aug_audio)
                    for aw in aug_windows[:n_windows]:  # keep same count
                        aug_hist  = hist  # re-use pitch hist (unchanged for small shifts)
                        aug_trans = trans
                        aug_mel   = compute_mel_spectrogram(aw)
                        split_data["train"]["hist"].append(aug_hist)
                        split_data["train"]["trans"].append(aug_trans)
                        split_data["train"]["mel"].append(aug_mel)
                        split_data["train"]["label"].append(label)

        except Exception as exc:
            tqdm.write(f"  [WARN] track {tid}: {exc}")
            continue

    # -- 6. Save features -----------------------------------------------------
    print("\n[Saving features]")
    mel_times = {}  # store T dimension per split for later padding
    for split, data in split_data.items():
        if not data["label"]:
            print(f"  [WARN] Split '{split}' is empty — skipping")
            continue

        hists  = np.array(data["hist"],  dtype=np.float32)   # (N, 12)
        trans  = np.array(data["trans"], dtype=np.float32)   # (N, 144)
        labels = np.array(data["label"], dtype=np.int64)     # (N,)

        # Mel: variable T — pad/crop to median T
        mel_list = data["mel"]
        T_vals   = [m.shape[1] for m in mel_list]
        T_target = int(np.median(T_vals))
        mel_padded = np.zeros((len(mel_list), N_MELS, T_target), dtype=np.float32)
        for i, m in enumerate(mel_list):
            t = min(m.shape[1], T_target)
            mel_padded[i, :, :t] = m[:, :t]

        mel_times[split] = T_target
        out_path = features_path / f"{split}.npz"
        np.savez_compressed(
            out_path,
            hist=hists,
            trans=trans,
            mel=mel_padded,
            label=labels,
        )
        print(f"  -> {out_path}  [{split}]  samples={len(labels)}  mel_T={T_target}")

    # -- 7. Class weights -----------------------------------------------------
    train_labels = np.array(split_data["train"]["label"], dtype=np.int64)
    unique_cls   = np.unique(train_labels)
    weights      = compute_class_weight("balanced", classes=unique_cls, y=train_labels)
    full_weights  = np.ones(len(le.classes_), dtype=np.float32)
    for cls, w in zip(unique_cls, weights):
        full_weights[cls] = w
    np.save(features_path / "class_weights.npy", full_weights)
    print(f"\n  -> class_weights.npy  (max_weight={full_weights.max():.2f})")

    # -- 8. Manifest ----------------------------------------------------------
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = features_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  -> manifest.csv  ({len(manifest_df)} clips)")

    # -- 9. Save metadata for the backend -------------------------------------
    meta_out = {
        "n_ragas": int(len(le.classes_)),
        "raga_names": list(le.classes_),
        "mel_T": mel_times,
        "swar_names": SWAR_NAMES,
    }
    with open(features_path / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"  -> metadata.json  (n_ragas={meta_out['n_ragas']})")

    print("\n[OK] Pipeline complete.\n")


# --- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Saraga Hindustani Data Pipeline")
    parser.add_argument(
        "--data-home", default="./saraga_hindustani",
        help="Path where the dataset is stored (default: ./saraga_hindustani)"
    )
    parser.add_argument(
        "--features-dir", default="./features",
        help="Output directory for extracted features (default: ./features)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only the first 5 tracks (for quick validation)"
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable data augmentation on the training split"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download + validate the dataset before processing"
    )
    args = parser.parse_args()

    if args.download:
        print("[mirdata] Downloading saraga_hindustani ...")
        download_saraga_windows_safe(args.data_home)
        print("[mirdata] Download + validation complete.")
        # Exit after download so user can verify before running the full pipeline
        sys.exit(0)

    run_pipeline(
        data_home=args.data_home,
        features_dir=args.features_dir,
        dry_run=args.dry_run,
        augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
