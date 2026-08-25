"""
pitch_engine.py
────────────────────────────────────────────────────────────────────────────────
Real-time pitch / swar / stability extraction engine.

Responsibilities
  • Accept raw PCM audio chunks (Float32, 16 kHz, mono) from a WebSocket
  • Detect fundamental frequency with torchcrepe-tiny (CUDA) or librosa.pyin (CPU)
  • Normalise to Sa-relative cents using a calibrated tonic
  • Quantise to the nearest Indian swar (12-tone chromatic)
  • Compute a rolling pitch-stability score
  • Emit a JSON-serialisable dict per chunk (the fast-loop payload)
  • Standalone CLI mode: live mic → terminal display

Usage (standalone):
    python pitch_engine.py --test-mic
    python pitch_engine.py --test-mic --tonic 220
    python pitch_engine.py --calibrate     # hold Sa for 3 s, prints detected tonic
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE         = 16_000       # Hz
CHUNK_SAMPLES       = 2048         # ~128 ms per WebSocket chunk
CREPE_HOP_SAMPLES   = 256          # torchcrepe internal hop
PYIN_HOP_LENGTH     = 512          # librosa.pyin hop

STABILITY_WINDOW_S  = 0.5          # seconds for rolling stability calc
STABILITY_WINDOW_N  = int(STABILITY_WINDOW_S * SAMPLE_RATE / CHUNK_SAMPLES)
STABILITY_K         = 20.0         # sensitivity: variance decay constant (cents²)

UNVOICED_CENTS      = -9999.0      # sentinel for silence

# 12 Indian swar names + their cents from Sa
SWAR_NAMES  = ["Sa", "re", "Re", "ga", "Ga", "Ma", "ma", "Pa", "dha", "Dha", "ni", "Ni"]
SWAR_CENTS  = np.array([0,100,200,300,400,500,600,700,800,900,1000,1100], dtype=float)
SNAP_TOL    = 50.0                 # cents – beyond this = ornament frame (-1)

# Calibration
CALIBRATION_DURATION_S = 3.0      # seconds to collect for tonic calibration
PYIN_FMIN              = 80.0     # Hz
PYIN_FMAX              = 1200.0   # Hz

# Torchcrepe: only import when available
_CREPE_AVAILABLE = False
try:
    import torch
    import torchcrepe
    _CREPE_AVAILABLE = True
except ImportError:
    pass

# librosa always required
import librosa


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class PitchFrame:
    timestamp_ms    : float
    pitch_hz        : float
    pitch_cents     : float          # Sa-relative
    swar            : str            # "Sa", "Re", …, "Ni", or "–" (unvoiced/ornament)
    swar_idx        : int            # 0-11, or -1
    swar_cents_offset: float         # how many cents off the ideal swar position
    stability       : float          # 0-100
    voiced          : bool

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Pitch Detection ──────────────────────────────────────────────────────────

class PitchDetector:
    """Thin wrapper around torchcrepe-tiny (GPU) with pYIN (CPU) fallback."""

    def __init__(self, device: str = "auto"):
        self._use_crepe = False

        if device == "auto":
            device = "cuda" if (_CREPE_AVAILABLE and self._cuda_available()) else "cpu"

        if device == "cuda" and _CREPE_AVAILABLE:
            self._use_crepe = True
            self._device    = torch.device("cuda")
            print("[PitchDetector] Using torchcrepe-tiny on CUDA")
        else:
            print("[PitchDetector] Using librosa.pyin on CPU")

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def predict(self, audio_f32: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[float, float]:
        """Return (pitch_hz, confidence) for a mono audio chunk.

        pitch_hz = 0.0  → unvoiced / below threshold
        """
        if self._use_crepe:
            return self._predict_crepe(audio_f32, sr)
        return self._predict_pyin(audio_f32, sr)

    def _predict_crepe(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        import torch
        t = torch.from_numpy(audio).unsqueeze(0).to(self._device)  # (1, N)
        with torch.no_grad():
            pitch, periodicity = torchcrepe.predict(
                t, sr,
                fmin=PYIN_FMIN, fmax=PYIN_FMAX,
                model="tiny",
                batch_size=1,
                device=self._device,
                return_periodicity=True,
                decoder=torchcrepe.decode.weighted_argmax,
            )
        f0    = float(pitch.median().cpu())
        conf  = float(periodicity.median().cpu())
        if conf < 0.3 or f0 <= 0:
            return 0.0, 0.0
        return f0, conf

    def _predict_pyin(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=PYIN_FMIN,
            fmax=PYIN_FMAX,
            sr=sr,
            hop_length=PYIN_HOP_LENGTH,
            fill_na=0.0,
        )
        voiced = voiced_flag & (voiced_prob > 0.5)
        if voiced.sum() == 0:
            return 0.0, 0.0
        median_f0   = float(np.median(f0[voiced]))
        mean_conf   = float(voiced_prob[voiced].mean())
        return median_f0, mean_conf


# ─── Swar Quantisation ────────────────────────────────────────────────────────

def hz_to_sa_cents(pitch_hz: float, tonic_hz: float) -> float:
    if pitch_hz <= 0 or tonic_hz <= 0:
        return UNVOICED_CENTS
    return 1200.0 * np.log2(pitch_hz / tonic_hz)


def snap_to_swar(cents: float) -> tuple[int, float]:
    """Return (swar_idx, offset_cents).  swar_idx=-1 if out of tolerance."""
    if cents == UNVOICED_CENTS:
        return -1, 0.0
    c_mod = cents % 1200.0
    dists = np.abs(SWAR_CENTS - c_mod)
    best  = int(np.argmin(dists))
    if dists[best] <= SNAP_TOL:
        return best, float(c_mod - SWAR_CENTS[best])
    return -1, 0.0


# ─── Stability Scoring ────────────────────────────────────────────────────────

class StabilityEstimator:
    """Rolling pitch-variance → stability score 0-100."""

    def __init__(self, window: int = STABILITY_WINDOW_N, k: float = STABILITY_K):
        self._buf = collections.deque(maxlen=window)
        self._k   = k

    def update(self, cents: float) -> float:
        if cents != UNVOICED_CENTS:
            self._buf.append(cents)
        if len(self._buf) < 2:
            return 100.0
        variance  = float(np.var(list(self._buf)))
        stability = 100.0 * float(np.exp(-variance / self._k))
        return round(min(100.0, max(0.0, stability)), 1)


# ─── Tonic Calibration ────────────────────────────────────────────────────────

def calibrate_tonic(
    audio_chunks: list[np.ndarray],
    sr: int = SAMPLE_RATE,
) -> float:
    """Estimate tonic (Sa) from a list of audio chunks.

    Median of voiced pYIN frames from the concatenated audio.
    Returns tonic in Hz (0.0 if calibration failed).
    """
    if not audio_chunks:
        return 0.0
    audio = np.concatenate(audio_chunks).astype(np.float32)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=sr,
        hop_length=PYIN_HOP_LENGTH,
        fill_na=0.0,
    )
    voiced = voiced_flag & (voiced_prob > 0.6)
    if voiced.sum() == 0:
        return 0.0
    tonic = float(np.median(f0[voiced]))
    return round(tonic, 2)


# ─── Main Engine ──────────────────────────────────────────────────────────────

class PitchEngine:
    """
    Stateful engine: call `process_chunk(pcm_bytes)` for every arriving audio
    chunk and get back a PitchFrame.

    Thread-safe: can be called from the FastAPI async context via
    asyncio.run_in_executor.
    """

    def __init__(self, tonic_hz: float = 0.0, device: str = "auto"):
        self._detector   = PitchDetector(device=device)
        self._stability  = StabilityEstimator()
        self._tonic_hz   = tonic_hz
        self._t_start    = time.time()
        self._lock       = threading.Lock()

    # ── Tonic ─────────────────────────────────────────────────────────────────
    @property
    def tonic_hz(self) -> float:
        return self._tonic_hz

    @tonic_hz.setter
    def tonic_hz(self, hz: float):
        with self._lock:
            self._tonic_hz = float(hz)

    def calibrate_from_chunks(self, chunks: list[np.ndarray]) -> float:
        tonic = calibrate_tonic(chunks)
        if tonic > 0:
            self.tonic_hz = tonic
        return tonic

    # ── Processing ────────────────────────────────────────────────────────────
    def process_chunk(self, raw_bytes: bytes) -> PitchFrame:
        """Main entry point from WebSocket handler.

        `raw_bytes` should be a flat Float32LE buffer at SAMPLE_RATE Hz.
        """
        audio = np.frombuffer(raw_bytes, dtype=np.float32).copy()

        with self._lock:
            tonic = self._tonic_hz

        # Detect pitch
        f0_hz, conf = self._detector.predict(audio, SAMPLE_RATE)

        # Normalise to cents
        cents = hz_to_sa_cents(f0_hz, tonic) if (f0_hz > 0 and tonic > 0) else UNVOICED_CENTS

        # Quantise
        swar_idx, offset = snap_to_swar(cents)
        swar_name = SWAR_NAMES[swar_idx] if swar_idx >= 0 else "–"

        # Stability
        stability = self._stability.update(cents)

        ts_ms = round((time.time() - self._t_start) * 1000, 1)

        return PitchFrame(
            timestamp_ms      = ts_ms,
            pitch_hz          = round(f0_hz, 2),
            pitch_cents       = round(cents, 2) if cents != UNVOICED_CENTS else 0.0,
            swar              = swar_name,
            swar_idx          = swar_idx,
            swar_cents_offset = round(offset, 2),
            stability         = stability,
            voiced            = f0_hz > 0,
        )


# ─── Standalone CLI ───────────────────────────────────────────────────────────

def _mic_test(tonic: float, device: str = "auto"):
    """Live microphone → terminal pitch display."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed.  Run: pip install sounddevice")
        sys.exit(1)

    engine = PitchEngine(tonic_hz=tonic, device=device)
    q: list[np.ndarray] = []

    print(f"\n[Mic Test]  tonic={tonic:.1f} Hz  Ctrl+C to stop\n")
    print(f"{'Time(s)':>8}  {'Hz':>7}  {'Cents':>7}  {'Swar':>5}  {'Offset':>7}  {'Stability':>10}")
    print("─" * 60)

    def callback(indata, frames, time_info, status):
        chunk = indata[:, 0].astype(np.float32).copy()
        frame = engine.process_chunk(chunk.tobytes())
        t = frame.timestamp_ms / 1000
        bar = "█" * int(frame.stability / 10) + "░" * (10 - int(frame.stability / 10))
        print(
            f"{t:>8.2f}  {frame.pitch_hz:>7.2f}  {frame.pitch_cents:>7.1f}  "
            f"{frame.swar:>5}  {frame.swar_cents_offset:>+7.1f}  "
            f"{bar} {frame.stability:.0f}%",
            end="\r",
        )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=CHUNK_SAMPLES,
        dtype="float32",
        callback=callback,
    ):
        try:
            while True:
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n\nStopped.")


def _calibrate_cli(device: str = "auto"):
    """Collect 3 s of mic audio, detect and print the tonic."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed.  Run: pip install sounddevice")
        sys.exit(1)

    print(f"\n[Calibration] Sing or play Sa for {CALIBRATION_DURATION_S:.0f} seconds …")
    chunks = []

    def cb(indata, frames, time_info, status):
        chunks.append(indata[:, 0].astype(np.float32).copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SAMPLES,
        dtype="float32", callback=cb,
    ):
        for remaining in range(int(CALIBRATION_DURATION_S), 0, -1):
            print(f"  {remaining}…", end="\r")
            time.sleep(1.0)

    tonic = calibrate_tonic(chunks)
    if tonic > 0:
        print(f"\n  ✓ Detected tonic: {tonic:.2f} Hz")
    else:
        print("\n  ✗ Could not detect tonic — ensure mic is working and singer is audible")
    return tonic


def main():
    parser = argparse.ArgumentParser(description="PitchEngine CLI")
    parser.add_argument("--test-mic",   action="store_true", help="Live mic → terminal display")
    parser.add_argument("--calibrate",  action="store_true", help="Calibrate tonic from 3s mic audio")
    parser.add_argument("--tonic",      type=float, default=0.0, help="Tonic Hz for --test-mic")
    parser.add_argument("--device",     default="auto", choices=["auto","cuda","cpu"])
    args = parser.parse_args()

    if args.calibrate:
        _calibrate_cli(device=args.device)
    elif args.test_mic:
        tonic = args.tonic
        if tonic <= 0:
            print("No tonic specified — auto-calibrating first …")
            tonic = _calibrate_cli(device=args.device)
        _mic_test(tonic=tonic, device=args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
