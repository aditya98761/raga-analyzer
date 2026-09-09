"""
backend/main.py
────────────────────────────────────────────────────────────────────────────────
FastAPI application — Hindustani Raga Practice Analyzer backend.

Two processing loops over a single WebSocket:

  FAST LOOP  (~50-100 ms)
    PCM bytes  →  PitchEngine.process_chunk()
               →  PitchFrame JSON  →  client

  SLOW LOOP  (~1-2 s, async background task)
    Accumulated swar stream  →  RagaScorer.score()
                             +  ONNX raga classifier (if no raga pre-selected)
                             →  ScoreResult JSON  →  client

Endpoints
  GET   /                          serve frontend/index.html
  GET   /static/{path}             serve frontend/ assets
  GET   /api/ragas                 list of ragas + grammar
  GET   /api/raga/{name}           single raga grammar
  POST  /api/calibrate-tonic       upload 3 s PCM → return tonic Hz
  WS    /ws/audio                  main streaming WebSocket

Run:
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import librosa

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pitch_engine import PitchEngine, CHUNK_SAMPLES, SAMPLE_RATE, calibrate_tonic
from raga_scorer  import RagaScorer, SWAR_NAMES

GRAMMAR_PATH   = ROOT / "raga_grammar.json"
FRONTEND_DIR   = ROOT / "frontend"
MODELS_DIR     = Path(__file__).parent / "models"
ONNX_MODEL     = MODELS_DIR / "raga_classifier.onnx"
FEATURES_META  = ROOT / "features" / "metadata.json"

# ─── ONNX Raga Classifier (optional – used if model has been trained) ─────────

class RagaClassifier:
    """Wrapper around the ONNX raga classifier exported from training."""

    def __init__(self):
        self._session  = None
        self._labels   : list[str] = []
        self._loaded   = False

    def load(self, model_path: Path, meta_path: Path) -> bool:
        if not model_path.exists():
            print(f"[Classifier] ONNX model not found at {model_path} — skipping")
            return False
        try:
            import onnxruntime as ort
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._cuda_available()
                else ["CPUExecutionProvider"]
            )
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            print(f"[Classifier] Loaded ONNX model: {model_path}")
            print(f"[Classifier] Providers: {self._session.get_providers()}")

            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self._labels = meta.get("raga_names", [])
                print(f"[Classifier] {len(self._labels)} raga classes")
            self._loaded = True
            return True
        except Exception as e:
            print(f"[Classifier] Load failed: {e}")
            return False

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def predict(self, mel_patch: np.ndarray) -> dict[str, float]:
        """Return {raga_name: probability, …} for a (128, T) mel patch."""
        if not self._loaded or self._session is None:
            return {}
        try:
            # Shape: (1, 1, 128, T)
            x     = mel_patch[np.newaxis, np.newaxis, :, :].astype(np.float32)
            out   = self._session.run(None, {"input": x})
            probs = self._softmax(out[0][0])
            return {name: round(float(p), 4)
                    for name, p in zip(self._labels, probs)}
        except Exception as e:
            print(f"[Classifier] Inference error: {e}")
            return {}

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    @property
    def available(self) -> bool:
        return self._loaded


# ─── App globals ──────────────────────────────────────────────────────────────

app = FastAPI(title="Raga Practice Analyzer", version="1.0.0")


@app.middleware("http")
async def no_cache_middleware(request, call_next):
    """Prevent browser from caching static files during development."""
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


_classifier = RagaClassifier()
_grammar_cache: dict = {}
_scorer_pool: dict[str, RagaScorer] = {}   # one scorer per connected client id


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _grammar_cache
    _classifier.load(ONNX_MODEL, FEATURES_META)
    if GRAMMAR_PATH.exists():
        with open(GRAMMAR_PATH) as f:
            _grammar_cache = json.load(f)
        print(f"[App] Loaded grammar for {len(_grammar_cache)} ragas")
    else:
        print(f"[App] WARNING: {GRAMMAR_PATH} not found")

    # Mount static frontend
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─── Static / Frontend ────────────────────────────────────────────────────────

@app.get("/")
async def index():
    html = FRONTEND_DIR / "index.html"
    if not html.exists():
        raise HTTPException(404, "Frontend not found — did you build it?")
    return FileResponse(str(html), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


# ─── REST API ─────────────────────────────────────────────────────────────────

@app.get("/api/ragas")
async def list_ragas():
    """Return list of raga names and their full grammar."""
    return JSONResponse({"ragas": _grammar_cache})


@app.get("/api/raga/{raga_name}")
async def get_raga(raga_name: str):
    """Return grammar for a single raga (for the frontend raga graph)."""
    if raga_name not in _grammar_cache:
        raise HTTPException(404, f"Raga '{raga_name}' not in grammar database")
    return JSONResponse(_grammar_cache[raga_name])


@app.post("/api/calibrate-tonic")
async def calibrate_tonic_endpoint(request_body: dict):
    """Accept base64-encoded Float32LE PCM audio, return detected tonic Hz.

    Expects JSON: {"audio_b64": "<base64>", "sample_rate": 16000}
    """
    import base64
    audio_b64 = request_body.get("audio_b64", "")
    sr        = int(request_body.get("sample_rate", SAMPLE_RATE))
    raw       = base64.b64decode(audio_b64)
    audio     = np.frombuffer(raw, dtype=np.float32)
    tonic     = calibrate_tonic([audio], sr=sr)
    if tonic <= 0:
        raise HTTPException(422, "Could not detect tonic — ensure audio contains a clear Sa")
    return {"tonic_hz": tonic}


@app.post("/api/upload-audio")
async def upload_audio_endpoint(
    file: UploadFile = File(...),
    raga: str = Form(...),
    tonic_hz: float = Form(0.0)
):
    """Accept an uploaded audio file (WAV, MP3, etc.), analyze it, and return frame scores."""
    import tempfile
    import os

    contents = await file.read()
    suffix = Path(file.filename).suffix if file.filename else ".wav"

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_name = tmp.name

    try:
        # Load audio at 16000 Hz, mono
        audio, sr = librosa.load(tmp_name, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load audio file: {e}")
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass

    # Limit duration to 60 seconds to prevent high latency/CPU usage
    MAX_SAMPLES = 60 * SAMPLE_RATE
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]

    # Autocalibrate tonic if not provided / <= 0
    auto_tonic = False
    if tonic_hz <= 0:
        tonic_hz = calibrate_tonic([audio], sr=SAMPLE_RATE)
        if tonic_hz <= 0:
            # Fallback: use C4 (261.63 Hz), the most common Sa in Hindustani vocal music
            tonic_hz = 261.63
            auto_tonic = True
            print(f"[Upload] Auto-tonic detection failed, using default Sa = {tonic_hz} Hz")

    # Initialize processing engine and scorer
    engine = PitchEngine(tonic_hz=tonic_hz, device="cpu")  # use CPU to avoid blocking GPU
    scorer = RagaScorer(grammar_path=str(GRAMMAR_PATH))
    if not scorer.set_target_raga(raga):
        raise HTTPException(status_code=400, detail=f"Raga '{raga}' not found in grammar database.")

    frames_output = []

    # Chunk-by-chunk processing (2048 samples = ~128ms)
    num_samples = len(audio)
    for start_sample in range(0, num_samples, CHUNK_SAMPLES):
        chunk = audio[start_sample : start_sample + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        frame = engine.process_chunk(chunk.tobytes())
        scorer.push_swar(frame.swar_idx, frame.voiced)
        score_res = scorer.score()

        frames_output.append({
            "frame": frame.to_dict(),
            "score": score_res.to_dict() if score_res else None
        })

    return {
        "tonic_hz": tonic_hz,
        "auto_tonic": auto_tonic,
        "raga": raga,
        "frames": frames_output
    }


# ─── WebSocket Handler ────────────────────────────────────────────────────────

SLOW_LOOP_INTERVAL = 1.5   # seconds between raga score updates

@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    print(f"[WS] Client connected: {client_id}")

    # Per-connection state
    engine  = PitchEngine(tonic_hz=0.0, device="auto")
    scorer  = RagaScorer(grammar_path=str(GRAMMAR_PATH))
    target_raga: Optional[str] = None
    last_slow  = time.time()

    # Accumulator for mel patches (used by ONNX classifier in slow loop)
    mel_accum: list[np.ndarray] = []

    try:
        async for message in websocket.iter_text():
            # ── Control messages (JSON text frames) ──────────────────────────
            if message.startswith("{"):
                ctrl = json.loads(message)

                # Set tonic
                if "tonic_hz" in ctrl:
                    engine.tonic_hz = float(ctrl["tonic_hz"])
                    print(f"[WS] {client_id}: tonic set to {engine.tonic_hz:.2f} Hz")

                # Set target raga
                if "raga" in ctrl:
                    target_raga = ctrl["raga"]
                    scorer.set_target_raga(target_raga)
                    print(f"[WS] {client_id}: raga = {target_raga}")
                    await websocket.send_text(json.dumps({"type": "raga_set", "raga": target_raga}))

                # Calibration request: payload contains base64 PCM
                if "calibrate_audio_b64" in ctrl:
                    import base64
                    raw   = base64.b64decode(ctrl["calibrate_audio_b64"])
                    audio = np.frombuffer(raw, dtype=np.float32)
                    tonic = await asyncio.get_event_loop().run_in_executor(
                        None, calibrate_tonic, [audio]
                    )
                    engine.tonic_hz = tonic if tonic > 0 else engine.tonic_hz
                    await websocket.send_text(json.dumps({
                        "type": "calibration_result",
                        "tonic_hz": tonic,
                        "success": tonic > 0,
                    }))
                continue  # no binary processing for control frames

        # ── Binary audio frames handled separately ────────────────────────────
    except WebSocketDisconnect:
        pass
    finally:
        print(f"[WS] Client disconnected: {client_id}")
        if client_id in _scorer_pool:
            del _scorer_pool[client_id]


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """Primary binary-streaming WebSocket endpoint.

    Protocol:
      CLIENT → SERVER:
        Binary frames: raw PCM Float32LE at 16 kHz (CHUNK_SAMPLES samples)
        Text frames:   JSON control messages (tonic_hz, raga, calibrate)

      SERVER → CLIENT:
        Text frames: JSON with type="pitch" or type="score"
    """
    await websocket.accept()
    client_id = id(websocket)
    print(f"[WS] Stream client: {client_id}")

    engine = PitchEngine(tonic_hz=0.0, device="auto")
    scorer = RagaScorer(grammar_path=str(GRAMMAR_PATH))

    target_raga : Optional[str] = None
    last_slow   : float = time.time()
    mel_buf     : list[np.ndarray] = []

    loop = asyncio.get_event_loop()

    async def send_json(data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    try:
        while True:
            try:
                # Non-blocking receive with a short timeout so slow loop can fire
                data = await asyncio.wait_for(websocket.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                data = None
            except WebSocketDisconnect:
                break

            if data is not None:
                if data["type"] == "websocket.receive":
                    if "bytes" in data and data["bytes"]:
                        # ── FAST LOOP ─────────────────────────────────────────
                        raw_bytes = data["bytes"]

                        # Run pitch extraction off the event loop thread
                        frame = await loop.run_in_executor(
                            None, engine.process_chunk, raw_bytes
                        )

                        # Push swar to scorer
                        scorer.push_swar(frame.swar_idx, frame.voiced)

                        # Send pitch frame immediately
                        payload = frame.to_dict()
                        payload["type"] = "pitch"
                        await send_json(payload)

                    elif "text" in data and data["text"]:
                        # ── CONTROL MESSAGES ──────────────────────────────────
                        try:
                            ctrl = json.loads(data["text"])
                        except json.JSONDecodeError:
                            continue

                        if "tonic_hz" in ctrl:
                            engine.tonic_hz = float(ctrl["tonic_hz"])
                            await send_json({"type": "ack", "tonic_hz": engine.tonic_hz})

                        if "raga" in ctrl:
                            target_raga = ctrl["raga"]
                            scorer.set_target_raga(target_raga)
                            await send_json({"type": "raga_set", "raga": target_raga})

                        if "reset" in ctrl:
                            scorer.reset()
                            await send_json({"type": "ack", "reset": True})

            # ── SLOW LOOP ─────────────────────────────────────────────────────
            now = time.time()
            if now - last_slow >= SLOW_LOOP_INTERVAL:
                last_slow = now

                score_payload: dict = {"type": "score"}

                # Grammar-based scoring
                result = scorer.score()
                if result is not None:
                    score_payload.update(result.to_dict())

                # ONNX classifier (if model loaded and no raga pre-selected)
                if _classifier.available and target_raga is None and mel_buf:
                    mel = mel_buf[-1]
                    probs = await loop.run_in_executor(
                        None, _classifier.predict, mel
                    )
                    if probs:
                        top_raga = max(probs, key=probs.get)
                        score_payload["classifier_top_raga"] = top_raga
                        score_payload["classifier_probs"]    = probs

                await send_json(score_payload)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error for {client_id}: {e}")
    finally:
        print(f"[WS] Stream client disconnected: {client_id}")


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status"          : "ok",
        "classifier_ready": _classifier.available,
        "ragas_loaded"    : len(_grammar_cache),
        "swar_names"      : SWAR_NAMES,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
