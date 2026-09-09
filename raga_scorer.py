"""
raga_scorer.py
--------------------------------------------------------------------------------
Grammar-based + DTW hybrid Raga Match % scorer.

Architecture
  +---------------------------------------------------------+
  |  Live swar stream  (updated each ~100 ms from fast loop) |
  +--------------------------+------------------------------+
                             |
              +--------------+----------------+
              |              |                |
    +---------▼------+  +---▼-----------+  +-▼------------+
    | Swar conformance|  | Transition    |  | Pakad DTW    |
    | (35%)           |  | conformance   |  | matching     |
    | rolling 10s     |  | (25%) 10s     |  | (25%) 20s    |
    +---------+-------+  +---+-----------+  +-+------------+
              |              |                |
              +--------------+----------------+
                             |
                    +--------▼--------+
                    | Vadi emphasis   |
                    | (15%) 30s       |
                    +--------+--------+
                             |
                    +--------▼--------+
                    | Raga Match %    |
                    | (weighted sum)  |
                    +-----------------+

Usage:
    # In the FastAPI slow loop
    scorer = RagaScorer(grammar_path="raga_grammar.json")
    scorer.set_target_raga("Yaman")
    scorer.push_swar(swar_idx=0, voiced=True)   # called per pitch frame
    result = scorer.score()                     # dict with breakdown

    # Standalone test
    python raga_scorer.py --test-raga Yaman
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

# dtaidistance: fast C-extension DTW; falls back to numpy/pure python if C extension is not compiled
_DTW_BACKEND = "numpy"
try:
    from dtaidistance import dtw as _dtw_lib
    import dtaidistance.dtw_cc as _dtw_cc
    _DTW_BACKEND = "dtaidistance"
except ImportError:
    pass

# --- Constants ----------------------------------------------------------------

SWAR_NAMES  = ["Sa", "re", "Re", "ga", "Ga", "Ma", "ma", "Pa", "dha", "Dha", "ni", "Ni"]
SWAR_IDX    = {s: i for i, s in enumerate(SWAR_NAMES)}

# Window sizes (number of voiced swar frames; assumes ~10 fps swar events)
WIN_CONFORMANCE = 100   # last 10s at 10 fps
WIN_TRANSITION  = 100
WIN_PAKAD       = 200   # last 20s
WIN_VADI        = 300   # last 30s

# Score weights
W_CONFORM  = 0.35
W_TRANS    = 0.25
W_PAKAD    = 0.25
W_VADI     = 0.15

# DTW normalisation constant (lower -> faster decay / stricter scoring)
DTW_NORM_K = 8.5


# --- Pure-Python DTW fallback -------------------------------------------------

def _dtw_numpy(s: list[int], t: list[int]) -> float:
    """Simple O(nm) DTW on integer swar sequences."""
    n, m = len(s), len(t)
    if n == 0 or m == 0:
        return float("inf")
    D = np.full((n + 1, m + 1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s[i - 1] - t[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m]) / (n + m)


def dtw_distance(s: list[int], t: list[int]) -> float:
    """DTW distance between two swar index sequences, normalised by length."""
    if _DTW_BACKEND == "dtaidistance":
        sa = np.array(s, dtype=float)
        ta = np.array(t, dtype=float)
        try:
            d = _dtw_lib.distance_fast(sa, ta)
            return float(d) / (len(s) + len(t) + 1e-9)
        except Exception:
            pass
    return _dtw_numpy(s, t)


# --- Raga Grammar -------------------------------------------------------------

@dataclass
class RagaGrammar:
    name            : str
    aroha           : list[str]
    avaroha         : list[str]
    pakad           : list[list[str]]
    vadi            : str
    samvadi         : str
    scale_swars     : list[str]
    forbidden_swars : list[str]

    # Derived index representations (populated post-init)
    scale_idx       : set[int]      = field(default_factory=set, repr=False)
    forbidden_idx   : set[int]      = field(default_factory=set, repr=False)
    vadi_idx        : int           = field(default=-1,          repr=False)
    samvadi_idx     : int           = field(default=-1,          repr=False)
    aroha_idx       : list[int]     = field(default_factory=list, repr=False)
    avaroha_idx     : list[int]     = field(default_factory=list, repr=False)
    pakad_idx       : list[list[int]] = field(default_factory=list, repr=False)
    allowed_bigrams : set[tuple]    = field(default_factory=set, repr=False)

    def __post_init__(self):
        self.scale_idx     = {SWAR_IDX[s] for s in self.scale_swars if s in SWAR_IDX}
        self.forbidden_idx = {SWAR_IDX[s] for s in self.forbidden_swars if s in SWAR_IDX}
        self.vadi_idx      = SWAR_IDX.get(self.vadi, -1)
        self.samvadi_idx   = SWAR_IDX.get(self.samvadi, -1)
        self.aroha_idx     = [SWAR_IDX[s] for s in self.aroha if s in SWAR_IDX]
        self.avaroha_idx   = [SWAR_IDX[s] for s in self.avaroha if s in SWAR_IDX]
        self.pakad_idx     = [
            [SWAR_IDX[s] for s in phrase if s in SWAR_IDX]
            for phrase in self.pakad
        ]
        # Valid bigrams = all transitions between non-forbidden scale swars + self loops + pakad phrases
        bigrams = set()
        for seq in (self.aroha_idx, self.avaroha_idx):
            if len(seq) > 1:
                bigrams.update(zip(seq[:-1], seq[1:]))
        for phrase in self.pakad_idx:
            if len(phrase) > 1:
                bigrams.update(zip(phrase[:-1], phrase[1:]))
        for s in self.scale_idx:
            bigrams.add((s, s))
        # Allow transitions between all valid scale notes
        scale_list = [s for s in self.scale_idx if s not in self.forbidden_idx]
        for s1 in scale_list:
            for s2 in scale_list:
                bigrams.add((s1, s2))

        self.allowed_bigrams = bigrams


def load_grammar(grammar_path: str = "raga_grammar.json") -> dict[str, RagaGrammar]:
    path = Path(grammar_path)
    if not path.exists():
        # Try one level up
        path = Path(__file__).parent / grammar_path
    with open(path) as f:
        raw = json.load(f)
    grammars = {}
    for name, d in raw.items():
        try:
            grammars[name] = RagaGrammar(
                name            = name,
                aroha           = d.get("aroha", []),
                avaroha         = d.get("avaroha", []),
                pakad           = d.get("pakad", []),
                vadi            = d.get("vadi", "Sa"),
                samvadi         = d.get("samvadi", "Pa"),
                scale_swars     = d.get("scale_swars", []),
                forbidden_swars = d.get("forbidden_swars", []),
            )
        except Exception as e:
            print(f"[RagaScorer] Skipping '{name}': {e}")
    return grammars


# --- Scoring Result -----------------------------------------------------------

@dataclass
class ScoreResult:
    raga_name           : str
    match_percent       : float         # 0-100, composite
    swar_conformance    : float
    transition_conform  : float
    pakad_match         : float
    vadi_emphasis       : float
    voiced_frames       : int
    last_swar           : str
    timestamp_s         : float

    def to_dict(self) -> dict:
        return {k: round(v, 1) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


# --- Main Scorer --------------------------------------------------------------

class RagaScorer:
    """Stateful scorer. Push swar frames, call score() for the latest result."""

    def __init__(self, grammar_path: str = "raga_grammar.json"):
        self._grammars    : dict[str, RagaGrammar] = load_grammar(grammar_path)
        self._target      : Optional[RagaGrammar]  = None
        self._swar_buf    : collections.deque       = collections.deque(maxlen=WIN_VADI)
        self._voiced_total: int = 0
        self._t_start     : float = time.time()

        print(f"[RagaScorer] Loaded {len(self._grammars)} ragas: {', '.join(sorted(self._grammars.keys()))}")

    @property
    def available_ragas(self) -> list[str]:
        return sorted(self._grammars.keys())

    def set_target_raga(self, raga_name: str) -> bool:
        if raga_name in self._grammars:
            self._target = self._grammars[raga_name]
            self.reset()
            print(f"[RagaScorer] Target raga: {raga_name}")
            return True
        print(f"[RagaScorer] Raga '{raga_name}' not found")
        return False

    def reset(self):
        self._swar_buf.clear()
        self._voiced_total = 0
        self._t_start = time.time()

    def push_swar(self, swar_idx: int, voiced: bool):
        """Push a single swar frame (call this from the fast pitch loop)."""
        if voiced and swar_idx >= 0:
            self._swar_buf.append(swar_idx)
            self._voiced_total += 1
        elif voiced:
            # ornament / meend – push a None-marker to break bigram sequences
            self._swar_buf.append(-1)

    # -- Component scores ------------------------------------------------------

    def _score_swar_conformance(self, g: RagaGrammar, buf: list[int]) -> float:
        """Fraction of voiced swar frames that belong to the raga's scale."""
        voiced = [s for s in buf if s >= 0]
        if not voiced:
            return 100.0
        valid   = sum(1 for s in voiced if s in g.scale_idx and s not in g.forbidden_idx)
        penalty = sum(3 for s in voiced if s in g.forbidden_idx)  # triple-penalise forbidden
        raw = (valid - penalty) / len(voiced)
        return round(max(0.0, min(100.0, raw * 100)), 1)

    def _score_transition_conformance(self, g: RagaGrammar, buf: list[int]) -> float:
        """Fraction of consecutive swar pairs that are allowed by the raga grammar."""
        bigrams = [(buf[i], buf[i+1]) for i in range(len(buf)-1)
                   if buf[i] >= 0 and buf[i+1] >= 0]
        if not bigrams:
            return 100.0
        valid = sum(1 for b in bigrams if b in g.allowed_bigrams)
        return round(valid / len(bigrams) * 100, 1)

    def _score_pakad_matching(self, g: RagaGrammar, buf: list[int]) -> float:
        """Best DTW match between user's swar sequence and any pakad phrase."""
        if not g.pakad_idx:
            return 100.0
        voiced = [s for s in buf if s >= 0]
        if len(voiced) < 3:
            return 50.0  # insufficient data

        best_score = 0.0
        for pakad in g.pakad_idx:
            if len(pakad) < 2:
                continue
            # Sliding window matching over voiced sequence
            win = len(pakad)
            min_dist = float("inf")
            for start in range(0, max(1, len(voiced) - win + 1), max(1, win // 2)):
                seg  = voiced[start : start + win]
                dist = dtw_distance(seg, pakad)
                min_dist = min(min_dist, dist)
            score = 100.0 * math.exp(-min_dist / DTW_NORM_K)
            best_score = max(best_score, score)
        return round(best_score, 1)

    def _score_vadi_emphasis(self, g: RagaGrammar, buf: list[int]) -> float:
        """Check if vadi and samvadi are appropriately emphasised."""
        voiced = [s for s in buf if s >= 0]
        if not voiced or g.vadi_idx < 0:
            return 100.0
        n_vadi    = voiced.count(g.vadi_idx)
        n_samvadi = voiced.count(g.samvadi_idx) if g.samvadi_idx >= 0 else 0
        vadi_frac    = n_vadi    / len(voiced)
        samvadi_frac = n_samvadi / len(voiced)
        vadi_score    = 75.0 + min(25.0, (vadi_frac / 0.12) * 25.0)
        samvadi_score = 75.0 + min(25.0, (samvadi_frac / 0.06) * 25.0)
        return round((vadi_score * 0.65 + samvadi_score * 0.35), 1)

    # -- Public API ------------------------------------------------------------

    def score(self) -> Optional[ScoreResult]:
        """Return current ScoreResult, or None if no target raga is set."""
        g = self._target
        if g is None:
            return None

        buf = list(self._swar_buf)

        conform  = self._score_swar_conformance(g, buf[-WIN_CONFORMANCE:])
        trans    = self._score_transition_conformance(g, buf[-WIN_TRANSITION:])
        pakad    = self._score_pakad_matching(g, buf[-WIN_PAKAD:])
        vadi     = self._score_vadi_emphasis(g, buf[-WIN_VADI:])

        composite = (
            conform * W_CONFORM +
            trans   * W_TRANS   +
            pakad   * W_PAKAD   +
            vadi    * W_VADI
        )

        last_swar_idx = next((s for s in reversed(buf) if s >= 0), -1)
        last_swar = SWAR_NAMES[last_swar_idx] if last_swar_idx >= 0 else "–"

        return ScoreResult(
            raga_name          = g.name,
            match_percent      = round(composite, 1),
            swar_conformance   = conform,
            transition_conform = trans,
            pakad_match        = pakad,
            vadi_emphasis      = vadi,
            voiced_frames      = self._voiced_total,
            last_swar          = last_swar,
            timestamp_s        = round(time.time() - self._t_start, 2),
        )

    def get_raga_grammar(self, raga_name: str) -> Optional[dict]:
        """Return serialisable grammar dict for the frontend raga graph."""
        g = self._grammars.get(raga_name)
        if g is None:
            return None
        return {
            "name"           : g.name,
            "aroha"          : g.aroha,
            "avaroha"        : g.avaroha,
            "pakad"          : g.pakad,
            "vadi"           : g.vadi,
            "samvadi"        : g.samvadi,
            "scale_swars"    : g.scale_swars,
            "forbidden_swars": g.forbidden_swars,
            "aroha_idx"      : g.aroha_idx,
            "avaroha_idx"    : g.avaroha_idx,
            "vadi_idx"       : g.vadi_idx,
        }


# --- Standalone Test ----------------------------------------------------------

def _test_raga(raga_name: str, grammar_path: str):
    scorer = RagaScorer(grammar_path=grammar_path)
    if not scorer.set_target_raga(raga_name):
        print(f"Available ragas: {', '.join(scorer.available_ragas)}")
        return

    g = scorer._target
    print(f"\n[Test] Raga: {raga_name}")
    print(f"  Aroha   : {' '.join(g.aroha)}")
    print(f"  Avaroha : {' '.join(g.avaroha)}")
    print(f"  Pakad   : {[' '.join(p) for p in g.pakad]}")
    print(f"  Vadi    : {g.vadi}  Samvadi: {g.samvadi}")

    # Simulate a "perfect" performance of the aroha
    print("\n[Sim 1] Perfect aroha ...")
    for s in g.aroha_idx * 20:
        scorer.push_swar(s, voiced=True)
    r = scorer.score()
    print(f"  -> {r.to_dict()}")

    # Simulate a "forbidden" note performance
    print("\n[Sim 2] Mixing in forbidden swars ...")
    scorer.reset()
    all_idx = list(range(12))
    for _ in range(200):
        # 70% correct, 30% random (including forbidden)
        if random.random() < 0.7:
            s = random.choice(g.aroha_idx + g.avaroha_idx)
        else:
            s = random.choice(all_idx)
        scorer.push_swar(s, voiced=True)
    r = scorer.score()
    print(f"  -> {r.to_dict()}")

    # Simulate pakad phrases
    print("\n[Sim 3] Pakad phrases ...")
    scorer.reset()
    for _ in range(10):
        for phrase in g.pakad_idx:
            for s in phrase:
                scorer.push_swar(s, voiced=True)
    r = scorer.score()
    print(f"  -> {r.to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="RagaScorer standalone test")
    parser.add_argument("--test-raga",     default="Yaman",       help="Raga name to test")
    parser.add_argument("--grammar-path",  default="raga_grammar.json")
    args = parser.parse_args()
    _test_raga(args.test_raga, args.grammar_path)


if __name__ == "__main__":
    main()
