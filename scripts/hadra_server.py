"""Local HADRA (Habibi-TTS Darija, F5-TTS) speech server. Run with the TTS env, not the main .venv:

    .venv-hadra\\python.exe scripts\\hadra_server.py        # http://127.0.0.1:8021

POST /speak {"text": "<Arabic script>", "speaker": "F1"}  ->  audio/wav
GET  /health
Speakers are reference clips in models/hadra-voices/ (F1-F4 female, M1-M3 male); the voice is
whichever clip + exact transcript is passed to the model.
"""
import argparse
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "habibi-tts-doda-darija"
VOICES = ROOT / "models" / "hadra-voices"

# Settings recommended by the model card.
NFE_STEP, CFG, SWAY, SPEED = 16, 2.0, -1.0, 0.95
TAIL_SILENCE = 0.25   # seconds

app = FastAPI()
lock = threading.Lock()
state: dict = {}


class SpeakReq(BaseModel):
    text: str
    speaker: str = "F1"
    speed: float = SPEED          # below 1.0 = slower; the model card recommends 0.90-0.95


@app.get("/health")
def health():
    return {"ready": "model" in state, "speakers": sorted(state.get("refs", {}))}


@app.post("/speak")
def speak(req: SpeakReq):
    if req.speaker not in state["refs"]:
        raise HTTPException(404, f"unknown speaker {req.speaker}")
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    ref_audio, ref_text = state["refs"][req.speaker]
    # F5 sizes the clip from text length, so the last word sometimes gets squeezed off the end
    # (speech still sounding in the final 50 ms: 5/30 clips). A trailing pause mark fixed it (0/30).
    text = req.text.rstrip() + " ..."
    with lock:
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            wav, sr, _ = infer_process(ref_audio, ref_text, text, state["model"], state["vocoder"],
                                       speed=req.speed, nfe_step=NFE_STEP, cfg_strength=CFG, sway_sampling_coef=SWAY)
        torch.cuda.synchronize()
        gen = time.perf_counter() - t0
    # Silent tail so the sound driver can't swallow the last few milliseconds on playback.
    wav = np.concatenate([wav, np.zeros(int(TAIL_SILENCE * sr), dtype=wav.dtype)])
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    print(f"[{time.strftime('%H:%M:%S')}] {req.speaker} {len(wav) / sr:.1f}s audio in {gen:.1f}s", flush=True)
    return Response(buf.getvalue(), media_type="audio/wav")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8021)
    args = ap.parse_args()
    cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    print("loading HADRA ...", flush=True)
    state["model"] = load_model(DiT, cfg, str(WEIGHTS / "model_ema.safetensors"), vocab_file=str(WEIGHTS / "vocab.txt"))
    state["vocoder"] = load_vocoder("vocos")
    golden = json.loads((VOICES / "golden.json").read_text(encoding="utf-8"))
    state["refs"] = {k: preprocess_ref_audio_text(str(VOICES / f"{k}.wav"), v["text"]) for k, v in golden.items()}
    print(f"ready: speakers {sorted(state['refs'])} on http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
