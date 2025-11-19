
import re
import soundfile as sf   # <-- replaces librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ======================================================
#              CONFIG
# ======================================================

WHISPER_MODEL = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IGNORED_FILLERS = {"uh", "umm", "hmm", "haan", "huh", "erm", "mmm"}
COMMAND_WORDS = {"stop", "wait", "hold on", "no", "not that", "cancel"}

# ======================================================
#               LOAD WHISPER
# ======================================================

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(DEVICE)
whisper.eval()

# ======================================================
#                  HELPERS
# ======================================================

def normalize(t):
    t = t.lower().strip()
    t = re.sub(r"[^a-zA-Z0-9\s']", " ", t)
    return re.sub(r"\s+", " ", t)


def classify_interrupt(text, agent_speaking=True):
    t = normalize(text)
    tokens = t.split()

    if not agent_speaking:
        return "VALID", "agent not speaking"

    if len(tokens) > 0 and all(tok in IGNORED_FILLERS for tok in tokens):
        return "IGNORE", "pure filler"

    for cmd in COMMAND_WORDS:
        if cmd in t:
            return "INTERRUPT", f"command '{cmd}'"

    return "INTERRUPT", "meaningful speech"


# ======================================================
#               FIXED NON-HALLUCINATING WHISPER
# ======================================================

def transcribe(audio):
    features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)

    with torch.no_grad():
        pred_ids = whisper.generate(
            features,
            max_new_tokens=200,
            temperature=0.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=3
        )

    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return text


# ======================================================
#                       MAIN
# ======================================================

def main(audio_path):
    audio, sr = sf.read(audio_path)     # <---- soundfile loader (no numba issues)

    if sr != 16000:
        import scipy.signal
        audio = scipy.signal.resample_poly(audio, 16000, sr)
        sr = 16000

    audio = audio.astype(np.float32)

    text = transcribe(audio)
    print("\nWhisper Transcript:", text)

    decision, reason = classify_interrupt(text, agent_speaking=True)

    print("\n=== FINAL DECISION ===")
    print("Decision:", decision)
    print("Reason:", reason)


if __name__ == "__main__":
    main("WhatsApp Ptt 2025-11-18 at 3.47.10 PM.ogg")
