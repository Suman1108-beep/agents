
import os
import re
import numpy as np
import soundfile as sf
import torch
import asyncio
from datetime import datetime, timezone
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ============================================================
#  CONFIG (dynamic filler + command lists)
# ============================================================

IGNORED_FILLERS = set(
    w.strip().lower() for w in
    (os.getenv("IGNORED_FILLERS", "uh,umm,hmm,haan,huh,erm,mmm")).split(",")
)

COMMAND_WORDS = set(
    w.strip().lower() for w in
    (os.getenv("COMMAND_WORDS", "stop,wait,hold on,no,not that,cancel")).split(",")
)

# ============================================================
#  LOAD WHISPER MODEL
# ============================================================

WHISPER_MODEL = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(DEVICE)
whisper.eval()

# ============================================================
#  HELPERS
# ============================================================

def utc_ts():
    return datetime.now(timezone.utc).isoformat()

def normalize(text: str):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text)

async def transcribe_full_audio(audio_np):
    """Stable + non-hallucination Whisper mode."""
    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        ids = whisper.generate(
            inputs,
            task="transcribe",
            language="en",
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text.strip()

# ============================================================
#  CLASSIFIER
# ============================================================

def classify_interrupt(text, agent_speaking):
    t = normalize(text)
    tokens = t.split()

    if not agent_speaking:
        return "VALID", "agent not speaking"

    # Pure filler
    if len(tokens) > 0 and all(tok in IGNORED_FILLERS for tok in tokens):
        return "IGNORE", "pure filler"

    # Commands
    for phrase in sorted(COMMAND_WORDS, key=lambda x: -len(x)):
        if phrase in t:
            return "INTERRUPT", f"command '{phrase}'"

    # Any meaningful speech
    if len(t.strip()) > 0:
        return "INTERRUPT", "meaningful speech"

    return "IGNORE", "empty or noise"

# ============================================================
#  LIVEKIT HANDLER
# ============================================================

class InterruptionHandler:
    def __init__(self, agent):
        self.agent = agent
        self.buffer = []
        self.sr = 16000

    async def on_audio_frame(self, pcm_float32: np.ndarray, agent_speaking: bool):
        self.buffer.append(pcm_float32)

        if len(self.buffer) < 4:  # ~400ms
            return

        audio = np.concatenate(self.buffer)
        self.buffer = []

        text = await transcribe_full_audio(audio)

        print({
            "ts": utc_ts(),
            "event": "FRAME_TRANSCRIBED",
            "text": text,
            "agent_speaking": agent_speaking
        })

        decision, reason = classify_interrupt(text, agent_speaking)

        print({
            "ts": utc_ts(),
            "event": "INTERRUPT_EVAL",
            "decision": decision,
            "reason": reason
        })

        if decision == "INTERRUPT" and agent_speaking:
            print({"ts": utc_ts(), "event": "ACTION_STOP_TTS"})
            await self.agent.stop_tts()

# ============================================================
#  SAFE AUDIO LOADER (NO LIBROSA)
# ============================================================

def load_audio_safe(path, sr=16000):
    audio, orig_sr = sf.read(path)

    # Convert stereo → mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if orig_sr != sr:
        import scipy.signal
        audio = scipy.signal.resample_poly(audio, sr, orig_sr)

    return audio.astype(np.float32)

# ============================================================
#  OFFLINE TEST DRIVER
# ============================================================

async def test_file(agent, path):
    handler = InterruptionHandler(agent)

    audio = load_audio_safe(path, sr=16000)
    frame = 16000 // 2  # 0.5s

    await agent.start_tts()

    for i in range(0, len(audio), frame):
        chunk = audio[i:i+frame]
        await handler.on_audio_frame(chunk, agent_speaking=True)

    await agent.end_tts()

# ============================================================
#  MOCK AGENT FOR TESTING
# ============================================================

class MockAgent:
    async def start_tts(self):
        print({"ts": utc_ts(), "event": "AGENT_TTS_START"})

    async def stop_tts(self):
        print({"ts": utc_ts(), "event": "AGENT_TTS_STOP"})

    async def end_tts(self):
        print({"ts": utc_ts(), "event": "AGENT_TTS_END"})

# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python interrupt_handler.py <audiofile>")
        exit()

    agent = MockAgent()
    asyncio.run(test_file(agent, sys.argv[1]))
