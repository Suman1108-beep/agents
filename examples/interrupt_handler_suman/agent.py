import os 
import re 
import time 
import asyncio 
import logging 
from typing import Optional, Callable, Dict, Set 
import numpy as np 
import scipy.signal 
import torch 
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
from livekit import rtc, api 
import sys # For checking librosa requirement

# Check for librosa availability for spectral gating 
try:
    import librosa
except ImportError:
    # Define a dummy frame function if librosa is not installed
    def librosa_frame_safe(y, frame_length, hop_length):
        # Fallback to manual framing implementation
        n = len(y)
        if n < frame_length:
            pad = frame_length - n
            y = np.concatenate([y, np.zeros(pad, dtype=y.dtype)])
            n = len(y)
        
        # Calculate number of frames
        if n < frame_length:
            n_frames = 0
        else:
            n_frames = 1 + (n - frame_length) // hop_length

        frames = np.zeros((frame_length, n_frames), dtype=y.dtype)
        for i in range(n_frames):
            start = i * hop_length
            frames[:, i] = y[start:start + frame_length]
        return frames

# If librosa is installed, use its framing function
if 'librosa' in sys.modules:
    def librosa_frame_safe(y, frame_length, hop_length):
        return librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)


LOG = logging.getLogger("caption_agent") 
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO")) 

# ---------------- Config ---------------- 
LIVEKIT_URL = os.getenv("LIVEKIT_URL") 
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY") 
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET") 
ROOM_NAME = os.getenv("ROOM_NAME", "my-test-room") 
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-small") 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
SAMPLE_RATE = 16000 
INPUT_SR = 48000 

# Caption-mode specifics 
CONTEXT_SECONDS = float(os.getenv("CONTEXT_SECONDS", "1.6")) 
CONTEXT_SAMPLES_48K = int(INPUT_SR * CONTEXT_SECONDS) 
MIN_ASR_SECONDS = float(os.getenv("MIN_ASR_SECONDS", "0.06")) 
ASR_GAP = float(os.getenv("ASR_GAP", "0.35")) 
ASR_CONCURRENCY = int(os.getenv("ASR_CONCURRENCY", "2")) 

# Tune thresholds 
RMS_MIN_FOR_SHORT = float(os.getenv("RMS_MIN_FOR_SHORT", "0.002")) 
CONF_MIN_KEEP = float(os.getenv("CONF_MIN_KEEP", "0.02")) 
DUPLICATE_COOLDOWN = float(os.getenv("DUPLICATE_COOLDOWN", "0.8")) 

# fillers & commands - CRITICAL FOR THE CHALLENGE
IGNORED_FILLERS = set(x.strip().lower() for x in os.getenv( 
    "IGNORED_FILLERS", "uh,umm,hmm,haan,huh,erm,mmm,um,oh").split(",") if x.strip()) 
COMMAND_WORDS = set(x.strip().lower() for x in os.getenv( 
    "COMMAND_WORDS", "stop,wait,hold on,no,not that,cancel,pause").split(",") if x.strip()) 

# Important names boosting (add your name tokens here) 
IMPORTANT_NAMES = set(x.strip().lower() for x in os.getenv("IMPORTANT_NAMES", "suman,sharma").split(",") if x.strip()) 

# Paralinguistic regexes 
LAUGH_RE = re.compile(r"\b(ha+|hah+|hehe+|hihi+|lol|lmao|rofl)\b", re.I) 
SIGH_RE = re.compile(r"\b(sigh|sighs)\b", re.I) 
COUGH_RE = re.compile(r"\b(cough|coughs)\b", re.I) 

# ---------------- Helpers ---------------- 
def normalize_text(t: str) -> str: 
    if not t: return "" 
    t = t.lower().strip() 
    # Only remove non-alphanumeric/non-space characters, preserving apostrophes for contractions
    t = re.sub(r"[^a-z0-9\s']+", " ", t) 
    return re.sub(r"\s+", " ", t).strip() 

def downmix_int16_to_float32(buf: np.ndarray) -> np.ndarray: 
    if buf.size == 0: return np.zeros(0, dtype=np.float32) 
    if buf.size % 2 == 0: 
        stereo = buf.reshape(-1, 2) 
        mono = stereo.mean(axis=1) 
        return (mono.astype(np.float32) / 32768.0) 
    return (buf.astype(np.float32) / 32768.0) 

def resample_48k_to_16k(arr48: np.ndarray) -> np.ndarray: 
    if arr48.size == 0: return np.zeros(0, dtype=np.float32) 
    return scipy.signal.resample_poly(arr48, SAMPLE_RATE, INPUT_SR).astype(np.float32) 

def rms(audio: np.ndarray) -> float: 
    if audio.size == 0: return 0.0 
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) 

# Simple HPF and AGC helpers from scipy.signal import butter, lfilter 
from scipy.signal import butter, lfilter

def highpass_filter(audio: np.ndarray, sr: int = SAMPLE_RATE, cutoff: float = 80.0) -> np.ndarray: 
    try: 
        b, a = butter(1, cutoff / (sr / 2), btype='high') 
        return lfilter(b, a, audio) 
    except Exception: return audio 

def normalize_rms(audio: np.ndarray, target: float = 0.04, eps: float = 1e-9) -> np.ndarray: 
    cur = float(np.sqrt(max(np.mean(audio**2), eps))) 
    if cur <= 0: return audio 
    return audio * (target / cur) 

# optional noise reduction (soft dependency) 
try: 
    import noisereduce as nr 
    def reduce_noise(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray: 
        try: return nr.reduce_noise(y=audio, sr=sr) 
        except Exception: return audio 
except Exception: 
    def reduce_noise(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray: return audio 

# stronger preprocess to reduce hallucinations 
def preprocess_strict(audio16: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray: 
    """HPF + light spectral gate + optional noise reduction + AGC + limiter.""" 
    if audio16 is None or audio16.size == 0: return audio16 
    
    # HPF 
    try: audio16 = highpass_filter(audio16, sr=sr, cutoff=80.0) 
    except Exception: pass 
    
    # simple spectral gating: short-time energy attenuation 
    try: 
        win_len = 1024 
        hop = 256 
        if len(audio16) >= win_len: 
            # frame and attenuate low-energy frames 
            frames = librosa_frame_safe(audio16, win_len, hop) 
            energies = np.sqrt((frames ** 2).mean(axis=0)) 
            median_e = max(np.median(energies), 1e-9) 
            factors = np.minimum(1.0, np.maximum(0.25, energies / (median_e * 0.6))) 
            out = np.zeros(len(audio16), dtype=np.float32) 
            win = np.hanning(win_len) 
            # The framing output shape might be (frame_length, n_frames) or (n_frames, frame_length)
            # Depending on librosa version or implementation. Assuming (frame_length, n_frames) for indexing.
            if frames.shape[0] != win_len: frames = frames.T 

            for i in range(frames.shape[1]): 
                start = i * hop 
                # Ensure the windowed segment is not out of bounds
                end = min(start + win_len, len(out))
                segment_len = end - start
                
                if segment_len > 0:
                     # Apply win and factor, only over the segment_len
                    out[start:end] += (frames[:segment_len, i] * win[:segment_len]) * factors[i] 
            audio16 = out 
    except Exception: 
        # if anything fails, continue with original 
        pass 
        
    # optional noise reduce 
    try: audio16 = reduce_noise(audio16, sr=sr) 
    except Exception: pass 
    
    # AGC / normalize 
    audio16 = normalize_rms(audio16, target=0.04) 
    
    # soft limiter 
    audio16 = np.clip(audio16, -0.99, 0.99) 
    return audio16 

# ---------------- ASR wrapper (Whisper small) ---------------- 
class WhisperASR: 
    def __init__(self, model_name: str = WHISPER_MODEL, device: str = DEVICE): 
        LOG.info("Loading Whisper model %s on %s ...", model_name, device) 
        self.device = device 
        self.processor = WhisperProcessor.from_pretrained(model_name) 
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device) 
        self.model.eval() 
        LOG.info("Whisper loaded.") 

    @torch.inference_mode() 
    def transcribe_with_confidence(self, audio16: np.ndarray) -> (str, float): 
        if audio16 is None or audio16.size == 0: return "", 0.0 
        
        proc_out = self.processor(audio16, sampling_rate=SAMPLE_RATE, return_tensors="pt") 
        input_features = proc_out.input_features.to(self.device) 
        attention_mask = proc_out.attention_mask.to(self.device) if "attention_mask" in proc_out else None 
        
        # beam search + early stopping tends to reduce short hallucinations 
        gen_kwargs = dict( 
            input_features=input_features, 
            max_new_tokens=200, 
            do_sample=False, 
            temperature=0.0, 
            return_dict_in_generate=True, 
            output_scores=True, 
            num_beams=4, 
            early_stopping=True, 
            no_repeat_ngram_size=3, 
            task="transcribe", 
            language="en", 
        ) 
        if attention_mask is not None: 
            gen_kwargs["attention_mask"] = attention_mask 
            
        outputs = self.model.generate(**gen_kwargs) 
        
        # decode robustly 
        text = "" 
        try: 
            text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip() 
        except Exception: 
            try: text = self.processor.decode(outputs.sequences[0], skip_special_tokens=True).strip() 
            except Exception: text = "" 

        # compute token-based confidence robustly (best-effort) 
        scores = getattr(outputs, "scores", None) or [] 
        if not scores: return text, 0.4 
        
        seq = outputs.sequences[0] if hasattr(outputs, "sequences") else None 
        if seq is None: return text, 0.4 
        
        token_probs = [] 
        min_len = min(len(scores), seq.shape[0] - 1) 
        for i in range(min_len): 
            try: 
                step_logits = scores[i] 
                logits = step_logits[0] if (step_logits.dim() == 2 and step_logits.shape[0] == 1) else step_logits 
                token_id = int(seq[i + 1].item()) 
                if 0 <= token_id < logits.shape[-1]: 
                    probs = torch.softmax(logits, dim=-1) 
                    token_probs.append(float(probs[token_id].cpu().numpy())) 
            except Exception: continue 
            
        confidence = float(sum(token_probs) / len(token_probs)) if token_probs else 0.4 
        confidence = max(0.0, min(1.0, confidence)) 
        return text, confidence 

# ---------------- Interrupt / Caption Agent ---------------- 
class CaptionAgent: 
    def __init__(self): 
        self.asr = WhisperASR() 
        self.buffers: Dict[str, np.ndarray] = {} 
        # State: False when agent is quiet, True when TTS is active.
        self.agent_speaking = False 
        self._asr_sema = asyncio.Semaphore(ASR_CONCURRENCY) 
        self._last_asr_time: Dict[str, float] = {} 
        self._last_transcript: Dict[str, str] = {} 
        self._last_transcript_ts: Dict[str, float] = {} 
        # This callback is what the decision logic will invoke
        self.stop_tts_cb: Optional[Callable] = None 

    # Public method to update agent state (called by your TTS logic)
    def set_agent_speaking(self, is_speaking: bool):
        """Used to signal the ASR loop whether the agent is currently generating TTS."""
        self.agent_speaking = is_speaking
        LOG.info(f"Agent speaking state updated to: {is_speaking}")

    async def attach_and_run(self, livekit_url: str, token_jwt: str): 
        self.room = rtc.Room() 
        @self.room.on("track_subscribed") 
        def track_sub(track, pub, participant): 
            pid = participant.identity 
            LOG.info("[%s] audio track subscribed", pid) 
            if track.kind != rtc.TrackKind.KIND_AUDIO: return 
            if pid in self.buffers: 
                LOG.info("[%s] already subscribed", pid) 
                return 
            
            self.buffers[pid] = np.zeros(0, dtype=np.float32) 
            self._last_asr_time[pid] = 0.0 
            self._last_transcript[pid] = "" 
            self._last_transcript_ts[pid] = 0.0 
            asyncio.create_task(self._consume_audio(track, pid)) 

        LOG.info("Connecting to LiveKit...") 
        await self.room.connect(livekit_url, token_jwt) 
        LOG.info("Connected to LiveKit room") 

    async def _consume_audio(self, track, pid: str): 
        LOG.info("[%s] starting audio consumer", pid) 
        stream = rtc.AudioStream(track) 
        async for evt in stream: 
            try: 
                raw = np.frombuffer(evt.frame.data, dtype=np.int16) 
                mono48 = downmix_int16_to_float32(raw) 
                buf = self.buffers.get(pid, np.zeros(0, dtype=np.float32)) 
                buf = np.concatenate([buf, mono48]) 
                
                # cap the buffer to context seconds 
                if len(buf) > CONTEXT_SAMPLES_48K: 
                    buf = buf[-CONTEXT_SAMPLES_48K:] 
                self.buffers[pid] = buf 
                
                # respect ASR gap (cooldown) to avoid spamming ASR 
                now = time.time() 
                if (now - self._last_asr_time.get(pid, 0.0)) < ASR_GAP: 
                    continue 
                
                # build 16k context and preprocess strictly to reduce hallucinations 
                seg16 = resample_48k_to_16k(buf) 
                seg16 = preprocess_strict(seg16, sr=SAMPLE_RATE) 
                
                # allow very short segments in caption mode only if RMS high 
                if (len(seg16) / SAMPLE_RATE) < MIN_ASR_SECONDS: 
                    if rms(seg16) < RMS_MIN_FOR_SHORT: 
                        continue 
                
                # schedule ASR 
                self._last_asr_time[pid] = now 
                asyncio.create_task(self._run_asr(seg16.copy(), pid)) 
            except Exception: 
                LOG.exception("[%s] audio consumer error", pid) 

    async def _run_asr(self, audio16: np.ndarray, pid: str): 
        async with self._asr_sema: 
            try: 
                text, conf = self.asr.transcribe_with_confidence(audio16) 
                text = (text or "").strip() 
                norm = normalize_text(text) 
                now = time.time() 
                
                # dedupe: suppress frequent duplicates 
                last = self._last_transcript.get(pid, "") 
                last_ts = self._last_transcript_ts.get(pid, 0.0) 
                if text and text == last and (now - last_ts) < DUPLICATE_COOLDOWN: 
                    LOG.debug("[%s] duplicate transcript suppressed (cooldown)", pid) 
                    self._last_transcript_ts[pid] = now # update timestamp to extend suppression window 
                    return 
                
                if text: 
                    self._last_transcript[pid] = text 
                    self._last_transcript_ts[pid] = now 
                
                rt = rms(audio16) 
                LOG.info('TRANSCRIPT [%s] — "%s" (conf=%.3f) agent_speaking=%s rms=%.6f', pid, text, conf, self.agent_speaking, rt) 

                # Paralinguistic logging (kept as-is)
                if LAUGH_RE.search(text): LOG.info('[%s] PARALINGUISTIC — LAUGH detected', pid) 
                if SIGH_RE.search(text): LOG.info('[%s] PARALINGUISTIC — SIGH detected', pid) 
                if COUGH_RE.search(text): LOG.info('[%s] PARALINGUISTIC — COUGH detected', pid) 
                if rt > 0.25: LOG.info('[%s] PARALINGUISTIC — LOUD_EVENT (possible cough/shout) rms=%.3f', pid, rt) 
                if 0.02 < rt < 0.07 and len(text) < 4: LOG.info('[%s] PARALINGUISTIC — BREATH/EXHALATION detected rms=%.3f', pid, rt) 

                # name boosting 
                tokens = set(norm.split()) 
                found_names = tokens.intersection(IMPORTANT_NAMES) 
                if found_names: LOG.info('[%s] HIT_NAME — %s (boosted)', pid, ','.join(found_names)) 

                # Low-confidence suppression: drop tiny low-confidence garbage 
                if not found_names: 
                    # Use a stricter check for very short segments to combat hallucination
                    is_short_and_quiet = (len(norm) <= 3) and rt < 0.06
                    if conf < CONF_MIN_KEEP and is_short_and_quiet: 
                        LOG.info('DECISION [%s] — IGNORE (very_low_confidence_short_quiet) — "%s" conf=%.3f', pid, text, conf) 
                        return 

                # Decision logic
                decision, reason = self._decide_from_text(norm, conf, self.agent_speaking, found_names) 
                
                # Log the decision clearly for debugging/validation
                if decision == "IGNORE" and self.agent_speaking and "pure_filler" in reason:
                     LOG.info('DECISION [%s] — **FILLER IGNORED** (Agent Speaking) — "%s" conf=%.3f', pid, text, conf) 
                elif decision == "INTERRUPT":
                     LOG.info('DECISION [%s] — **INTERRUPT DETECTED** (%s) — "%s" conf=%.3f', pid, reason, text, conf)
                else:
                    LOG.info('DECISION [%s] — %s (%s) — "%s" conf=%.3f', pid, decision, reason, text, conf) 

                if decision == "INTERRUPT" and self.agent_speaking: 
                    LOG.info('[%s] -> ACTION: stop_tts() - Valid Interruption', pid) 
                    try: 
                        if self.stop_tts_cb: 
                            maybe = self.stop_tts_cb() 
                            if asyncio.iscoroutine(maybe): 
                                asyncio.create_task(maybe) 
                    except Exception: 
                        LOG.exception('stop_tts_cb failed') 

            except Exception: 
                LOG.exception('ASR/decision task failed for pid=%s', pid) 

    def _decide_from_text(self, normalized_text: str, confidence: float, agent_speaking: bool, found_names: set) -> (str, str):
        """
        Implements the core challenge logic to distinguish fillers from meaningful interruptions.

        The logic flow is:
        1. Ignore if empty/noise.
        2. If Agent is **quiet**: Always register as VALID speech.
        3. If Agent is **speaking**:
           a. Immediate INTERRUPT if a command word or an important name is present.
           b. Only INTERRUPT if the transcript is NOT composed *only* of filler words.
           c. Otherwise, IGNORE (it's pure filler).
        """
        if not normalized_text:
            return "IGNORE", "empty_or_noise"
        
        tokens = [t for t in normalized_text.split() if t]

        # Scenario 2: Agent is QUIET -> Any speech is VALID
        if not agent_speaking:
            # Low confidence check: If the confidence is very low AND the text is very short/likely noise,
            # we might still want to ignore it even when the agent is quiet to prevent starting a TTS response 
            # to a hallucination. This is an extra robustness layer.
            if confidence < 0.5 and len(tokens) <= 2:
                # Still log it, but as 'VALID' if it passes a minimal threshold, or 'IGNORE' if too low.
                 return "VALID", "agent_not_speaking" # Lowering this threshold might be too aggressive, stick to VALID
            
            return "VALID", "agent_not_speaking" # All speech is meaningful when quiet

        # Scenario 3: Agent is SPEAKING -> Filter interruptions
        
        # 3a. Immediate INTERRUPT if Important Name
        if found_names:
            return "INTERRUPT", f"found_name_{','.join(sorted(found_names))}"

        # 3a. Immediate INTERRUPT if Command Word
        joined = " ".join(tokens)
        for phrase in sorted(COMMAND_WORDS, key=lambda x: -len(x)):
            if phrase in joined:
                return "INTERRUPT", f"command_{phrase}"
        
        # 3b. IGNORE if PURE FILLER
        if all(t in IGNORED_FILLERS for t in tokens):
            return "IGNORE", "pure_filler"
            
        # 3c. Otherwise, it's a meaningful interruption (e.g., "wait what" or "hello")
        return "INTERRUPT", "caption_mode_meaningful"

# ---------------- Main ---------------- 
async def main(): 
    if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET: 
        LOG.error("Missing LiveKit env vars (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)") 
        raise SystemExit(1) 
        
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
        .with_identity("interrupt-agent").with_name("InterruptAgent") \
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME)) \
        .to_jwt() 
        
    agent = CaptionAgent() 

    # --- Demo stop_tts callback and state management ---
    async def stop_tts(): 
        LOG.warning("stop_tts called: Stopping Agent TTS immediately!")
        # In a real agent, you would signal your TTS pipeline to stop here.
        agent.set_agent_speaking(False) # Update state after stopping TTS

    # Your agent's main TTS logic would call set_agent_speaking(True) 
    # when it starts generating speech. For this demo, we can simulate it.
    
    LOG.info("Registering stop_tts callback.")
    agent.stop_tts_cb = stop_tts 
    
    # SIMULATION: Start a dummy task to toggle agent_speaking state
    async def speaking_simulator():
        await asyncio.sleep(5)
        LOG.warning("SIMULATION: Agent STARTING to speak...")
        agent.set_agent_speaking(True)
        await asyncio.sleep(10)
        LOG.warning("SIMULATION: Agent FINISHING speaking...")
        agent.set_agent_speaking(False)
        await asyncio.sleep(5)
        LOG.warning("SIMULATION: Agent STARTING to speak again...")
        agent.set_agent_speaking(True)
        # It will stay speaking until an interrupt occurs or main loop stops

    asyncio.create_task(speaking_simulator())
    # --- End Demo ---
    
    await agent.attach_and_run(LIVEKIT_URL, token) 
    
    try: 
        await asyncio.Event().wait() 
    except asyncio.CancelledError: 
        LOG.info("shutting down") 

if __name__ == "__main__": 
    try: 
        # Check if running on the correct environment
        if not (os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_API_KEY") and os.getenv("LIVEKIT_API_SECRET")):
             print("!!! WARNING: LiveKit environment variables not set. Exiting. !!!")
             sys.exit(1)
             
        asyncio.run(main()) 
    except KeyboardInterrupt: 
        LOG.info("Exit")
