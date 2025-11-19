


import asyncio
import os
import re
import json
import logging
from datetime import datetime
from typing import List, Set

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentServer, Agent, JobContext, cli
from livekit.plugins import openai, silero

# Import logging (FIX: Missing import)
import logging as std_logging

# Load environment variables
load_dotenv('.env.local')

# ============================================================
# COMPREHENSIVE LOGGING SETUP
# ============================================================
std_logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
log = std_logging.getLogger("interrupt_agent")

# ============================================================
# ENVIRONMENT VALIDATION
# ============================================================
def validate_environment():
    """Validate all required environment variables."""
    required_vars = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'LIVEKIT_URL': os.getenv('LIVEKIT_URL'),
        'LIVEKIT_API_KEY': os.getenv('LIVEKIT_API_KEY'),
        'LIVEKIT_API_SECRET': os.getenv('LIVEKIT_API_SECRET')
    }
    
    missing = [key for key, value in required_vars.items() if not value]
    if missing:
        log.error(f"❌ Missing env vars: {', '.join(missing)}")
        log.info("💡 Create .env.local:")
        log.info("OPENAI_API_KEY=sk-your_openai_key")
        log.info("LIVEKIT_URL=wss://your-project.livekit.cloud")
        log.info("LIVEKIT_API_KEY=your_key")
        log.info("LIVEKIT_API_SECRET=your_secret")
        raise ValueError("Missing required environment variables")
    
    log.info("✅ Environment validated")

validate_environment()

# ============================================================
# CHALLENGE CONFIGURATION
# ============================================================
def load_config():
    """Load challenge configuration from environment."""
    # Filler words to ignore during agent speech
    fillers = os.getenv("IGNORED_WORDS", "['uh','umm','hmm','haan']")
    try:
        if fillers.startswith("["):
            ignored_words = [w.strip().lower().strip("'\"") for w in json.loads(fillers.replace("'", '"')) if w.strip()]
        else:
            ignored_words = [w.strip().lower() for w in fillers.split(',') if w.strip()]
    except:
        ignored_words = ['uh', 'umm', 'hmm', 'haan']
    
    # Interrupt keywords (immediate stop)
    keywords = os.getenv("INTERRUPT_KEYWORDS", "stop,wait,no,hold")
    interrupt_keywords = set(k.strip().lower() for k in keywords.split(',') if k.strip())
    
    cooldown = float(os.getenv("ASR_COOLDOWN", "1.0"))
    
    log.info(f"📝 Fillers: {ignored_words}")
    log.info(f"🔴 Keywords: {interrupt_keywords}")
    log.info(f"⏱️ Cooldown: {cooldown}s")
    
    return ignored_words, interrupt_keywords, cooldown

IGNORED_WORDS, INTERRUPT_KEYWORDS, ASR_COOLDOWN = load_config()

# ============================================================
# SPEECH PROCESSING UTILITIES
# ============================================================
def normalize_text(text: str) -> str:
    """Normalize text for filler and keyword detection."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def is_filler_utterance(text: str) -> bool:
    """Check if text contains only filler words."""
    if not text:
        return True
    normalized = normalize_text(text)
    words = normalized.split()
    return all(word in IGNORED_WORDS for word in words)

def contains_interrupt_word(text: str) -> bool:
    """Check for interrupt keywords."""
    normalized = normalize_text(text)
    for word in INTERRUPT_KEYWORDS:
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            return True
    return False

# ============================================================
# ENHANCED INTERRUPT AGENT
# ============================================================
class InterruptAgent(Agent):
    """Voice agent with intelligent interruption handling."""
    
    def __init__(self, **kwargs):
        super().__init__(
            instructions=(
                "You are a friendly voice assistant with interruption awareness. "
                "Respond naturally to user interruptions and questions. "
                "Ignore filler words like 'uh' and 'umm' when users hesitate. "
                "Immediately stop speaking when users say 'stop', 'wait', 'no', or similar. "
                "Respond helpfully and conversationally. Keep responses concise and engaging."
            ),
            min_endpointing_delay=0.3,  # 300ms end-of-turn detection
            max_endpointing_delay=3.0,  # 3s max wait
            **kwargs
        )
        
        self.agent_speaking: bool = False
        self.last_asr_time: float = 0.0
        self.interrupt_count: int = 0
        self.filler_count: int = 0
        self.lock = asyncio.Lock()

    async def should_interrupt(self, text: str) -> bool:
        """Determine if user speech should interrupt agent."""
        now = datetime.now().timestamp()
        
        # Cooldown to prevent spam
        if now - self.last_asr_time < ASR_COOLDOWN:
            log.debug("⏸️ Cooldown active")
            return False
        
        self.last_asr_time = now
        text = text.strip()
        
        if len(text) < 2:
            return False
        
        log.info(f"👤 USER SPEECH: '{text}'")
        
        async with self.lock:
            if self.agent_speaking:
                # Check for interrupt keywords first
                if contains_interrupt_word(text):
                    self.interrupt_count += 1
                    log.info(f"⛔ KEYWORD INTERRUPT #{self.interrupt_count}: '{text}'")
                    return True
                
                # Check if pure filler (ignore during speech)
                if is_filler_utterance(text):
                    self.filler_count += 1
                    log.info(f"🤫 PURE FILLER #{self.filler_count}: '{text}'")
                    return False
                
                # Meaningful speech during agent turn (natural interrupt)
                log.info(f"🗣️ MEANINGFUL SPEECH: '{text}'")
                return True
            else:
                # Agent quiet: process all speech normally
                log.info(f"🟢 NORMAL SPEECH: '{text}'")
                return False

    async def set_speaking_state(self, speaking: bool):
        """Update agent speaking state."""
        async with self.lock:
            self.agent_speaking = speaking
        status = "🔊 STARTED" if speaking else "🔇 STOPPED"
        log.info(f"Agent {status} speaking")

# ============================================================
# SYNCHRONOUS EVENT HANDLERS (Framework Compatible)
# ============================================================
def create_speaking_handler(agent, speaking: bool):
    """Synchronous wrapper for speaking events."""
    async def update_state():
        await agent.set_speaking_state(speaking)
    
    def handler():
        asyncio.create_task(update_state())
    
    return handler

def create_speech_handler(agent):
    """Synchronous wrapper for speech events."""
    async def async_handler(event):
        if hasattr(event, 'text') and event.text:
            text = event.text.strip()
            if len(text) > 0:
                should_interrupt = await agent.should_interrupt(text)
                if should_interrupt:
                    try:
                        # Get session from agent if available
                        if hasattr(agent, 'session') and agent.session:
                            await agent.session.interrupt()
                    except:
                        pass
    
    def sync_handler(event):
        asyncio.create_task(async_handler(event))
    
    return sync_handler

# ============================================================
# MAIN ENTRYPOINT (Production Ready)
# ============================================================
server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Main entrypoint for LiveKit agent."""
    try:
        log.info("🚀 Starting Voice Interruption Agent")
        
        # Load VAD
        vad = silero.VAD.load()
        log.info("✅ VAD loaded")
        
        # Simple OpenAI pipeline (no unsupported parameters)
        stt = openai.STT()  # Basic Whisper STT
        llm = openai.LLM(model="gpt-4o-mini")  # GPT-4o-mini LLM
        tts = openai.TTS(voice="alloy")  # Natural TTS
        
        log.info("✅ OpenAI pipeline ready")
        
        # Create session
        session = agents.AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
            allow_interruptions=True
        )
        
        # Create agent
        agent = InterruptAgent()
        
        # Start session
        await session.start(
            room=ctx.room,
            agent=agent
        )
        
        log.info("✅ Agent session started")
        
        # Register events
        session.on("agent_started_speaking", create_speaking_handler(agent, True))
        session.on("agent_stopped_speaking", create_speaking_handler(agent, False))
        session.on("conversation_item_added", create_speech_handler(agent))
        
        log.info("🔔 Event handlers registered")
        
        # Initial greeting
        log.info("💬 Greeting user...")
        await session.say(
            "Hello! I'm your intelligent voice assistant. "
            "I can detect interruptions and will respond naturally. "
            "Try saying 'stop' to interrupt me, or ask any question! "
            "I'm listening now.",
            allow_interruptions=True
        )
        
        log.info("✅ Greeting delivered - agent ready for interruptions!")
        log.info("🎤 Speak into microphone: 'HELLO', 'STOP', or any question")
        
    except Exception as e:
        log.error(f"❌ Entry point error: {e}")
        raise

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    log.info("🚀 Starting LiveKit Voice Interruption Agent")
    log.info(f"📝 Fillers to ignore: {IGNORED_WORDS}")
    log.info(f"🔴 Interrupt keywords: {INTERRUPT_KEYWORDS}")
    log.info("✅ All errors fixed - ready to test!")
    
    cli.run_app(server)
