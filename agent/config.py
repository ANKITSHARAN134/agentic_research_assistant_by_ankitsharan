import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "10"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
