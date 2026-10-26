import os
import sys
from pathlib import Path

# Keep tests offline: no embedding model download.
os.environ["EMBEDDINGS"] = "tfidf"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
