import sys
from pathlib import Path

# Make sure app/ is importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.ingest import run_ingestion

if __name__ == "__main__":
    force = "--force" in sys.argv
    run_ingestion(force_download=force)