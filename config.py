from pathlib import Path
import os


MODEL_NAME = "openai:gpt-5.4-mini"
TEMPERATURE = 0
TIMEOUT = 300
MAX_TOKENS = 25000
THREAD_ID = "great-gatsby-da"

SYSTEM_PROMPT = """You are a business document assistant.

## Capabilities

- `ingest_pdf_corpus`: loads and parses internal PDF documents into the corpus store.
Do not guess ingestion counts or corpus size—ground them in tool results from the parsed files."""


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value
