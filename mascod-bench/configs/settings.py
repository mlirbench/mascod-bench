import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Nova validation endpoint (via SSH tunnel)
VALIDATION_API_URL = os.getenv(
    "VALIDATION_API_URL"
)

# Safety checks (fail early instead of debugging later)
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env")