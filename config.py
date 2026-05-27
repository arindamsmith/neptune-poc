import os
from dotenv import load_dotenv

load_dotenv()

NEPTUNE_ENDPOINT  = os.getenv("NEPTUNE_ENDPOINT", "localhost")
NEPTUNE_PORT      = int(os.getenv("NEPTUNE_PORT", 8182))
AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID  = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
NEPTUNE_URL       = f"https://{NEPTUNE_ENDPOINT}:{NEPTUNE_PORT}"
