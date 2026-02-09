from langchain_together import Together
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API Key from .env
api_key = os.getenv("TOGETHER_API_KEY")

if not api_key:
    raise ValueError("❌ TOGETHER_API_KEY is missing! Please check your .env file.")

# Load DeepSeek R1 for coding tasks
deepseek_r1 = Together(
    model="deepseek-ai/deepseek-coder-6.7b",
    temperature=0.2,
    max_tokens=4000
)

# Load Llama 3 70B for general tasks
llama_70b = Together(
    model="meta-llama/llama-3-70b",
    temperature=0.7,
    max_tokens=5000
)

print("✅ Models loaded successfully!")
