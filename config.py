import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    name: str
    model_id: str
    
MODELS = {
    "gemini": ModelConfig("Gemini", "gemini-1.5-flash"),
    "openai": ModelConfig("OpenAI", "gpt-4o-mini"),
    "together": ModelConfig("Together", "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"),
    "groq": ModelConfig("Groq", "llama-3.2-11b-vision-preview"),
    "grok": ModelConfig("Grok", "grok-vision-beta")
}

# API Keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Note: Using GOOGLE_API_KEY for Gemini
GROK_API_KEY = os.getenv("GROK_API_KEY")
