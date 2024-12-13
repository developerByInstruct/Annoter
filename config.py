import os
from dataclasses import dataclass
from dotenv import load_dotenv
import streamlit as st


# Load environment variables from .env file for local development
load_dotenv()

def get_api_key(key_name: str) -> str:
    """
    Get API key from either Streamlit secrets (production) or environment variables (local)
    """
    # Try to get from Streamlit secrets first (for production)
    try:
        return st.secrets["api_keys"][key_name.lower()]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variables (for local development)
        return os.getenv(key_name)

@dataclass
class ModelConfig:
    name: str
    model_id: str
    
# API Keys
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY")
GROQ_API_KEY = get_api_key("GROQ_API_KEY")
GROK_API_KEY = get_api_key("GROK_API_KEY")
TOGETHER_API_KEY = get_api_key("TOGETHER_API_KEY")

# Model configurations
MODELS = {
    "gemini": ModelConfig("Gemini", "gemini-1.5-flash"),
    "openai": ModelConfig("OpenAI", "gpt-4o-mini"),
    "together": ModelConfig("Together", "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"),
    "groq": ModelConfig("Groq", "llama-3.2-11b-vision-preview"),
    "grok": ModelConfig("Grok", "grok-vision-beta")
}
