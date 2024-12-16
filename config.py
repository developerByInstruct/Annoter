import os
from dotenv import load_dotenv
from dataclasses import dataclass
import streamlit as st

# Load environment variables from .env file for local development
load_dotenv()

def get_api_key(key_name: str) -> str:
    """
    Get API key from either Streamlit secrets (production) or environment variables (local)
    """
    # Try to get from Streamlit secrets first (for production)
    try:
        # Try exact key name first
        if key_name in st.secrets["api_keys"]:
            return st.secrets["api_keys"][key_name]
        # Try lowercase version
        if key_name.lower() in st.secrets["api_keys"]:
            return st.secrets["api_keys"][key_name.lower()]
        # Try uppercase version
        if key_name.upper() in st.secrets["api_keys"]:
            return st.secrets["api_keys"][key_name.upper()]
        raise KeyError(f"API key {key_name} not found in secrets")
    except (KeyError, FileNotFoundError):
        # Fall back to environment variables (for local development)
        return os.getenv(key_name)

@dataclass
class ModelConfig:
    name: str
    model_id: str
    
# API Keys
OPENAI_API_KEY = get_api_key("openai_api_key")
GOOGLE_API_KEY = get_api_key("google_api_key")
GROQ_API_KEY = get_api_key("groq_api_key")
GROK_API_KEY = get_api_key("grok_api_key")
TOGETHER_API_KEY = get_api_key("together_api_key")

# Model configurations
MODELS = {
    "gemini": ModelConfig("Gemini", "gemini-1.5-flash"),
    #"gpt-4o": ModelConfig("OpenAI", "gpt-4o"),
    #"together": ModelConfig("Together", "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"),
    #"groq": ModelConfig("Groq", "llama-3.2-11b-vision-preview"),
    "grok": ModelConfig("Grok", "grok-vision-beta")
}
