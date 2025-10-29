
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Vector Store Configuration
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "C:/Users/DELL PC/KUVE-1/data/processed/vector_store")

# Data Configuration
RAW_DATA_PATH = "C:/Users/DELL PC/KUVE-1/data/raw"
PROCESSED_DATA_PATH = "C:/Users/DELL PC/KUVE-1/data/processed"
CHAT_HISTORY_PATH = "C:/Users/DELL PC/KUVE-1/data/chat_history"

# LLM Parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# RAG Parameters
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Conversation Settings
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "5"))
MEMORY_KEY = "chat_history"

# Domain Settings
CHATBOT_NAME = os.getenv("CHATBOT_NAME", "Assistant")
DOMAIN = os.getenv("DOMAIN", "Customer Support")