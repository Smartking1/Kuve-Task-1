"# KUVE Chatbot with RAG

An intelligent chatbot system powered by LangChain and Groq LLM, designed to provide information about KUVE - an AI-Powered Classifieds Marketplace platform. The chatbot uses Retrieval Augmented Generation (RAG) to deliver accurate, context-aware responses based on KUVE's documentation.

## 🌟 Features

- **RAG-Enhanced Responses**: Utilizes FAISS vector store for accurate information retrieval
- **Real-time Streaming**: Stream responses as they're generated
- **Multi-Interface Support**: 
  - CLI interface for quick interactions
  - Streamlit web interface for rich user experience
- **Conversation History**: Track and manage chat history
- **Source Attribution**: View source documents for RAG responses
- **Flexible Deployment**: Run with or without RAG capabilities

## 🛠️ Technology Stack

- **LangChain**: Framework for RAG and chain-of-thought operations
- **Groq**: Fast LLM inference engine
- **FAISS**: Vector store for efficient similarity search
- **HuggingFace**: Sentence transformers for embeddings
- **Streamlit**: Web interface
- **Python**: Core programming language

## 📋 Prerequisites

- Python 3.13+
- [Groq API Key](https://www.groq.com/)

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Smartking1/Kuve-Task-1.git
   cd KUVE-1
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv kuve
   source kuve/bin/activate  # Linux/Mac
   .\kuve\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Create a `.env` file in the project root
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

5. **Prepare the data (for RAG support)**
   ```bash
   python scripts/prepare_data.py
   ```

## 💻 Usage

### CLI Interface
```bash
python scripts/run_chatbot.py
```

Available commands:
- Type your question to chat
- `clear`: Clear conversation memory
- `history`: View conversation history
- `quit` or `exit`: Exit the chatbot

### Web Interface
```bash
streamlit run streamlit_app/app.py
```

Features:
- Chat interface with streaming responses
- Source document attribution
- Conversation history management
- Settings panel

## 📁 Project Structure

```
KUVE-1/
├── config/               # Configuration files
├── data/                # Data directory
│   ├── chat_history/    # Conversation logs
│   ├── processed/       # Processed vector store
│   └── raw/            # Raw text data
├── scripts/             # Utility scripts
│   ├── prepare_data.py  # Data preparation
│   └── run_chatbot.py  # CLI interface
├── src/                 # Source code
│   ├── chatbot.py      # Core chatbot logic
│   ├── data_loader.py  # Data loading utilities
│   └── vector_store.py # Vector store management
├── streamlit_app/      # Web interface
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## ⚙️ Configuration

Key settings in `config.py`:
- Model parameters (temperature, max tokens)
- RAG settings (chunk size, overlap)
- Conversation settings (history length)
- File paths and environment variables

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is proprietary software. All rights reserved.

## 🔗 Links

- [KUVE Website](https://getkuve.com)
- [Contact Support](mailto:info@getkuve.com)" 
