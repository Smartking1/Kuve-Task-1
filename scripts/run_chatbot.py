import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import LangChainChatbot
from src.vector_store import LangChainVectorStore
from config.config import VECTOR_STORE_PATH, CHATBOT_NAME, DOMAIN


def print_welcome():
    """Print welcome message"""
    print("\n" + "=" * 60)
    print(f"  Welcome to {CHATBOT_NAME} - {DOMAIN} Assistant")
    print("  Powered by Kuve")
    print("=" * 60)
    print("\nCommands:")
    print("  - Type your question to chat")
    print("  - Type 'clear' to clear conversation memory")
    print("  - Type 'history' to view conversation history")
    print("  - Type 'quit' or 'exit' to exit")
    print("\n" + "=" * 60 + "\n")


def main():
    """
    Main function to run CLI chatbot
    """
    print("Initializing chatbot...")
    
    # Load vector store
    vector_store_manager = LangChainVectorStore(store_path=VECTOR_STORE_PATH)
    
    if vector_store_manager.load():
        print("✓ Vector store loaded successfully")
        vectorstore = vector_store_manager.get_vectorstore()
        chatbot = LangChainChatbot(vectorstore=vectorstore)
        rag_enabled = True
    else:
        print("⚠ Vector store not found. Running without RAG.")
        print("  Run 'python scripts/prepare_data.py' to enable RAG.")
        chatbot = LangChainChatbot(vectorstore=None)
        rag_enabled = False
    
    print("✓ Chatbot ready\n")
    
    print_welcome()
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input(f"You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print(f"\nThank you for using {CHATBOT_NAME}. Goodbye!\n")
                break
            
            elif user_input.lower() == 'clear':
                chatbot.clear_memory()
                continue
            
            elif user_input.lower() == 'history':
                history = chatbot.get_chat_history()
                if not history:
                    print("No conversation history yet.\n")
                else:
                    print("\n--- Conversation History ---")
                    for entry in history:
                        print(entry)
                    print("--- End of History ---\n")
                continue
            
            # Generate response with streaming
            print(f"\n{CHATBOT_NAME}: ", end="", flush=True)
            
            for chunk in chatbot.chat_stream(user_input):
                print(chunk, end="", flush=True)
            
            print("\n")  # New line after response
            
        except KeyboardInterrupt:
            print(f"\n\nThank you for using {CHATBOT_NAME}. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


if __name__ == "__main__":
    main()