
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import LangChainDataLoader
from src.vector_store import LangChainVectorStore
from config.config import RAW_DATA_PATH, VECTOR_STORE_PATH


def main():
    """
    Main function to prepare data using LangChain
    """
    print("=" * 60)
    print("DATA PREPARATION PIPELINE (LangChain)")
    print("=" * 60)
    
    # Step 1: Load and split documents
    print("\n[Step 1/2] Loading and splitting documents...")
    data_loader = LangChainDataLoader(data_path=RAW_DATA_PATH)
    documents = data_loader.load_and_split()
    
    if not documents:
        print("\n❌ No documents found!")
        print("Please add text or CSV files to the data/raw/ directory.")
        return
    
    print(f"\n✓ Processed {len(documents)} document chunks")
    
    # Step 2: Create vector store
    print("\n[Step 2/2] Creating vector store with embeddings...")
    vector_store = LangChainVectorStore(store_path=VECTOR_STORE_PATH)
    vector_store.create_vectorstore(documents)
    
    # Save vector store
    print("\nSaving vector store...")
    vector_store.save()
    
    # Summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE ✓")
    print("=" * 60)
    print(f"Documents processed: {len(documents)}")
    print(f"Vector store location: {VECTOR_STORE_PATH}")
    print("\n✓ You can now run the chatbot!")
    print("  CLI: python scripts/run_chatbot.py")
    print("  Web: streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()