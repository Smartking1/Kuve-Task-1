import os
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.config import VECTOR_STORE_PATH, EMBEDDING_MODEL


class LangChainVectorStore:
    """
    Manage FAISS vector store using LangChain
    """
    
    def __init__(self, embedding_model_name=None, store_path=None):
        """
        Initialize vector store
        
        Args:
            embedding_model_name (str): Name of embedding model
            store_path (str): Path to save/load vector store
        """
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL
        self.store_path = store_path or VECTOR_STORE_PATH
        self.vectorstore = None
        
        # Initialize embeddings
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embedding model loaded")
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create vector store from documents
        
        Args:
            documents (List[Document]): List of documents to index
            
        Returns:
            FAISS: Created vector store
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"\nCreating vector store from {len(documents)} documents...")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print("✓ Vector store created")
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to existing vector store
        
        Args:
            documents (List[Document]): Documents to add
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        self.vectorstore.add_documents(documents)
        print(f"✓ Added {len(documents)} documents to vector store")
    
    def save(self, path: Optional[str] = None):
        """
        Save vector store to disk
        
        Args:
            path (str): Path to save vector store
        """
        if self.vectorstore is None:
            raise ValueError("No vector store to save")
        
        save_path = path or self.store_path
        os.makedirs(save_path, exist_ok=True)
        
        self.vectorstore.save_local(save_path)
        print(f"✓ Vector store saved to: {save_path}")
    
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load vector store from disk
        
        Args:
            path (str): Path to load vector store from
            
        Returns:
            bool: True if successful, False otherwise
        """
        load_path = path or self.store_path
        
        if not os.path.exists(load_path):
            print(f"✗ Vector store not found at: {load_path}")
            return False
        
        try:
            self.vectorstore = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✓ Vector store loaded from: {load_path}")
            return True
        except Exception as e:
            print(f"✗ Error loading vector store: {e}")
            return False
    
    def get_vectorstore(self) -> Optional[FAISS]:
        """
        Get the vector store instance
        
        Returns:
            FAISS: Vector store instance or None
        """
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            
        Returns:
            List[Document]: Similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4):
        """
        Search with similarity scores
        
        Args:
            query (str): Query text
            k (int): Number of results
            
        Returns:
            List[Tuple[Document, float]]: Documents with scores
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)