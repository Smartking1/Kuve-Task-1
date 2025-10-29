
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader, CSVLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.config import RAW_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP


class LangChainDataLoader:
    """
    Load and process documents using LangChain loaders
    """
    
    def __init__(self, data_path=None, chunk_size=None, chunk_overlap=None):
        """
        Initialize data loader
        
        Args:
            data_path (str): Path to raw data directory
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        """
        self.data_path = data_path or RAW_DATA_PATH
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_text_files(self) -> List[Document]:
        """
        Load all text files from directory
        
        Returns:
            List[Document]: List of LangChain documents
        """
        documents = []
        data_dir = Path(self.data_path)
        
        if not data_dir.exists():
            print(f"Data directory not found: {self.data_path}")
            return documents
        
        # Load text and markdown files
        for file_path in data_dir.glob("*.txt"):
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")
        
        for file_path in data_dir.glob("*.md"):
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")
        
        return documents
    
    def load_csv_files(self) -> List[Document]:
        """
        Load all CSV files from directory
        
        Returns:
            List[Document]: List of LangChain documents
        """
        documents = []
        data_dir = Path(self.data_path)
        
        if not data_dir.exists():
            return documents
        
        for file_path in data_dir.glob("*.csv"):
            try:
                loader = CSVLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")
        
        return documents
    
    def load_all_documents(self) -> List[Document]:
        """
        Load all documents from the data directory
        
        Returns:
            List[Document]: List of all loaded documents
        """
        print(f"Loading documents from: {self.data_path}")
        
        documents = []
        documents.extend(self.load_text_files())
        documents.extend(self.load_csv_files())
        
        print(f"\nTotal documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents (List[Document]): Documents to split
            
        Returns:
            List[Document]: Split document chunks
        """
        print(f"\nSplitting documents into chunks...")
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def load_and_split(self) -> List[Document]:
        """
        Load all documents and split into chunks
        
        Returns:
            List[Document]: List of document chunks
        """
        documents = self.load_all_documents()
        
        if not documents:
            print("⚠ No documents found!")
            return []
        
        chunks = self.split_documents(documents)
        return chunks