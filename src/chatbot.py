from typing import Optional, List
import os
from datetime import datetime
import json

from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langgraph.checkpoint.memory import InMemorySaver  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from config.config import (
    GROQ_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_K_RESULTS,
    MAX_CONVERSATION_HISTORY,
    MEMORY_KEY,
    CHATBOT_NAME,
    DOMAIN,
    CHAT_HISTORY_PATH
)


class LangChainChatbot:
    """
    Domain-specific chatbot using LangChain with RAG
    """
    
    def __init__(self, vectorstore=None, enable_logging=True):
        """
        Initialize LangChain chatbot
        
        Args:
            vectorstore: FAISS vector store instance
            enable_logging (bool): Enable conversation logging
        """
        self.vectorstore = vectorstore
        self.enable_logging = enable_logging
        self.chatbot_name = CHATBOT_NAME
        self.domain = DOMAIN
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Initialize conversation memory
        self.memory = InMemorySaver()
        self.chat_history = []
        
        # Create prompt template
        self.qa_prompt = self._create_prompt_template()
        
        # Initialize chain
        self.chain = None
        if vectorstore is not None:
            self._create_chain()
    
    def _create_prompt_template(self, messages=None) -> ChatPromptTemplate:
        """
        Create custom prompt template for the chatbot
        
        Returns:
            ChatPromptTemplate: Custom prompt template
        """
        system_message = f"""You are {self.chatbot_name}, a helpful assistant specialized in {self.domain}.
Answer the question based on the provided context and conversation history.
If you don't know the answer based on the context, say so honestly.
Be concise, accurate, and helpful.

Context:
{{context}}

Chat History:
{{chat_history}}"""
        
        return ChatPromptTemplate(
            [
                ("system", system_message),
                ("human", "{question}"),
            ]
        )
    
    def _create_chain(self):
        """
        Create conversational retrieval chain
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not provided")
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": TOP_K_RESULTS}
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            combine_docs_chain_kwargs={
                "prompt": self.qa_prompt
            },
            return_source_documents=True,
            verbose=False
        )
    
    def set_vectorstore(self, vectorstore):
        """
        Set or update the vector store
        
        Args:
            vectorstore: FAISS vector store instance
        """
        self.vectorstore = vectorstore
        self._create_chain()
    
    def chat(self, question: str) -> dict:
        """
        Generate response to user question
        
        Args:
            question (str): User's question
            
        Returns:
            dict: Response with answer and source documents
        """
        if self.chain is None:
            # Fallback to direct LLM if no RAG
            response = self.llm.invoke(question)
            result = {
                'answer': response.content,
                'source_documents': [],
                'question': question
            }
        else:
            # Use RAG chain with chat history
            result = self.chain.invoke({
                "question": question,
                "chat_history": self.chat_history
            })
            
            # Update chat history
            self.chat_history.append((question, result['answer']))
            
            # Keep only recent history
            if len(self.chat_history) > MAX_CONVERSATION_HISTORY:
                self.chat_history = self.chat_history[-MAX_CONVERSATION_HISTORY:]
        
        # Log conversation
        if self.enable_logging:
            self._log_conversation(
                question=question,
                answer=result['answer'],
                sources=result.get('source_documents', [])
            )
        
        return result
    
    def chat_stream(self, question: str):
        """
        Stream response
        
        Args:
            question (str): User's question
            
        Yields:
            str: Response chunks
        """
        if self.chain is None:
            # No RAG - direct streaming
            for chunk in self.llm.stream(question):
                yield chunk.content
        else:
            # With RAG - get context first, then stream
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": TOP_K_RESULTS}
            )
            docs = retriever.invoke(question)
            
            # Build context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format chat history
            history_text = ""
            for q, a in self.chat_history[-MAX_CONVERSATION_HISTORY:]:
                history_text += f"User: {q}\n{self.chatbot_name}: {a}\n\n"
            
            # Build prompt manually for streaming
            prompt_text = f"""You are {self.chatbot_name}, a helpful assistant specialized in {self.domain}.
Answer the question based on the provided context and conversation history.
If you don't know the answer based on the context, say so honestly.
Be concise, accurate, and helpful.

Context:
{context}

Chat History:
{history_text}

Question: {question}

Answer:"""
            
            # Stream response
            full_response = ""
            for chunk in self.llm.stream(prompt_text):
                content = chunk.content
                full_response += content
                yield content
            
            # Update chat history
            self.chat_history.append((question, full_response))
            
            # Keep only recent history
            if len(self.chat_history) > MAX_CONVERSATION_HISTORY:
                self.chat_history = self.chat_history[-MAX_CONVERSATION_HISTORY:]
            
            # Log
            if self.enable_logging:
                self._log_conversation(question, full_response, docs)
    
    def _log_conversation(self, question: str, answer: str, sources: List[Document]):
        """
        Log conversation to file
        
        Args:
            question (str): User's question
            answer (str): Bot's answer
            sources (List[Document]): Source documents
        """
        os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)
        
        log_file = os.path.join(
            CHAT_HISTORY_PATH,
            f"chat_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'num_sources': len(sources),
            'sources': [
                {
                    'content': doc.page_content[:200],
                    'metadata': doc.metadata
                }
                for doc in sources
            ]
        }
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def clear_memory(self):
        """
        Clear conversation memory
        """
        self.chat_history = []
        print("✓ Conversation memory cleared")
    
    def get_memory(self) -> dict:
        """
        Get conversation memory
        
        Returns:
            dict: Memory contents
        """
        return {MEMORY_KEY: self.chat_history}
    
    def get_chat_history(self) -> List[str]:
        """
        Get formatted chat history
        
        Returns:
            List[str]: List of conversation exchanges
        """
        history = []
        for question, answer in self.chat_history:
            history.append(f"User: {question}")
            history.append(f"{self.chatbot_name}: {answer}")
        
        return history