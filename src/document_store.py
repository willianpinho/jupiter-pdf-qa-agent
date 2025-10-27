"""Document store using ChromaDB for vector storage and retrieval."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pypdf
import io

class PDFDocumentStore:
    """Manages PDF document storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the document store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        # Initialize embeddings (using free local model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            client=self.client,
            collection_name="pdf_documents",
            embedding_function=self.embeddings
        )
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_pdf(self, file_bytes: bytes, filename: str) -> Dict:
        """Extract text from PDF, chunk it, and store in ChromaDB.
        
        Args:
            file_bytes: PDF file content as bytes
            filename: Name of the PDF file
            
        Returns:
            Dict with processing stats (num_pages, num_chunks)
        """
        # Extract text from PDF
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        num_pages = len(pdf_reader.pages)
        
        # Extract text from all pages
        full_text = ""
        page_texts = []
        
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            page_texts.append((page_num, page_text))
            full_text += f"\n\n[Page {page_num}]\n{page_text}"
        
        # Split into chunks
        chunks = self.text_splitter.split_text(full_text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # Determine which page this chunk is from
            page_num = self._find_page_number(chunk, page_texts)
            
            doc = Document(
                page_content=chunk,
                metadata={
                    "filename": filename,
                    "chunk_id": i,
                    "page": page_num,
                    "source": f"{filename} (p.{page_num})"
                }
            )
            documents.append(doc)
        
        # Add to vector store
        self.vectorstore.add_documents(documents)
        
        return {
            "num_pages": num_pages,
            "num_chunks": len(chunks),
            "filename": filename
        }
    
    def _find_page_number(self, chunk: str, page_texts: List[tuple]) -> int:
        """Find which page a chunk belongs to.
        
        Args:
            chunk: Text chunk
            page_texts: List of (page_num, text) tuples
            
        Returns:
            Page number
        """
        # Look for page marker in chunk
        for page_num, _ in page_texts:
            if f"[Page {page_num}]" in chunk:
                return page_num
        
        # Fallback: return first page
        return 1
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        results = self.vectorstore.similarity_search(query, k=top_k)
        return results
    
    def retrieve_with_scores(self, query: str, top_k: int = 5) -> List[tuple]:
        """Retrieve relevant chunks with similarity scores.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return results


# Global document store instance
_document_store = None


def get_document_store() -> PDFDocumentStore:
    """Get or create the global document store instance.

    Returns:
        PDFDocumentStore instance
    """
    global _document_store
    if _document_store is None:
        _document_store = PDFDocumentStore()
    return _document_store