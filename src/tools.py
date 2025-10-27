"""Tool registry for the PDF QA agent."""

from typing import List
from langchain_core.tools import tool
from src.document_store import get_document_store


@tool
def retrieve_pdf_chunks(query: str, top_k: int = 5) -> str:
    """Retrieve relevant chunks from uploaded PDF documents.

    Uses semantic search to find the most relevant text chunks from uploaded PDFs
    and returns them with citations including filename and page number.

    Args:
        query: The search query or question
        top_k: Number of top results to return (default: 5)

    Returns:
        A formatted string with retrieved chunks and citations
    """
    # Get the document store
    doc_store = get_document_store()

    # Retrieve relevant chunks with scores
    results = doc_store.retrieve_with_scores(query, top_k=top_k)

    if not results:
        return "No relevant information found in the uploaded documents."

    # Format results with citations
    formatted_chunks = []
    for i, (doc, score) in enumerate(results, start=1):
        chunk_text = doc.page_content.strip()
        source = doc.metadata.get("source", "Unknown")

        formatted_chunks.append(
            f"**[{i}] {source}** (relevance: {score:.2f})\n{chunk_text}"
        )

    return "\n\n---\n\n".join(formatted_chunks)


def get_available_tools() -> List:
    """Get the list of tools available to the agent.

    Returns:
        List of LangChain tools
    """
    return [retrieve_pdf_chunks]
