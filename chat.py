import os
from typing import List, Any
import numpy as np
import gradio as gr
from pydantic import Field, BaseModel
from rank_bm25 import BM25Okapi

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain.schema import BaseRetriever
from langchain.schema.document import Document

from models import Models

# Initialize environment and models
models = Models()
embeddings = models.embeddings_openai
llm = models.model_openrouter

# Choose whether to use hybrid retrieval (BM25 + Vector) or simple retrieval (Vector)
USE_HYBRID = True

# Initialize vector store for document storage and retrieval
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",
)

# Define the system prompt template for the AI assistant
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant analyzing DC housing policy documents. Your role is to:
1. Provide comprehensive and accurate answers using information from the provided documents
2. Include specific numbers, dates, and details when available
3. For factual questions, cite specific figures and timeframes
4. When answering questions about funding or statistics, break down the numbers by year or category
5. Maintain clarity while being thorough in your responses
6. Only use information from the provided documents"""),
    
    ("human", """Please answer this question: {input}

Context: {context}

Please provide only a detailed and concise answer to the question without citing sources (sources will be automatically added below):

Answer: [Detailed and concise answer to the question]""")
])


class HybridRetriever(BaseRetriever):
    """
    Custom retriever that combines vector similarity search with BM25 keyword search
    for improved document retrieval accuracy
    """
    vector_store: Any = Field(description="Vector store for hybrid search")
    k: int = Field(default=10, description="Number of documents to return")
    docs: List[str] = Field(default_factory=list, description="List of documents")
    bm25: Any = Field(default=None, description="BM25 search instance")

    def __init__(self, vector_store, k=10, **kwargs):
        super().__init__(vector_store=vector_store, k=k, **kwargs)
        
        # Initialize BM25 with documents from vector store
        results = self.vector_store.get()
        self.docs = results['documents']
        self.bm25 = BM25Okapi([doc.split() for doc in self.docs])
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents using both vector and BM25 search
        Returns combined and ranked results
        """
        # Perform vector similarity search
        vector_results = self.vector_store.similarity_search_with_score(query, k=self.k)
        
        # Perform BM25 keyword search
        bm25_scores = self.bm25.get_scores(query.split())
        
        # Normalize scores from both methods
        vector_scores = np.array([score for _, score in vector_results])
        if len(vector_scores) > 0:
            vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min())
        
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        
        # Combine scores with weights (80% vector, 20% BM25)
        final_scores = []
        for i, (doc, v_score) in enumerate(vector_results):
            combined_score = 0.8 * v_score + 0.2 * bm25_scores[i]  # Adjusted weights if needed
            final_scores.append((doc, combined_score))
        
        # Return top k documents
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in final_scores[:self.k]]


# Initialize both retrieval chains
def initialize_chains():
    """
    Initialize both hybrid and simple retrieval chains
    Returns both chains for comparison
    """
    # Hybrid retrieval chain (current implementation)
    hybrid_retriever = HybridRetriever(vector_store)
    hybrid_chain = create_retrieval_chain(
        hybrid_retriever, 
        create_stuff_documents_chain(llm, prompt)
    )

    # Simple retrieval chain (using default similarity search)
    simple_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    simple_chain = create_retrieval_chain(
        simple_retriever, 
        create_stuff_documents_chain(llm, prompt)
    )

    return hybrid_chain, simple_chain

# Initialize both chains
hybrid_chain, simple_chain = initialize_chains()

# Set which chain to use (can be toggled between hybrid and simple)
retrieval_chain = hybrid_chain if USE_HYBRID else simple_chain


def format_sources(source_docs):
    """
    Enhanced source formatting with better page extraction and metadata handling
    """
    sources_dict = {}
    
    for doc in source_docs:
        # Extract filename without path
        full_path = doc.metadata['source']
        filename = os.path.basename(full_path)
        
        # Get page number and content snippet
        page = doc.metadata['page']
        content_preview = doc.page_content[:150] + "..."  # Optional: add content preview
        
        # Group by filename
        if filename in sources_dict:
            if page not in sources_dict[filename]['pages']:
                sources_dict[filename]['pages'].append(page)
                sources_dict[filename]['content'].append(content_preview)
        else:
            sources_dict[filename] = {
                'pages': [page],
                'content': [content_preview]
            }
    
    # Format output
    formatted_sources = []
    for i, (filename, info) in enumerate(sources_dict.items(), 1):
        if i > 4:  # limit to top 4 sources
            break
        pages = sorted(info['pages'])
        pages_str = ", ".join(str(page) for page in pages)
        formatted_sources.append(f"{i}. File: {filename}\n   Page(s): {pages_str}")
        # Optional: Add content preview
        # formatted_sources.append(f"   Preview: {info['content'][0]}")
    
    return formatted_sources

def get_response(query: str) -> str:
    """
    Process user query and return formatted response with sources
    """
    retriever_type = "Hybrid （BM25 + Vector）" if USE_HYBRID else "Simple (Vector)"
    result = retrieval_chain.invoke({"input": query})
    source_docs = result.get("context", [])
    response = f"[Using {retriever_type} Retriever]\n\n{result['answer']}\n\nSources:\n"
    formatted_sources = format_sources(source_docs)
    response += "\n".join(formatted_sources)
    return response

def chat_response(message, history):
    """
    Handle chat messages for Gradio interface
    """
    if message.lower().strip() == 'q':
        return """Closing chat... 
        <script>
        setTimeout(function() {
            window.close();
            document.body.innerHTML = "You can now close this window.";
        }, 1000);
        </script>
        """
    return get_response(message)

def main():
    """
    Main function to run either command line or graphical interface
    """
    print("Choose interface:")
    print("1. Command Line")
    print("2. Graphical Interface")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        # Command line interface
        while True:
            query = input("User (or type 'q' to end): ")
            if query.lower() == 'q':
                break
            result = get_response(query)
            print("\nAssistant:", result, "\n")
    else:
        # Graphical interface
        demo = gr.ChatInterface(
            chat_response,
            title="DC Housing Policy Assistant",
            description="Ask questions about DC housing policy documents (type 'q' to quit)",
            theme="soft",
            examples=[
                "What are the main housing affordability challenges in DC?",
                "What portion of DC's Housing Production Trust Fund is legally required to support the lowest-income residents?"
            ]
        )
        demo.launch(show_api=False, share=True)

if __name__ == "__main__":
    main()

