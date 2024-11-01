import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from models import Models

load_dotenv()

# Initialize the models
models = Models()
embeddings = models.embeddings_openai

# Define constants
data_folder = "./data"
chunk_size = 1000
chunk_overlap = 200
check_interval = 10

# Chroma vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",  # Where to save data locally
)

# Ingest a file
def ingest_file(file_path):
    """
    Process a single PDF file and add its contents to the vector store
    
    Args:
        file_path: Path to the PDF file to process
    """
    if not file_path.lower().endswith('.pdf'):
        print(f"Skipping non-PDF file: {file_path}")
        return
    
    print(f"Starting to ingest file: {file_path}")
    
    # Load PDF and its documents
    loader = PyPDFLoader(file_path)
    loaded_documents = loader.load()
    
    # Add source metadata to original documents
    for doc in loaded_documents:
        doc.metadata['source'] = os.path.basename(file_path)
        # Page numbers in PyPDFLoader are 0-based, so add 1 for human-readable format
        doc.metadata['page'] = doc.metadata.get('page', 0) + 1
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    documents = text_splitter.split_documents(loaded_documents)
    
    # Ensure source metadata and page numbers are preserved in all chunks
    for doc in documents:
        doc.metadata['source'] = os.path.basename(file_path)
        # Ensure page number is human-readable (1-based)
        doc.metadata['page'] = doc.metadata.get('page', 0) + 1
    
    # Generate unique IDs for each chunk
    uuids = [str(uuid4()) for _ in range(len(documents))]
    
    print(f"Adding {len(documents)} documents to the vector store")
    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"Finished ingesting file: {file_path}")

# Main loop
def main_loop():
    while True:
        for filename in os.listdir(data_folder):
            if not filename.startswith("_"):
                file_path = os.path.join(data_folder, filename)
                ingest_file(file_path)
                new_filename = "_" + filename
                new_file_path = os.path.join(data_folder, new_filename)
                os.rename(file_path, new_file_path)
        time.sleep(check_interval)  # Check the folder every 10 seconds

# Run the main loop
if __name__ == "__main__":
    main_loop()
