"""
Utilities for processing uploaded files.
"""
import os
import tempfile
import shutil
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from chainlit.types import AskFileResponse

import config

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
    separators=config.SEPARATORS
)

def process_file(file: AskFileResponse):
    """
    Process an uploaded file and split it into text chunks.
    
    Args:
        file: The uploaded file response from Chainlit
        
    Returns:
        List of document chunks
    """
    print(f"Processing file: {file.name}")
    
    # Create a temporary file with the correct extension
    suffix = f".{file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        # Copy the uploaded file content to the temporary file
        shutil.copyfile(file.path, temp_file.name)
        print(f"Created temporary file at: {temp_file.name}")
        
        try:
            # Create appropriate loader
            if file.name.lower().endswith('.pdf'):
                loader = PyPDFLoader(temp_file.name)
            else:
                loader = TextLoader(temp_file.name)
                
            # Load and process the documents
            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            return texts
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")