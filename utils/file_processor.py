"""
Utilities for processing uploaded files.
"""
import os
import tempfile
import shutil
from typing import List, Optional
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    UnstructuredExcelLoader,
    Docx2txtLoader
)
from langchain_core.documents import Document
from chainlit.types import AskFileResponse

import config

def get_document_loader(file_path: str):
    """
    Get appropriate document loader based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Document loader instance
    """
    file_extension = Path(file_path).suffix.lower()
    
    # Select appropriate loader based on file extension
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension == '.txt' or file_extension == '.md' or file_extension == '.py':
        return TextLoader(file_path)
    elif file_extension == '.csv':
        return CSVLoader(file_path)
    elif file_extension == '.xlsx' or file_extension == '.xls':
        return UnstructuredExcelLoader(file_path)
    elif file_extension == '.docx' or file_extension == '.doc':
        return Docx2txtLoader(file_path)
    else:
        # Default to text loader
        return TextLoader(file_path)

def create_text_splitter():
    """
    Create a text splitter with the configured settings.
    
    Returns:
        Initialized text splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=config.SEPARATORS
    )

def process_file(file: AskFileResponse) -> Optional[List[Document]]:
    """
    Process an uploaded file and split it into text chunks.
    
    Args:
        file: The uploaded file response from Chainlit
        
    Returns:
        List of document chunks or None if processing fails
    """
    print(f"Processing file: {file.name}")
    
    # Create a temporary file with the correct extension
    suffix = f".{file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        try:
            # Copy the uploaded file content to the temporary file
            shutil.copyfile(file.path, temp_file.name)
            print(f"Created temporary file at: {temp_file.name}")
            
            # Get the appropriate loader
            loader = get_document_loader(temp_file.name)
                
            # Load documents
            documents = loader.load()
            
            # Initialize text splitter
            text_splitter = create_text_splitter()
            
            # Split documents into chunks
            texts = text_splitter.split_documents(documents)
            
            return texts
        except Exception as e:
            print(f"Error processing file: {e}")
            return None
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")