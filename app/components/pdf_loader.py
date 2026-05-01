import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdfs_from_directory():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data directory '{DATA_PATH}' does not exist.")
        loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
        documents = loader.load()
    
        if not documents:
            logger.warning(f"No PDF documents found in directory '{DATA_PATH}'.")
        else:
             logger.info(f"Loaded {len(documents)} documents from '{DATA_PATH}'")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDFs from directory: {e}")
        raise CustomException("Failed to load PDFs from directory", e)  
        return []

def split_documents(documents):
    try:
        if not documents:
            raise CustomException("No documents to split.")
        logger.info(f"Splitting {len(documents)} documents into chunks with chunk size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise CustomException("Failed to split documents", e)
        return []