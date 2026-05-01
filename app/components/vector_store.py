from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def load_vector_store():
    try:
        logger.info(f"Loading vector store from path: {DB_FAISS_PATH}")
        embedding_model = get_embedding_model()
        vector_store = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        logger.info("Vector store loaded successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise CustomException("Failed to load vector store", e)

def create_vector_store(chunks):
    try:
        if not chunks:
            raise CustomException("No document chunks provided to create vector store.")
        logger.info(f"Creating vector store with {len(chunks)} chunks.")
        embedding_model = get_embedding_model()
        vector_store = FAISS.from_documents(chunks, embedding_model)
        vector_store.save_local(DB_FAISS_PATH)
        logger.info(f"Vector store created and saved to {DB_FAISS_PATH} successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise CustomException("Failed to create vector store", e)