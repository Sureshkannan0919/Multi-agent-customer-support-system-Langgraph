
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
import pandas as pd
import os
import logging
from typing import List, Dict, Optional
from llama_index.core.settings import Settings
try:
    from .firebase_manager import FirestoreManager
except ImportError:
    from firebase_manager import FirestoreManager

base_dir = os.path.dirname(os.path.abspath(__file__))

try:
    manager = FirestoreManager()
    manager.connection("cred.json")
except Exception as e:
    print(f"Warning: Firebase manager initialization failed: {e}")
    manager = None
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# Disable default LLM to avoid OpenAI key requirement
Settings.llm = None

class ProductIndexer:
    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Product Indexer
        Args:
            embed_model_name: Name of the HuggingFace embedding model to use
        """
        self.dimension = 384  # Embedding dimension for the chosen model
        self.embed_model_name = embed_model_name
        self.setup_system()

    def setup_system(self):
        """Initialize embedding model and vector store"""
        try:
            # Setup embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embed_model_name
            )
            
            # Initialize FAISS vector store
            self.faiss_index = faiss.IndexFlatL2(self.dimension)
            self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
            
            logger.info("System initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up system: {str(e)}")
            raise

    def load_csv_as_documents(self, df) -> List[Document]:
        """
        Load and process DataFrame into documents
        Args:
            df: DataFrame containing product data
        Returns:
            List of Document objects
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is None or empty")
            
            required_cols = ['name','category', 'description', 'price']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
            documents = []
            for idx, row in df.iterrows():
                try:
                    content = (f"Product: {row['name']}\n"
                             f"Category: {row['category']}\n"
                             f"Description: {row['description']}\n"
                             f"Price: {row['price']}")
                    metadata = {
                        "product_id": str(idx),  # Use index as ID instead of price
                        "product_name": str(row["name"]),
                        "category": str(row["category"]),
                        "price": str(row["price"])
                    }
                    doc = Document(text=content, metadata=metadata)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {str(e)}")
                    continue
            
            logger.info(f"Loaded {len(documents)} documents from DataFrame")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading DataFrame: {str(e)}")
            raise

    def create_index(self, df):
        """
        Create search index from documents
        Args:
            csv_path: Path to the CSV file
        """
        try:
            documents = self.load_csv_as_documents(df)
            if not documents:
                raise ValueError("No valid documents loaded from df")
            
            self.index = VectorStoreIndex.from_documents(
                documents,
                vector_store=self.vector_store,
                embed_model=self.embed_model
            )
            
            # Create basic query engine without LLM (avoid OpenAI default)
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=2,
                llm=None
            )
            
            logger.info("Index created successfully")
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise

    def save_index(self, save_path: str):
        """
        Save the FAISS index to disk
        Args:
            save_path: Path to save the index
        """
        try:
            faiss.write_index(self.faiss_index, save_path)
            logger.info(f"Index saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

    def load_index(self, load_path: str):
        """
        Load a saved FAISS index from disk
        Args:
            load_path: Path to the saved index
        """
        try:
            self.faiss_index = faiss.read_index(load_path)
            self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
            
            # Recreate the index and query engine
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
            
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=2,
                llm=None
            )
            
            logger.info(f"Index loaded from {load_path}")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise

    def get_similar_products(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Get similar products based on query
        Args:
            query: Search query
            top_k: Number of similar products to return
        Returns:
            List of similar products with metadata
        """
        try:
            if not hasattr(self, 'query_engine'):
                raise ValueError("Index not created. Call create_index first.")
            
            response = self.query_engine.query(query)
            
            # Extract source nodes and metadata
            similar_products = []
            for node in response.source_nodes:
                similar_products.append({
                    'content': node.text,
                    'metadata': node.metadata,
                    'score': node.score if hasattr(node, 'score') else None
                })
            
            return similar_products[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting similar products: {str(e)}")
            raise



# Global variables
indexer = None
df = None
faiss_index_path = os.path.join(base_dir, "rag_index", "skate_products.faiss")

def initialize_indexer():
    """Initialize the indexer with proper error handling"""
    global indexer, df
    
    try:
        if indexer is None:
            indexer = ProductIndexer()
            
        if df is None:
            df = manager.show_data("products")
            
        # Ensure the rag_index directory exists
        os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
        
        return True
    except Exception as e:
        logger.error(f"Error initializing indexer: {str(e)}")
        return False

def route_index():
    """Route to either load existing index or create new one"""
    global indexer, df
    
    if not initialize_indexer():
        raise Exception("Failed to initialize indexer")
    
    try:
        if os.path.exists(faiss_index_path):
            logger.info("Loading existing index...")
            try:
                indexer.load_index(faiss_index_path)
            except Exception as load_error:
                logger.warning(f"Failed to load existing index: {str(load_error)}")
                logger.info("Deleting corrupted index and creating new one...")
                # Delete corrupted index
                if os.path.exists(faiss_index_path):
                    os.remove(faiss_index_path)
                # Create new index
                indexer.create_index(df)
                indexer.save_index(faiss_index_path)
        else:
            logger.info("Creating new index...")
            indexer.create_index(df)
            indexer.save_index(faiss_index_path)
        
        return indexer
    except Exception as e:
        logger.error(f"Error in route_index: {str(e)}")
        raise


from langchain_core.tools import tool

@tool
def search_product(query: str) -> str:
    """Search for similar products based on query."""
    try:
        product_indexer = route_index()
        
        if not product_indexer:
            return "Error: Could not initialize product indexer"
        
        similar_products = product_indexer.get_similar_products(query, top_k=2)
        
        if not similar_products:
            return f"No products found matching '{query}'. Please try a different search term."
        
        # Format the response nicely
        result = f"Product Recommendations for '{query}':\n"
        for idx, product in enumerate(similar_products, 1):
            result += f"\n{idx}. {product['metadata']['product_name']}\n"
            result += f"   Category: {product['metadata'].get('category', 'N/A')}\n"
            result += f"   Price: {product['metadata'].get('price', 'N/A')}\n"
            result += f"   {product['content']}\n"
        
        return result
    except Exception as e:
        logger.error(f"Error in search_product: {str(e)}")
        return f"Error getting recommendations: {str(e)}"

