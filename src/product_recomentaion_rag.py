
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
import pandas as pd
import os
import logging
from typing import List, Dict, Optional
from llama_index.core.settings import Settings

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

    def load_csv_as_documents(self, csv_path: str) -> List[Document]:
        """
        Load and process CSV file into documents
        Args:
            csv_path: Path to the CSV file
        Returns:
            List of Document objects
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            df = pd.read_csv(csv_path)
            required_cols = ['name', 'description', 'price']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            documents = []
            for idx, row in df.iterrows():
                try:
                    content = (f"Product: {row['name']}\n"
                             f"Description: {row['description']}")
                    metadata = {
                        "product_id": str(row["price"]),
                        "product_name": str(row["name"])
                    }
                    doc = Document(text=content, metadata=metadata)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {str(e)}")
                    continue
            
            logger.info(f"Loaded {len(documents)} documents from CSV")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

    def create_index(self, csv_path: str):
        """
        Create search index from documents
        Args:
            csv_path: Path to the CSV file
        """
        try:
            documents = self.load_csv_as_documents(csv_path)
            if not documents:
                raise ValueError("No valid documents loaded from CSV")
            
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


indexer = ProductIndexer()

# Create index from CSV
indexer.create_index("/home/suresh/projects/skate_products.csv")


from langchain_core.tools import tool

@tool
def serach_product(query: str) -> str:
    """Search for similar products based on query."""
    try:
        print(query)
        global indexer
        if indexer is None:
            initialize_indexer()
        
        indexer.save_index("/home/suresh/projects/skate_products.faiss")
        similar_products = indexer.get_similar_products(query, top_k=2)
        
        # Format the response nicely
        result = "Product Recommendations:\n"
        for idx, product in enumerate(similar_products, 1):
            result += f"\n{idx}. {product['metadata']['product_name']}\n"
            result += f"   {product['content']}\n"
            if product.get('score'):
                result += f"   Similarity Score: {product['score']:.4f}\n"
        
        return result
    except Exception as e:
        return f"Error getting recommendations: {str(e)}"

