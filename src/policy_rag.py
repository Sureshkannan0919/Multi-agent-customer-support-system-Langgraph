import re
import numpy as np
import google.generativeai as genai
from langchain_core.tools import tool
from typing import List, Dict

try:
    with open("/home/suresh/Documents/ecommerce_policies_txt.md", "r", encoding='utf-8') as file:
        content = file.read()
except FileNotFoundError:
    print("Warning: Policy file not found. Using default content.")
    content = """
## Return Policy
Our return policy allows returns within 30 days of purchase.

## Shipping Policy  
We offer free shipping on orders over $50.

## Warranty Policy
All products come with a 1-year manufacturer warranty.
"""
except Exception as e:
    print(f"Error reading policy file: {e}")
    content = "Default policy content"

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", content)]

class GeminiVectorStoreRetriever:
    def __init__(self, docs: List[Dict], vectors: List, gemini_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = gemini_client

    @classmethod
    def from_docs(cls, docs: List[Dict], gemini_client):
        # Configure the Gemini API
        genai.configure(api_key='AIzaSyBYq4iNSABoFq28fAHcmplc0EZME0qDnqI')
        
        # Get embeddings using Gemini's embedding model
        model = 'models/embedding-001'  # Updated model name
        vectors = []
        
        for doc in docs:
            result = genai.embed_content(
                model=model,
                content=doc["page_content"],
                task_type="retrieval_document"
            )
            vectors.append(result['embedding'])
            
        return cls(docs, vectors, gemini_client)

    def query(self, query: str, k: int = 5) -> List[Dict]:
        # Get embedding for the query
        model = 'models/embedding-001'  # Updated model name
        query_embedding = genai.embed_content(
            model=model,
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        # Calculate similarity scores using dot product
        scores = np.array(query_embedding) @ self._arr.T
        
        # Get top k matches
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]

# Initialize the retriever
retriever = GeminiVectorStoreRetriever.from_docs(docs, genai)

@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted or not."""
    print(query)
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])
    #return "Policy information not available"

