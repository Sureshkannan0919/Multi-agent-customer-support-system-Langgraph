from ast import Dict
from .amazon_product_extract import get_amazon_product
from .flipkart_product_extract import get_flipkart_product
import concurrent.futures
import time

def search_amazon(query: str):
    """Search Amazon products."""
    try:
        return get_amazon_product(query)
    except Exception as e:
        return {"error": f"Amazon search failed: {str(e)}"}

def search_flipkart(query: str):
    """Search Flipkart products."""
    try:
        return get_flipkart_product(query)
    except Exception as e:
        return {"error": f"Flipkart search failed: {str(e)}"}


def get_product(query: str) -> str:
    """Get products from both Amazon and Flipkart in parallel and display results."""
    print(f"🔍 Searching for '{query}' on Amazon and Flipkart...")
    print("⏳ Running parallel searches...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run both searches in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        amazon_future = executor.submit(search_amazon, query)
        flipkart_future = executor.submit(search_flipkart, query)
        
        # Wait for both to complete and get results
        amazon_results = amazon_future.result()
        flipkart_results = flipkart_future.result()
    
    end_time = time.time()
    search_duration = end_time - start_time
    result={"amazon": amazon_results, "flipkart": flipkart_results}

    return str(result)

