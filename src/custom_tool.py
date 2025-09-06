try:
    from .firebase_manager import FirestoreManager
except ImportError:
    from firebase_manager import FirestoreManager
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
try:
    from .product_search import get_product
except ImportError:
    from product_search import get_product

try:
    manager = FirestoreManager()
    manager.connection("cred.json")
except Exception as e:
    print(f"Warning: Firebase manager initialization failed: {e}")
    manager = None


@tool
def fetch_user_details(config: RunnableConfig = None) -> dict:
    """Fetch user details from the database."""
    if not manager:
        return "Firebase manager not available"
    
    try:
        if config:
            configuration = config.get("configurable", {})
            user_id = configuration.get("uid", "default_user")
        else:
            user_id = "default_user"
        
        user_info = manager.get_user_info(user_id)
        return user_info
    except Exception as e:
        return f"Error fetching user details: {str(e)}"

@tool 
def fetch_order_details(config: RunnableConfig = None) -> dict:
    """Fetch order details from the database."""
    if not manager:
        return "Firebase manager not available"

    try:
        if config:
            configuration = config.get("configurable", {})
            uid = configuration.get("uid", "default_user")
        else:
            uid = "default_user"

        orders = manager.get_orders(uid)
        return orders
    except Exception as e:
        return f"Error fetching order details: {str(e)}"
    """Fetch order details for a given uid."""


@tool
def search_product_online(search_term: str) -> str:
    """Search for a simiar product in amazon and flikart."""
    try:
        products = get_product(search_term)
        return products

    except Exception as e:
        return f"Error searching for products: {str(e)}"

