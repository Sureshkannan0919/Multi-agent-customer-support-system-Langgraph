try:
    from .firebase_manager import FirestoreManager
except ImportError:
    from firebase_manager import FirestoreManager
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


try:
    manager = FirestoreManager()
    manager.connection("/home/suresh/projects/shopwave-9yek6-firebase-adminsdk-fbsvc-57be3221d4.json")
except Exception as e:
    print(f"Warning: Firebase manager initialization failed: {e}")
    manager = None


@tool
def fetch_user_details(config: RunnableConfig = None) -> str:
    """Fetch user details from the database."""
    if not manager:
        return "Firebase manager not available"
    
    try:
        if config:
            print(config)
            configuration = config.get("configurable", {})
            user_id = configuration.get("uid", "default_user")
            print(user_id)
        else:
            user_id = "default_user"
        
        user_info = manager.get_user_info(user_id)
        return str(user_info)
    except Exception as e:
        return f"Error fetching user details: {str(e)}"

@tool 
def fetch_order_details(email: str) -> str:
    """Fetch order details for a given email address."""
    if not manager:
        return "Firebase manager not available"
    
    try:
        orders = manager.get_orders_by_email(email)
        return str(orders)
    except Exception as e:
        return f"Error fetching order details: {str(e)}"

print(fetch_user_details.invoke({"configurable": {"uid": "ARaZlmY5mJRaEl3JaTKxes7oSVj2"}}))


