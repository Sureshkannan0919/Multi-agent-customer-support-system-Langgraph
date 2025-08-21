import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")


class FirestoreManager:
    def __init__(self):
        self.db = None
        self.current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        self.current_user = None

    def connection(self, api_key_json, current_user="Anonymous"):
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(api_key_json)
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            self.current_user = current_user
            print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {self.current_time}")
            print(f"Current User's Login: {self.current_user}")
            return "Connection successful"
        except Exception as e:
            return f"Connection failed: {str(e)}"

    def show_collections(self):
        try:
            return [collection.id for collection in self.db.collections()]
        except Exception as e:
            return f"Error listing collections: {str(e)}"

    def show_columns(self, collection_name):
        try:
            docs = list(self.db.collection(collection_name).limit(1).stream())
            if docs:
                return list(docs[0].to_dict().keys())
            return []
        except Exception as e:
            return f"Error getting columns: {str(e)}"

    def show_data(self, collection_name):
        try:
            docs = list(self.db.collection(collection_name).stream())
            data = [doc.to_dict() for doc in docs]
            return pd.DataFrame(data)
        except Exception as e:
            return f"Error getting data: {str(e)}"

    def get_user_info(self, uid) -> dict:
        try:
            user_ref = self.db.collection('users').document(uid)
            user_doc = user_ref.get()

            if user_doc.exists:
                user_data = user_doc.to_dict()
                return {
                        'username': user_data.get('name', 'Not found'),
                        'email': user_data.get('email', 'Not found') }
            else:
                return {
                    'status': 'error',
                    'message': f'User with UID {uid} not found',
                    'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }

    def get_orders_by_email(self, email) -> dict:
        try:
            orders_ref = self.db.collection('orders')
            query = orders_ref.where('customer.email', '==', email)
            docs = list(query.stream())
            
            if not docs:
                return {
                    'status': 'error',
                    'message': f'No orders found for email: {email}',
                    'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                }

            orders_data = []
            for doc in docs:
                order_dict = doc.to_dict()
                flat_order = {
                    'order_id': doc.id,
                    'created_at': order_dict.get('createdAt', ''),
                    'total': order_dict.get('total', 0),
                    'status': order_dict.get('status', ''),
                    'items': order_dict.get('items', []),
                    'customer_name': order_dict.get('customer', {}).get('name', ''),
                    'customer_email': order_dict.get('customer', {}).get('email', ''),
                    'customer_mobile': order_dict.get('customer', {}).get('mobile', ''),
                    'address_flat': order_dict.get('customer', {}).get('address', {}).get('flat', ''),
                    'address_street': order_dict.get('customer', {}).get('address', {}).get('street', ''),
                    'address_city': order_dict.get('customer', {}).get('address', {}).get('city', ''),
                    'address_state': order_dict.get('customer', {}).get('address', {}).get('state', ''),
                    'address_zip': order_dict.get('customer', {}).get('address', {}).get('zip', ''),
                    'address_landmark': order_dict.get('customer', {}).get('address', {}).get('landmark', '')
                }
                orders_data.append(flat_order)

            
            return orders_data

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }

    def close_connection(self):
        """
        Close the Firestore connection
        :return: Dictionary with closure status and timestamp
        """
        try:
            if firebase_admin._apps:
                for app in firebase_admin._apps.values():
                    firebase_admin.delete_app(app)
                self.db = None
                close_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\nConnection closed at: {close_time}")
                print(f"User session ended: {self.current_user}")
                return {
                    'status': 'success',
                    'message': 'Connection closed successfully',
                    'timestamp': close_time,
                    'user': self.current_user
                }
            return {
                'status': 'warning',
                'message': 'No active connection to close',
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error closing connection: {str(e)}',
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }


