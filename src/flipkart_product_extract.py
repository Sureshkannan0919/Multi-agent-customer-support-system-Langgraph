from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import warnings
warnings.filterwarnings('ignore')


def get_flipkart_search_html(search_term: str, silent: bool = False) -> str:
    """
    Use Selenium to scrape Flipkart search results and return HTML source.
    
    Args:
        search_term: The search query for Flipkart
        silent: If True, suppress output messages
        
    Returns:
        str: HTML source of the search results page
    """
    
    # Set up Chrome options for silent operation
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode for faster execution
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Try to use system Chromium if available
    try:
        chrome_options.binary_location = "/usr/bin/chromium"
        service = Service()
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        if not silent:
            print(f"Searching for: {search_term}...")
        
        driver.get("https://www.flipkart.in/")
        time.sleep(2)
        
        search_box = driver.find_element(By.NAME, "q")
        search_box.clear()
        search_box.send_keys(search_term)
        search_box.send_keys(Keys.RETURN)
        time.sleep(4)
        
        return driver.page_source
        
    except Exception as e:
        if not silent:
            print(f"Error: {e}")
        return ""
    
    finally:
        driver.quit()


def extract_single_product_data(product_card) -> Dict:
    """Extract product data from a single Flipkart product card element."""
    
    product_data = {
        'product_id': None,
        'product_description': None,
        'brand': None,
        'current_price': None,
        'rating': None,
        'number_of_reviews': None,
        'image_src': None,
        'product_url': None,
    }
    
    try:
        # Extract product ID from data-id attribute
        product_data['product_id'] = product_card.get('data-id')
        
        # Extract product description/title
        title_link = product_card.find('a', class_='wjcEIp')
        if title_link:
            product_data['product_description'] = title_link.get('title', '').strip()
            product_data['product_url'] = title_link.get('href', '')
            if product_data['product_url'] and not product_data['product_url'].startswith('http'):
                product_data['product_url'] = f"https://www.flipkart.com{product_data['product_url']}"
        
        # Extract brand from product description (first word typically)
        if product_data['product_description']:
            brand_match = re.match(r'^([A-Za-z]+)', product_data['product_description'])
            if brand_match:
                product_data['brand'] = brand_match.group(1)
        
        
        # Extract rating
        rating_element = product_card.find('div', class_='XQDdHH')
        if rating_element:
            rating_text = rating_element.get_text(strip=True)
            try:
                product_data['rating'] = float(rating_text)
            except ValueError:
                pass
        
        # Extract number of reviews
        review_element = product_card.find('span', class_='Wphh3N')
        if review_element:
            review_text = review_element.get_text(strip=True)
            review_match = re.search(r'\((\d+)\)', review_text)
            if review_match:
                product_data['number_of_reviews'] = int(review_match.group(1))
        
        # Extract image source
        img_element = product_card.find('img', class_='DByuf4')
        if img_element:
            product_data['image_src'] = img_element.get('src', '')
        
        # Extract pricing information
        current_price_elem = product_card.find('div', class_='Nx9bqj')
    
        
        if current_price_elem:
            product_data['current_price'] = current_price_elem.get_text(strip=True)
        
        dge = product_card.find('div', class_='yiggsN')
        
    
    except Exception as e:
        pass  # Continue with partial data
    
    return product_data


def extract_all_products_data(html_content: str, silent: bool = False) -> List[Dict]:
    """Extract product data from all product cards in the Flipkart HTML content."""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Flipkart product card selectors
    selectors = [
        '[data-id]',  # Main product container with data-id
        '.slAVV4',    # Product card container
        '._1xHGtK',   # Alternative product card class
        '._4ddWXP'    # Another potential product card class
    ]
    
    product_cards = []
    for selector in selectors:
        cards = soup.select(selector)
        if cards:
            # Filter cards that actually contain product information
            filtered_cards = []
            for card in cards:
                # Check if card has essential product elements
                has_image = card.find('img', class_='DByuf4') is not None
                has_title = card.find('a', class_='wjcEIp') is not None
                has_price = card.find('div', class_='Nx9bqj') is not None
                has_data_id = card.get('data-id') is not None
                
                if has_image or has_title or has_price or has_data_id:
                    filtered_cards.append(card)
            
            if filtered_cards:
                product_cards = filtered_cards
                if not silent:
                    print(f"Found {len(product_cards)} products using selector: {selector}")
                break
    
    if not product_cards:
        if not silent:
            print("No product cards found")
        return []
    
    all_products_data = []
    stop_loop = 20
    for i,card in enumerate(product_cards):
        if i == stop_loop:
            break
        product_data = extract_single_product_data(card)
        # Only include products with meaningful data
        if (product_data['product_description'] or 
            product_data['product_id'] or 
            product_data['current_price']):
            all_products_data.append(product_data)
    
    return all_products_data


def scrape_flipkart_products(search_term: str) -> pd.DataFrame:
    """Main function to scrape Flipkart products and return clean DataFrame."""
    
    print(f"Scraping Flipkart for: {search_term}")
    
    # Get HTML source
    html_source = get_flipkart_search_html(search_term, silent=True)
    
    if not html_source:
        print("Failed to scrape data")
        return pd.DataFrame()
    
    # Extract products data
    products_data = extract_all_products_data(html_source, silent=True)
    
    if not products_data:
        print("No products found")
        return pd.DataFrame()
    
    # Create clean DataFrame
    df = pd.DataFrame(products_data)
    
    # Clean and reorder columns
    column_order = [
        'product_id', 'product_description', 'brand', 'color', 'size',
        'current_price', 'original_price', 'discount', 'rating', 
        'number_of_reviews', 'badges', 'image_src', 'product_url'
    ]
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    print(f"✓ Extracted {len(df)} products")
    
    return df


def boost_results(result_df, rating_weight=0.6, review_weight=0.4, top_k=5):
    """Boost results based on rating and reviews."""
    if result_df.empty:
        return result_df
        
    boosted_results = []
    
    for i in range(len(result_df)):
        similarity_score = 1.0  # Base score
        
        # Handle rating score
        rating_val = result_df['rating'].iloc[i] if 'rating' in result_df.columns else None
        if rating_val is not None and not pd.isna(rating_val):
            rating_score = float(rating_val) / 5.0  # Assuming rating is out of 5
        else:
            rating_score = 0.0
        
        # Handle review score - use 'number_of_reviews' for Flipkart, 'reviews' for Amazon
        review_col = 'number_of_reviews' if 'number_of_reviews' in result_df.columns else 'reviews'
        review_val = result_df[review_col].iloc[i] if review_col in result_df.columns else None
        if review_val is not None and not pd.isna(review_val):
            review_score = np.log1p(float(review_val)) / np.log1p(1000)  # Normalize
        else:
            review_score = 0.0
        
        boosted_score = similarity_score + (rating_weight * rating_score) + (review_weight * review_score)
        boosted_results.append(boosted_score)
    
    result_df = result_df.copy()  # Avoid modifying original
    result_df["boosted_score"] = boosted_results
    # Sort from higher to lower
    result_df = result_df.sort_values(by='boosted_score', ascending=False)
    return result_df

def get_flipkart_product(query:str):
    df = scrape_flipkart_products(query)
    boosted_df = boost_results(df)
    boosted_df.drop(columns=["boosted_score","product_id","number_of_reviews"], inplace=True)
    return boosted_df.head(2).to_dict(orient = "records")
