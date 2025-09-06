#!/usr/bin/env python3
"""
Amazon Product Scraper
Extracts product information from Amazon search results using Selenium and Beautiful Soup.
Returns clean pandas DataFrame with product data.
"""

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


def get_amazon_search_html(search_term: str, silent: bool = False) -> str:
    """
    Use Selenium to scrape Amazon search results and return HTML source.
    
    Args:
        search_term: The search query for Amazon
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
        
        driver.get("https://www.amazon.in/")
        time.sleep(2)
        
        search_box = driver.find_element(By.ID, "twotabsearchtextbox")
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
    """Extract product data from a single product card element."""
    
    product_data = {
        'asin': product_card.get('data-asin'),
        'product_description': None,
        'brand': None,
        'price': None,
        'rating': None,
        'number_of_reviews': None,
        'image_src': None,
        'product_url': None
    }
    
    try:
        # Extract product description
        title_element = product_card.find('h2', {'aria-label': True})
        if title_element and 'aria-label' in title_element.attrs:
            aria_label = title_element['aria-label']
            product_data['product_description'] = aria_label.replace('Sponsored Ad - ', '') if aria_label.startswith('Sponsored Ad - ') else aria_label
        
        if not product_data['product_description']:
            title_link = product_card.find('a', class_=re.compile('s-link-style'))
            if title_link:
                h2_element = title_link.find('h2')
                if h2_element:
                    span_text = h2_element.find('span')
                    if span_text:
                        product_data['product_description'] = span_text.get_text(strip=True)
        
        # Extract brand
        brand_element = product_card.find('span', class_='a-size-base-plus a-color-base')
        if brand_element:
            product_data['brand'] = brand_element.get_text(strip=True)
        else:
            product_data['brand'] = product_data['product_description'].split()[0]
        
        if not product_data['brand']:
            brand_h2 = product_card.find('h2', class_='a-size-mini')
            if brand_h2:
                brand_span = brand_h2.find('span', class_='a-size-base-plus a-color-base')
                if brand_span:
                    product_data['brand'] = brand_span.get_text(strip=True)
        
        # Extract rating
        rating_element = product_card.find('span', class_='a-icon-alt')
        if rating_element:
            rating_text = rating_element.get_text(strip=True)
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
            if rating_match:
                product_data['rating'] = float(rating_match.group(1))
        
        # Extract number of reviews
        review_links = product_card.find_all('a', {'aria-label': True})
        for review_link in review_links:
            if 'aria-label' in review_link.attrs:
                aria_label = review_link['aria-label']
                review_match = re.search(r'(\d+)\s+ratings?', aria_label)
                if review_match:
                    product_data['number_of_reviews'] = int(review_match.group(1))
                    break
        
        if not product_data['number_of_reviews']:
            review_elements = product_card.find_all('span', class_='a-size-small puis-normal-weight-text s-underline-text')
            for element in review_elements:
                text = element.get_text(strip=True)
                review_match = re.search(r'\((\d+)\)', text)
                if review_match:
                    product_data['number_of_reviews'] = int(review_match.group(1))
                    break
        
        # Extract product URL
        product_links = product_card.find_all('a', class_=re.compile('a-link-normal'))
        for link in product_links:
            href = link.get('href', '')
            if '/dp/' in href or 'sspa/click' in href:
                product_data['product_url'] = href if href.startswith('http') else f"https://www.amazon.in{href}"
                break
        
        # Extract image src
        img_element = product_card.find('img', class_='s-image')
        if img_element and 'src' in img_element.attrs:
            product_data['image_src'] = img_element['src']
        
        # Extract price
        price_element = product_card.find('span', class_='a-price-whole')
        if price_element:
            price_text = price_element.get_text(strip=True)
            price_clean = re.sub(r'[,\s]', '', price_text)
            try:
                price_num = int(price_clean)
                price_symbol = product_card.find('span', class_='a-price-symbol')
                if price_symbol:
                    symbol = price_symbol.get_text(strip=True)
                    product_data['price'] = f"{symbol}{price_num}"
                else:
                    product_data['price'] = price_num
            except ValueError:
                product_data['price'] = price_text
    
    except Exception:
        pass
    
    return product_data


def extract_all_products_data(html_content: str, silent: bool = False) -> List[Dict]:
    """Extract product data from all product cards in the HTML content."""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    selectors = [
        '[data-component-type="s-search-result"]',
        '.s-result-item',
        '[data-asin]',
        '.s-widget-container'
    ]
    
    product_cards = []
    for selector in selectors:
        cards = soup.select(selector)
        if cards:
            filtered_cards = []
            for card in cards:
                if (card.find('img', class_='s-image') or card.find('h2') or card.get('data-asin')):
                    filtered_cards.append(card)
            
            if filtered_cards:
                product_cards = filtered_cards
                if not silent:
                    print(f"Found {len(product_cards)} products")
                break
    
    if not product_cards:
        if not silent:
            print("No products found")
        return []
    
    all_products_data = []
    stop_loop = 20
    for i,card in enumerate(product_cards):
        product_data = extract_single_product_data(card)
        if (product_data['product_description'] or product_data['asin'] or product_data['image_src']):
            all_products_data.append(product_data)
        if stop_loop == i:
            break
    return all_products_data


def scrape_amazon_products(search_term: str) -> pd.DataFrame:
    """Main function to scrape Amazon products and return clean DataFrame."""
    
    print(f"Scraping Amazon for: {search_term}")
    
    # Get HTML source
    html_source = get_amazon_search_html(search_term, silent=True)
    
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
    column_order = ['asin', 'product_description', 'brand', 'price', 'rating', 'number_of_reviews', 'image_src', 'product_url']
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
        
        # Handle review score - use 'number_of_reviews' for Amazon, 'number_of_reviews' for consistency
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

def get_amazon_product(query:str):
    df = scrape_amazon_products(
        query
    )
    if not df.empty:
        boosted_df = boost_results(df)
        boosted_df.drop(["boosted_score","asin","number_of_reviews"], axis=1, inplace=True)
        return boosted_df.head(2).to_dict(orient = "records")
    else:
        return "can't find similar product in amazon"


# search_term = "professional inline skate"
# df = get_amazon_product(search_term)
# print(df)