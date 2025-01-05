# scraper_page.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import io
from typing import Dict, List, Optional
import base64
from dataclasses import dataclass
from error_handlers import (
    ErrorHandler, PlatformDetector, PlatformAdapter,
)
from data_pipeline import DataPreparationPipeline
from rate_limiter import RateLimiter
from checkpoint_manager import CheckpointManager
from llm_analyzers import URLAnalyzer, PageAnalyzer
import json
import os
import pandas as pd
from datetime import datetime

@dataclass
class ScrapedProduct:
    url: str
    title: str
    description: str
    all_images: List[str]
    page_text: str
    metadata: Dict

@dataclass
class ProcessedProduct:
    brand_url: str
    product_url: str
    product_image: str
    lifestyle_image: str
    confidence: float
    metadata: Dict
    verification_status: str

class ProductScraper:
    def __init__(self, brand_url: str):
        self.brand_url = brand_url
        self.visited_urls = set()
        self.processed_product_urls = set()  # Track processed product URLs
        self.raw_products = []
        self.max_depth = 3
        self.rate_limiter = RateLimiter()
        self.checkpoint_manager = CheckpointManager()
        self.url_analyzer = URLAnalyzer()
        self.page_analyzer = PageAnalyzer()
        self.error_handler = ErrorHandler()  # Initialize error handler
        
        # Initialize platform detection
        self.platform = None
        self.adapter = None
        
        # Try to detect platform from URL
        try:
            response = requests.get(brand_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            self.platform = PlatformDetector.detect_platform(brand_url, response.text)
            self.adapter = PlatformAdapter(self.platform)
        except Exception as e:
            st.error(f"Error detecting platform: {str(e)}")
            self.platform = None
            self.adapter = PlatformAdapter(None)
        
        # Load previous checkpoint if exists
        checkpoint = self.checkpoint_manager.load_scrape_checkpoint()
        if checkpoint:
            self.visited_urls = set(checkpoint.get('visited_urls', []))
            self.processed_product_urls = set(checkpoint.get('processed_product_urls', []))
            self.raw_products = checkpoint.get('raw_products', [])
            self.last_url = checkpoint.get('last_url')
            st.info(f"Loaded checkpoint from {checkpoint.get('timestamp')}")
            st.info(f"Resume from URL: {self.last_url}")
        else:
            self.last_url = None

    def _save_checkpoint(self, current_url: str, urls_to_visit: List[str], pagination_queue: List[str]) -> None:
        """Save current scraping progress"""
        checkpoint_data = {
            'visited_urls': list(self.visited_urls),
            'processed_product_urls': list(self.processed_product_urls),
            'raw_products': self.raw_products,
            'last_url': current_url,
            'urls_to_visit': urls_to_visit,
            'pagination_queue': pagination_queue,
            'brand_url': self.brand_url
        }
        self.checkpoint_manager.save_scrape_checkpoint(checkpoint_data)

    def scrape_site(self, max_products: int) -> List[ScrapedProduct]:
        """Enhanced main scraping loop with intelligent pagination and duplicate prevention"""
        # Initialize or load from checkpoint
        checkpoint = self.checkpoint_manager.load_scrape_checkpoint()
        if checkpoint and checkpoint.get('brand_url') == self.brand_url:
            urls_to_visit = checkpoint.get('urls_to_visit', [self.brand_url])
            pagination_queue = checkpoint.get('pagination_queue', [])
            self.raw_products = checkpoint.get('raw_products', [])
            st.info(f"Resuming scrape from checkpoint with {len(self.raw_products)} products")
        else:
            urls_to_visit = [self.brand_url]
            pagination_queue = []
        
        with st.spinner("Collecting product pages..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while (urls_to_visit or pagination_queue) and len(self.raw_products) < max_products:
                # Process pagination queue when main queue is empty
                if not urls_to_visit and pagination_queue:
                    urls_to_visit = [pagination_queue.pop(0)]
                    
                if not urls_to_visit:
                    break
                    
                current_url = urls_to_visit.pop(0)
                
                if current_url in self.visited_urls:
                    continue
                    
                self.visited_urls.add(current_url)
                
                try:
                    status_text.text(f"Processing: {current_url}")
                    
                    # Extract and classify links
                    classified_links = self._extract_links(current_url)
                    
                    # Filter out already processed product pages
                    new_product_pages = [
                        url for url in classified_links['product_pages']
                        if url not in self.processed_product_urls
                    ]
                    
                    # Add new product pages to queue
                    urls_to_visit.extend(new_product_pages)
                    
                    # Add new pagination links if we need more products
                    if len(self.raw_products) < max_products:
                        new_pagination_links = [
                            url for url in classified_links['pagination_links']
                            if url not in self.visited_urls and url not in pagination_queue
                        ]
                        pagination_queue.extend(new_pagination_links)
                    
                    # Process current page if it's a product page
                    if self._is_product_page(current_url) and current_url not in self.processed_product_urls:
                        product = self._scrape_page(current_url)
                        if product:
                            self.raw_products.append(product)
                            self.processed_product_urls.add(current_url)
                            
                            if len(self.raw_products) >= max_products:
                                st.success(f"Reached target of {max_products} products")
                                break
                    
                    # Save checkpoint
                    self._save_checkpoint(current_url, urls_to_visit, pagination_queue)
                    
                    # Update progress
                    total_urls = len(self.visited_urls) + len(urls_to_visit) + len(pagination_queue)
                    progress = len(self.visited_urls) / total_urls if total_urls > 0 else 0
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    self.error_handler.handle_url_error(current_url, e)
                    continue
        
        return self.raw_products

    def _is_product_page(self, url: str) -> bool:
        """Check if URL is a product page"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get platform-specific context
            platform_context = self.adapter.get_platform_context(soup)
            
            return self.page_analyzer.is_product_page(
                url,
                soup.get_text(),
                platform_context
            )
            
        except Exception as e:
            self.error_handler.handle_url_error(url, e)
            return False

    def _extract_links(self, url: str) -> Dict[str, List[str]]:
        """Extract and classify links using LLM analysis"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get all links
            all_links = {
                urljoin(self.brand_url, a['href'])
                for a in soup.find_all('a', href=True)
                if urljoin(self.brand_url, a['href']).startswith(self.brand_url)
            }
            # Prepare context
            context = {
                'url': url,
                'title': soup.title.text if soup.title else '',
                'platform': self.platform,
                'platform_context': self.adapter.get_platform_context(soup)
            }
            
            # Use URL analyzer
            classification = self.url_analyzer.analyze_urls(list(all_links), context)
            
            return {
                'product_pages': [url for url in classification['product_pages'] 
                                if url not in self.visited_urls],
                'pagination_links': [url for url in classification['pagination_links'] 
                                   if url not in self.visited_urls],
                'category_pages': classification['category_pages']
            }
            
        except Exception as e:
            self.error_handler.handle_url_error(url, e)
            return {'product_pages': [], 'pagination_links': [], 'category_pages': []}
    
    def _scrape_page(self, url: str) -> Optional[ScrapedProduct]:
        """Scrape single page and return raw data"""
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Collect all page data with normalized URLs
        images = [
            self._normalize_url(img['src']) 
            for img in soup.find_all('img') 
            if 'src' in img.attrs
        ]
        text = soup.get_text(separator='\n', strip=True)

        # Basic metadata extraction
        title = soup.find('title').text if soup.find('title') else ''
        description = soup.find('meta', {'name': 'description'})
        description = description['content'] if description else ''

        # Add logging
        st.write(f"Scraped data for {url}:")
        st.write(f"Title: {title}")
        st.write(f"Description: {description}")
        st.write(f"Images: {images}")
        st.write(f"Text: {text[:100]}...")  # Print first 100 characters of text

        return ScrapedProduct(
            url=url,
            title=title,
            description=description,
            all_images=images[:10],
            page_text=text,
            metadata={
                'headers': [h.text for h in soup.find_all(['h1', 'h2', 'h3'])],
                'structured_data': self._extract_structured_data(soup)
            }
        )
        
    def _extract_structured_data(self, soup) -> Dict:
        """Extract any structured data from the page"""
        structured_data = {}
        for script in soup.find_all('script', {'type': 'application/ld+json'}):
            try:
                data = json.loads(script.string)
                structured_data.update(data)
            except:
                continue
        return structured_data

    def _normalize_url(self, url: str) -> str:
        """Ensure URL has proper protocol"""
        if url.startswith('//'):
            return f'https:{url}'
        elif not url.startswith(('http://', 'https://')):
            return f'https://{url}'
        return url

    def save_raw_products(self, filename: str):
        """Save raw products to file"""
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump([vars(p) for p in self.raw_products], f)
        elif filename.endswith('.md'):
            with open(filename, 'w') as f:
                for product in self.raw_products:
                    f.write(f"# {product.title}\n\n")
                    f.write(f"URL: {product.url}\n\n")
                    f.write(f"Description: {product.description}\n\n")
                    f.write("Images:\n")
                    for img in product.all_images:
                        f.write(f"- {img}\n")
                    f.write("\n---\n\n")

def process_raw_products(raw_products: List[ScrapedProduct], max_products: int, checkpoint_manager: CheckpointManager) -> pd.DataFrame:
    """Process raw products with checkpointing"""
    checkpoint = checkpoint_manager.load_process_checkpoint()
    start_index = 0
    processed_products = []
    
    if checkpoint:
        start_index = checkpoint.get('last_processed_index', 0)
        processed_products = checkpoint.get('processed_products', [])
        
    pipeline = DataPreparationPipeline(raw_products[start_index:start_index + max_products])
    
    with st.spinner("Processing raw products..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, product in enumerate(pipeline.raw_products):
            current_index = start_index + i
            status_text.text(f"Processing product {current_index + 1}/{len(raw_products)}")
            
            # Process product
            processed = pipeline._process_single_product(product)
            if processed:
                # Convert ProcessedProduct to dictionary with correct field names
                processed_dict = {
                    'brand_url': processed.brand_url,
                    'product_url': processed.product_url,
                    'product_image': processed.product_images[0] if processed.product_images else "",  # Take first image
                    'lifestyle_image': processed.lifestyle_images[0] if processed.lifestyle_images else "",  # Take first image
                    'confidence': processed.confidence,
                    'verification_status': processed.status,
                }
                processed_products.append(processed_dict)
                
            # Update progress
            progress = (i + 1) / len(pipeline.raw_products)
            progress_bar.progress(progress)
            
            # Save checkpoint
            checkpoint_data = {
                'last_processed_index': current_index + 1,
                'processed_products': processed_products
            }
            checkpoint_manager.save_process_checkpoint(checkpoint_data)
            
    # Convert checkpoint data back to ProcessedProduct objects for DataFrame creation
    processed_objects = [ProcessedProduct(**p) for p in processed_products]
    return pipeline._create_final_dataframe(processed_objects)

def display_scraper_page():
    st.title("Product Scraper")
    
    # Add mode selection
    mode = st.radio("Select Mode", ["Scrape Products", "Process Raw Products"])
    
    # Add rate limit settings
    max_items = st.number_input("Maximum items to process", min_value=1, value=10)
    
    if st.button("Clear Checkpoints"):
        CheckpointManager().clear_checkpoints()
        st.success("Checkpoints cleared")
    
    if mode == "Scrape Products":
        # URL input
        brand_url = st.text_input("Enter Brand URL")
        file_format = st.selectbox("Save Format", ["json", "md"])
        
        if st.button("Start Scraping"):
            if brand_url:
                try:
                    scraper = ProductScraper(brand_url)
                    raw_products = scraper.scrape_site(max_items)
                    
                    if raw_products:
                        filename = f"raw_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_format}"
                        scraper.save_raw_products(filename)
                        st.success(f"Saved {len(raw_products)} products to {filename}")
                    else:
                        st.warning("No products found during scraping.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    
    else:  # Process Raw Products
        uploaded_file = st.file_uploader("Upload raw products file", type=['json', 'md'])
        
        if uploaded_file and st.button("Process Products"):
            try:
                if uploaded_file.name.endswith('.json'):
                    raw_data = json.load(uploaded_file)
                    raw_products = [ScrapedProduct(**p) for p in raw_data]
                else:
                    # Implement markdown parsing if needed
                    st.error("Markdown parsing not implemented yet")
                    return
                
                df = process_raw_products(raw_products, max_items, CheckpointManager())
                
                if not df.empty:
                    st.write(f"Processed {len(df)} products")
                    st.dataframe(df)
                    
                    # Download option
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False)
                    st.download_button(
                        label="Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name="processed_products.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                else:
                    st.warning("No products could be processed.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)