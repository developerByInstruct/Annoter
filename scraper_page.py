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
    product_images: List[str]
    lifestyle_images: List[str]
    confidence: float
    status: str
    assigned_to: str 

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
        self.urls_to_visit = [brand_url]  # Initialize with brand URL
        self.pagination_queue = []  # Initialize pagination queue
        self.last_url = None
        
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

    def _save_checkpoint(self, current_url: str) -> None:
        """Save current scraping progress"""
        checkpoint_data = {
            'visited_urls': list(self.visited_urls),
            'processed_product_urls': list(self.processed_product_urls),
            'raw_products': self.raw_products,
            'last_url': current_url,
            'urls_to_visit': self.urls_to_visit,
            'pagination_queue': self.pagination_queue,
            'brand_url': self.brand_url,
            'timestamp': datetime.now().isoformat()
        }
        self.checkpoint_manager.save_scrape_checkpoint(checkpoint_data)

    def scrape_site(self, max_products: int) -> List[ScrapedProduct]:
        """Enhanced main scraping loop with intelligent pagination and duplicate prevention"""
        # Calculate target number of products when continuing from checkpoint
        current_products = len(self.raw_products)
        target_products = current_products + max_products
        
        with st.spinner("Collecting product pages..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while (self.urls_to_visit or self.pagination_queue) and len(self.raw_products) < target_products:
                # Process pagination queue when main queue is empty
                if not self.urls_to_visit and self.pagination_queue:
                    self.urls_to_visit = [self.pagination_queue.pop(0)]
                    
                if not self.urls_to_visit:
                    break
                    
                current_url = self.urls_to_visit.pop(0)
                self.last_url = current_url  # Update last URL
                
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
                    self.urls_to_visit.extend(new_product_pages)
                    
                    # Add new pagination links if we need more products
                    if len(self.raw_products) < target_products:
                        new_pagination_links = [
                            url for url in classified_links['pagination_links']
                            if url not in self.visited_urls and url not in self.pagination_queue
                        ]
                        self.pagination_queue.extend(new_pagination_links)
                    
                    # Process current page if it's a product page
                    if self._is_product_page(current_url) and current_url not in self.processed_product_urls:
                        product = self._scrape_page(current_url)
                        if product:
                            self.raw_products.append(product)
                            self.processed_product_urls.add(current_url)
                            
                            if len(self.raw_products) >= target_products:
                                st.success(f"Reached target of {len(self.raw_products)} products ({current_products} from checkpoint + {max_products} new)")
                                break
                    
                    # Save checkpoint
                    self._save_checkpoint(current_url)
                    
                    # Update progress
                    total_urls = len(self.visited_urls) + len(self.urls_to_visit) + len(self.pagination_queue)
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

        # Get platform-specific selectors
        selectors = self.adapter.get_product_selectors()
        
        # First try to get images from product-specific containers
        product_images = []
        
        # Try common product gallery containers first
        gallery_containers = soup.find_all(class_=lambda x: x and any(term in str(x).lower() 
            for term in ['product-gallery', 'product-images', 'product-photos', 'gallery-container',
                        'main-image', 'product-media', 'product-thumbnails', 'product-slider']))
        
        for container in gallery_containers:
            images = container.find_all('img')
            for img in images:
                if 'src' in img.attrs:
                    product_images.append(self._normalize_url(img['src']))
                elif 'data-src' in img.attrs:
                    product_images.append(self._normalize_url(img['data-src']))
        
        # If no images found in galleries, try product-specific image attributes
        if not product_images:
            images = soup.find_all('img', attrs={
                'class': lambda x: x and any(term in str(x).lower() 
                    for term in ['product', 'gallery', 'main-image', 'featured']),
                'id': lambda x: x and any(term in str(x).lower() 
                    for term in ['product', 'gallery', 'main-image', 'featured'])
            })
            
            for img in images:
                if 'src' in img.attrs:
                    product_images.append(self._normalize_url(img['src']))
                elif 'data-src' in img.attrs:
                    product_images.append(self._normalize_url(img['data-src']))
        
        # Filter out common non-product images
        filtered_images = []
        excluded_patterns = [
            'logo', 'icon', 'banner', 'payment', 'social', 'related', 'recommended',
            'cart', 'shipping', 'guarantee', 'review', 'rating', 'thumbnail-mini',
            'placeholder', 'advertisement', 'promotion', 'widget', 'footer', 'header'
        ]
        
        for img_url in product_images:
            if not any(pattern in img_url.lower() for pattern in excluded_patterns):
                filtered_images.append(img_url)
        
        # Remove duplicate URLs while preserving order
        seen = set()
        unique_images = []
        for img in filtered_images:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)
        
        text = soup.get_text(separator='\n', strip=True)

        # Basic metadata extraction
        title = soup.find('title').text if soup.find('title') else ''
        description = soup.find('meta', {'name': 'description'})
        description = description['content'] if description else ''

        # Add logging
        st.write(f"Scraped data for {url}:")
        st.write(f"Title: {title}")
        st.write(f"Description: {description}")
        st.write(f"Total images found: {len(product_images)}")
        st.write(f"After filtering: {len(filtered_images)}")
        st.write(f"Unique product images: {len(unique_images)}")
        st.write(f"First 15 unique product images: {unique_images[:15]}")
        st.write(f"Text: {text[:100]}...")  # Print first 100 characters of text

        return ScrapedProduct(
            url=url,
            title=title,
            description=description,
            all_images=unique_images[:15],  # Take first 15 unique product images
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
        st.info(f"Resuming from checkpoint: {start_index} products processed")
        
        # Calculate remaining products to process
        remaining_products = max_products - len(processed_products)
        if remaining_products <= 0:
            st.success("Already processed requested number of products")
            # Convert processed products to DataFrame and return
            processed_objects = [ProcessedProduct(**p) for p in processed_products]
            pipeline = DataPreparationPipeline([])  # Empty pipeline just for DataFrame creation
            return pipeline._create_final_dataframe(processed_objects)
            
        max_products = remaining_products
        st.info(f"Will process {remaining_products} more products")
    
    pipeline = DataPreparationPipeline(raw_products[start_index:start_index + max_products])
    
    with st.spinner("Processing raw products..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_to_process = len(pipeline.raw_products)
        for i, product in enumerate(pipeline.raw_products):
            current_index = start_index + i
            status_text.text(f"Processing product {current_index + 1}/{len(raw_products)} (Batch progress: {i + 1}/{total_to_process})")
            
            # Process product
            processed = pipeline._process_single_product(product)
            if processed:
                # Convert ProcessedProduct to dictionary with correct field names
                processed_dict = {
                    'brand_url': processed.brand_url,
                    'product_url': processed.product_url,
                    'product_images': processed.product_images if processed.product_images else "",
                    'lifestyle_images': processed.lifestyle_images if processed.lifestyle_images else "",
                    'confidence': processed.confidence,
                    'status': processed.status,
                    'assigned_to': processed.assigned_to
                }
                processed_products.append(processed_dict)
                
            # Update progress
            progress = (i + 1) / total_to_process
            progress_bar.progress(progress)
            
            # Save checkpoint with timestamp
            checkpoint_data = {
                'last_processed_index': current_index + 1,
                'processed_products': processed_products,
                'timestamp': datetime.now().isoformat()
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
    
    # Check for existing checkpoints
    checkpoint_manager = CheckpointManager()
    scrape_checkpoint = checkpoint_manager.load_scrape_checkpoint()
    process_checkpoint = checkpoint_manager.load_process_checkpoint()
    
    if st.button("Clear Checkpoints"):
        checkpoint_manager.clear_checkpoints()
        st.success("Checkpoints cleared")
        st.rerun()
    
    if mode == "Scrape Products":
        # Initialize variables
        brand_url = None
        continue_scraping = False
        
        # Show checkpoint info if exists
        if scrape_checkpoint:
            st.info(f"Found existing checkpoint from {scrape_checkpoint.get('timestamp')}")
            st.info(f"Last URL: {scrape_checkpoint.get('last_url')}")
            st.info(f"Products collected so far: {len(scrape_checkpoint.get('raw_products', []))}")
            st.info(f"Checkpoint location: {checkpoint_manager.scrape_checkpoint}")
            
            col1, col2 = st.columns(2)
            with col1:
                continue_scraping = st.button("Continue from Checkpoint")
            with col2:
                if st.button("Start New Scraping"):
                    checkpoint_manager.clear_checkpoints()
                    st.rerun()
            
            # Move scraping logic outside columns
            if continue_scraping:
                brand_url = scrape_checkpoint.get('brand_url')
                try:
                    scraper = ProductScraper(brand_url)
                    
                    # If continuing from checkpoint, load the checkpoint data
                    scraper.visited_urls = set(scrape_checkpoint.get('visited_urls', []))
                    scraper.processed_product_urls = set(scrape_checkpoint.get('processed_product_urls', []))
                    # Convert checkpoint products back to ScrapedProduct objects
                    scraper.raw_products = [
                        ScrapedProduct(
                            url=p['url'],
                            title=p['title'],
                            description=p['description'],
                            all_images=p['all_images'],
                            page_text=p['page_text'],
                            metadata=p['metadata']
                        ) if isinstance(p, dict) else p
                        for p in scrape_checkpoint.get('raw_products', [])
                    ]
                    scraper.last_url = scrape_checkpoint.get('last_url')
                    
                    # Load URLs to visit and pagination queue from checkpoint
                    urls_to_visit = scrape_checkpoint.get('urls_to_visit', [])
                    pagination_queue = scrape_checkpoint.get('pagination_queue', [])
                    
                    # If we have URLs to visit, start from there
                    if urls_to_visit:
                        st.info(f"Resuming scraping from URL queue with {len(urls_to_visit)} URLs remaining")
                        scraper.urls_to_visit = urls_to_visit
                        scraper.pagination_queue = pagination_queue
                    # Otherwise, start from the last URL processed
                    elif scraper.last_url:
                        st.info(f"Resuming scraping from last URL: {scraper.last_url}")
                        scraper.urls_to_visit = [scraper.last_url]
                        scraper.pagination_queue = []
                    
                    st.info(f"Loaded checkpoint data: {len(scraper.raw_products)} products, {len(scraper.visited_urls)} visited URLs")
                    
                    raw_products = scraper.scrape_site(max_items)
                    
                    if raw_products:
                        # Create download buttons for both formats
                        # JSON format
                        json_output = json.dumps([{
                            'url': p.url if isinstance(p, ScrapedProduct) else p['url'],
                            'title': p.title if isinstance(p, ScrapedProduct) else p['title'],
                            'description': p.description if isinstance(p, ScrapedProduct) else p['description'],
                            'all_images': p.all_images if isinstance(p, ScrapedProduct) else p['all_images'],
                            'page_text': p.page_text if isinstance(p, ScrapedProduct) else p['page_text'],
                            'metadata': p.metadata if isinstance(p, ScrapedProduct) else p['metadata']
                        } for p in raw_products], indent=2)
                        json_filename = f"raw_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        st.download_button(
                            label="Download JSON",
                            data=json_output,
                            file_name=json_filename,
                            mime="application/json"
                        )
                        
                        st.success(f"Found {len(raw_products)} products. Click above to download.")
                    else:
                        st.warning("No products found during scraping.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
        
        # Show URL input if no checkpoint or starting new scrape
        if not continue_scraping:
            brand_url = st.text_input("Enter Brand URL")
            
        if st.button("Start Scraping"):
            if not brand_url:
                st.error("Please enter a brand URL or continue from checkpoint")
                return
                
            try:
                scraper = ProductScraper(brand_url)
                
                # If continuing from checkpoint, load the checkpoint data
                if continue_scraping and scrape_checkpoint:
                    scraper.visited_urls = set(scrape_checkpoint.get('visited_urls', []))
                    scraper.processed_product_urls = set(scrape_checkpoint.get('processed_product_urls', []))
                    # Convert checkpoint products back to ScrapedProduct objects
                    scraper.raw_products = [
                        ScrapedProduct(
                            url=p['url'],
                            title=p['title'],
                            description=p['description'],
                            all_images=p['all_images'],
                            page_text=p['page_text'],
                            metadata=p['metadata']
                        ) if isinstance(p, dict) else p
                        for p in scrape_checkpoint.get('raw_products', [])
                    ]
                    scraper.last_url = scrape_checkpoint.get('last_url')
                    
                    # Load URLs to visit and pagination queue from checkpoint
                    urls_to_visit = scrape_checkpoint.get('urls_to_visit', [])
                    pagination_queue = scrape_checkpoint.get('pagination_queue', [])
                    
                    # If we have URLs to visit, start from there
                    if urls_to_visit:
                        st.info(f"Resuming scraping from URL queue with {len(urls_to_visit)} URLs remaining")
                        scraper.urls_to_visit = urls_to_visit
                        scraper.pagination_queue = pagination_queue
                    # Otherwise, start from the last URL processed
                    elif scraper.last_url:
                        st.info(f"Resuming scraping from last URL: {scraper.last_url}")
                        scraper.urls_to_visit = [scraper.last_url]
                        scraper.pagination_queue = []
                    
                    st.info(f"Loaded checkpoint data: {len(scraper.raw_products)} products, {len(scraper.visited_urls)} visited URLs")
                
                raw_products = scraper.scrape_site(max_items)
                
                if raw_products:
                    # Create download buttons for both formats
                    # JSON format
                    json_output = json.dumps([{
                        'url': p.url if isinstance(p, ScrapedProduct) else p['url'],
                        'title': p.title if isinstance(p, ScrapedProduct) else p['title'],
                        'description': p.description if isinstance(p, ScrapedProduct) else p['description'],
                        'all_images': p.all_images if isinstance(p, ScrapedProduct) else p['all_images'],
                        'page_text': p.page_text if isinstance(p, ScrapedProduct) else p['page_text'],
                        'metadata': p.metadata if isinstance(p, ScrapedProduct) else p['metadata']
                    } for p in raw_products], indent=2)
                    json_filename = f"raw_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    st.download_button(
                        label="Download JSON",
                        data=json_output,
                        file_name=json_filename,
                        mime="application/json"
                    )
                    
                    st.success(f"Found {len(raw_products)} products. Click above to download.")
                else:
                    st.warning("No products found during scraping.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
                    
    else:  # Process Raw Products
        # Show checkpoint info if exists
        if process_checkpoint:
            st.info(f"Found existing checkpoint from {process_checkpoint.get('timestamp', '')}")
            st.info(f"Products processed so far: {len(process_checkpoint.get('processed_products', []))}")
            st.info(f"Last processed index: {process_checkpoint.get('last_processed_index', 0)}")
            st.info(f"Checkpoint location: {checkpoint_manager.process_checkpoint}")
            
            col1, col2 = st.columns(2)
            with col1:
                continue_processing = st.button("Continue Processing")
            with col2:
                new_processing = st.button("Start New Processing")
                
            if continue_processing:
                if not hasattr(st.session_state, 'raw_products'):
                    st.error("No raw products data found. Please upload a file first.")
                    st.stop()
                    
                try:
                    df = process_raw_products(st.session_state.raw_products, max_items, checkpoint_manager)
                    
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
                st.stop()
                
            elif new_processing:
                checkpoint_manager.clear_checkpoints()
                st.rerun()
            else:
                st.stop()
                
        uploaded_file = st.file_uploader("Upload raw products file", type=['json', 'md'])
        
        if uploaded_file and st.button("Process Products"):
            try:
                if uploaded_file.name.endswith('.json'):
                    raw_data = json.load(uploaded_file)
                    raw_products = [ScrapedProduct(**p) for p in raw_data]
                    # Store raw_products in session state for continuation
                    st.session_state.raw_products = raw_products
                else:
                    st.error("Input file must be a JSON file of scraped products.")
                    return
                
                df = process_raw_products(raw_products, max_items, checkpoint_manager)
                
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