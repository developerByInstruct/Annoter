# scraper_page.py

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
import io
import time
from openai import OpenAI
import google.generativeai as genai
from typing import Dict, List, Optional
import base64
from dataclasses import dataclass
from error_handlers import (
    ErrorHandler, PlatformDetector, PlatformAdapter,
    ScraperError, URLError, ImageError, APIError
)
from data_pipeline import DataPreparationPipeline, ProcessedProduct 
from rate_limiter import RateLimiter
from checkpoint_manager import CheckpointManager

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
        self.raw_products = []
        self.max_depth = 3
        self.rate_limiter = RateLimiter()
        self.checkpoint_manager = CheckpointManager()
        
        # Load previous checkpoint if exists
        checkpoint = self.checkpoint_manager.load_checkpoint()
        if checkpoint:
            self.visited_urls = set(checkpoint.get('visited_urls', []))
            self.last_url = checkpoint.get('last_url')
            self.rate_limiter.rate_limit.processed_images = checkpoint.get('processed_images', 0)
            st.info(f"Loaded checkpoint from {checkpoint.get('timestamp')}")
            st.info(f"Resume from URL: {self.last_url}")
        else:
            self.last_url = None
        
        # Initialize error handler
        self.error_handler = ErrorHandler()
        
        # Detect platform and initialize adapter
        self.platform = PlatformDetector.detect_platform(brand_url, '')
        self.adapter = PlatformAdapter(self.platform)

    def _save_checkpoint(self, current_url: str) -> None:
        """Save current scraping progress"""
        checkpoint_data = {
            'visited_urls': list(self.visited_urls),
            'last_url': current_url,
            'processed_images': self.rate_limiter.rate_limit.processed_images,
            'brand_url': self.brand_url
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_data)

    def _extract_links(self, url: str) -> List[str]:
        """Extract relevant links from a page"""
        # Calculate current depth
        current_depth = url.replace(self.brand_url, '').count('/')
        if current_depth >= self.max_depth:
            return []
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get platform-specific selectors
            selectors = self.adapter.get_product_selectors()
            
            # Extract links based on platform patterns
            links = set()
            
            # Product links
            product_patterns = [
                '/products/', 
                '/product/',
                '/item/',
                'product-detail'
            ]
            
            # Collection/category links
            collection_patterns = [
                '/collections/',
                '/category/',
                '/shop/',
                'product-category'
            ]
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                absolute_url = urljoin(self.brand_url, href)
                
                # Skip external links and already visited URLs
                if not absolute_url.startswith(self.brand_url) or absolute_url in self.visited_urls:
                    continue
                
                # Check if it's a product or collection link
                if any(pattern in href for pattern in product_patterns + collection_patterns):
                    links.add(absolute_url)
                
                # Check for pagination links
                if 'page=' in href or '/page/' in href:
                    links.add(absolute_url)
            
            # Check for platform-specific pagination
            pagination = soup.select(self.adapter.get_pagination_selector())
            if pagination:
                for page_link in pagination[0].find_all('a', href=True):
                    absolute_url = urljoin(self.brand_url, page_link['href'])
                    if absolute_url.startswith(self.brand_url):
                        links.add(absolute_url)
            
            return list(links)
            
        except Exception as e:
            self.error_handler.handle_url_error(url, e)
            return []
        
    def _get_analyzer(self):
        """Get appropriate LLM analyzer"""
        analyzers = {
            'gemini': GeminiAnalyzer,
            'gpt-4o': GPT4OAnalyzer,
            'grok': XAIAnalyzer
        }
        return analyzers[self.model](self.model)
    

    def scrape_site(self) -> List[ScrapedProduct]:
        """Enhanced main scraping loop with checkpoint support"""
        # Initialize URLs to visit based on checkpoint
        if self.last_url:
            urls_to_visit = [self.last_url]
        else:
            urls_to_visit = [self.brand_url]
        
        with st.spinner("Collecting product pages..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            rate_limit_status = st.empty()
            checkpoint_status = st.empty()
            
        try:
            while urls_to_visit:
                # Check rate limits
                if not self.rate_limiter.can_process():
                    limiter_status = self.rate_limiter.get_status()
                    
                    if limiter_status['processed_images'] >= self.rate_limiter.rate_limit.max_images:
                        # Save checkpoint before stopping
                        self._save_checkpoint(urls_to_visit[0])
                        checkpoint_status.info(f"Checkpoint saved. Resume from: {urls_to_visit[0]}")
                        st.info("Maximum image limit reached. Stopping scraping.")
                        break
                
                current_url = urls_to_visit.pop(0)
                
                # Skip if already visited
                if current_url in self.visited_urls:
                    continue
                    
                # Mark as visited before processing
                self.visited_urls.add(current_url)
                
                try:
                    # Add rate limiting
                    time.sleep(self.adapter.get_rate_limit_delay())
                    
                    # Update status
                    status_text.text(f"Processing: {current_url}")
                    
                    # Scrape page content
                    product = self._scrape_page(current_url)
                    if product:
                        self.raw_products.append(product)
                    
                    # Get new URLs to visit
                    new_urls = self._extract_links(current_url)
                    
                    # Only add new URLs that haven't been visited or queued
                    new_urls = [url for url in new_urls 
                            if url not in self.visited_urls 
                            and url not in urls_to_visit]
                    
                    urls_to_visit.extend(new_urls)
                    
                    # Update rate limiter status
                    self.rate_limiter.increment_counter()
                    limiter_status = self.rate_limiter.get_status()
                    rate_limit_status.text(
                        f"Processed products: {limiter_status['processed_images']} | "
                        f"Remaining: {limiter_status['images_remaining']} | "
                        f"Next reset: {limiter_status['next_reset'].strftime('%H:%M:%S')}"
                    )

                    # Save checkpoint periodically (e.g., every 10 URLs)
                    if len(self.raw_products) % 10 == 0:
                        self._save_checkpoint(current_url)
                        checkpoint_status.info(f"Checkpoint saved. Current URL: {current_url}")
                    
                except Exception as e:
                    # Save checkpoint on error
                    self._save_checkpoint(current_url)
                    checkpoint_status.error(f"Error occurred. Checkpoint saved at: {current_url}")
                    raise e

        except Exception as e:
            # Process any collected data before raising exception
            if self.raw_products:
                pipeline = DataPreparationPipeline(self.raw_products)
                df = pipeline.prepare_data()
                if not df.empty:
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False)
                    st.download_button(
                        label="Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name="scraped_products.xlsx",
                        mime="application/vnd.ms-excel"
                    )
            raise e

        return self.raw_products
    
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

def display_scraper_page():
    st.title("Product Scraper")
    
    # Add rate limit settings
    col1, col2 = st.columns(2)
    with col1:
        max_images = st.number_input("Maximum products to process", min_value=1, value=10)
    with col2:
        if st.button("Clear Checkpoint"):
            CheckpointManager().clear_checkpoint()
            st.success("Checkpoint cleared")
    
    # URL input
    brand_url = st.text_input("Enter Brand URL")
    
    # Load previous checkpoint
    checkpoint = CheckpointManager().load_checkpoint()
    if checkpoint:
        st.info(f"Previous session found from {checkpoint.get('timestamp')}")
        st.info(f"Resume from URL: {checkpoint.get('last_url')}")
        if brand_url and brand_url != checkpoint.get('brand_url'):
            st.warning("Warning: Entered URL differs from checkpoint URL")
    
    if st.button("Start Scraping"):
        if brand_url:
            try:
                # Initialize scraper with rate limits
                scraper = ProductScraper(brand_url)
                scraper.rate_limiter = RateLimiter(
                    max_images=max_images
                )
                
                # Step 1: Collect raw data
                raw_products = scraper.scrape_site()
                
                # Ensure raw_products is not None before checking length
                if raw_products is not None:
                    # Show intermediate results
                    with st.expander("Initial Data Collection", expanded=True):
                        st.write(f"Found {len(raw_products)} potential product pages")
                        
                        # Display platform info if detected
                        if scraper.platform:
                            st.info(f"Detected platform: {scraper.platform}")
                        
                        # Show error summary if any
                        error_summary = scraper.error_handler.get_error_summary()
                        if error_summary['total_errors'] > 0:
                            st.warning("Scraping completed with errors:", icon="⚠️")
                            st.json(error_summary)
                    
                    # Only proceed with pipeline if we have products
                    if raw_products:
                        # Step 2: Process and analyze data
                        pipeline = DataPreparationPipeline(raw_products)
                        df = pipeline.prepare_data()
                        
                        # Show final results
                        with st.expander("Processed Results", expanded=True):
                            st.write(f"Processed {len(df)} verified products")
                            st.dataframe(df)
                        
                        # Download options
                        if not df.empty:
                            excel_buffer = io.BytesIO()
                            df.to_excel(excel_buffer, index=False)
                            st.download_button(
                                label="Download Excel",
                                data=excel_buffer.getvalue(),
                                file_name="scraped_products.xlsx",
                                mime="application/vnd.ms-excel"
                            )
                    else:
                        st.warning("No products found during scraping.")
                else:
                    st.error("Scraping failed to return any results.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
        else:
            st.error("Please enter a brand URL")
