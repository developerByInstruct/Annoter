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
from llm_analyzers import GeminiAnalyzer, GPT4OAnalyzer, XAIAnalyzer
from data_pipeline import DataPreparationPipeline, ProcessedProduct 

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
    def __init__(self, brand_url: str, selected_model: str):
        self.brand_url = brand_url
        self.model = selected_model
        self.visited_urls = set()
        self.raw_products = []
        self.max_depth = 3
        
        # Initialize error handler
        self.error_handler = ErrorHandler()
        
        # Detect platform and initialize adapter
        self.platform = PlatformDetector.detect_platform(brand_url, '')
        self.adapter = PlatformAdapter(self.platform)
        
        # Initialize appropriate LLM analyzer
        self.analyzer = self._get_analyzer()

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
        """Enhanced main scraping loop with error handling"""
        urls_to_visit = [self.brand_url]
        self.visited_urls = set()  # Initialize visited URLs set
        
        with st.spinner("Collecting product pages..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while urls_to_visit:
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
                    
                except URLError as e:
                    self.error_handler.handle_url_error(current_url, e)
                    continue
                except Exception as e:
                    self.error_handler.handle_api_error(self.model, e)
                    continue
                
                # Update progress
                total_urls = len(self.visited_urls) + len(urls_to_visit)
                progress = len(self.visited_urls) / total_urls if total_urls > 0 else 1.0
                progress_bar.progress(progress)
        
        # Show error summary
        error_summary = self.error_handler.get_error_summary()
        if error_summary['total_errors'] > 0:
            st.warning(f"Completed with {error_summary['total_errors']} errors. Check logs for details.")
            
        return self.raw_products

    def _scrape_page(self, url: str) -> Optional[ScrapedProduct]:
        """Scrape single page and return raw data"""
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Collect all page data
        images = [img['src'] for img in soup.find_all('img') if 'src' in img.attrs]
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
            all_images=images,
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
        

def display_scraper_page():
    st.title("Product Scraper")
    
    # Model selection
    model_options = ["gemini", "gpt-4o", "grok"]  # Match exact names
    selected_model = st.selectbox("Select LLM Model", model_options)

    # URL input
    brand_url = st.text_input("Enter Brand URL")
    
    if st.button("Start Scraping"):
        if brand_url:
            try:
                # Initialize scraper
                scraper = ProductScraper(brand_url, selected_model)
                
                # Step 1: Collect raw data
                raw_products = scraper.scrape_site()
                
                # Show intermediate results
                st.subheader("Initial Data Collection")
                st.write(f"Found {len(raw_products)} potential product pages")
                
                # Allow user to review raw data
                if st.checkbox("Show raw data"):
                    st.json([{
                        'url': p.url,
                        'title': p.title,
                        'image_count': len(p.all_images)
                    } for p in raw_products])
                
                # Step 2: Process and analyze data
                pipeline = DataPreparationPipeline(raw_products, selected_model)
                df = pipeline.prepare_data()
                
                # Show final results
                st.subheader("Processed Results")
                st.write(f"Processed {len(df)} verified products")
                st.dataframe(df)
                
                # Provide download option
                if not df.empty:
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False)
                    st.download_button(
                        label="Download Excel File",
                        data=excel_buffer.getvalue(),
                        file_name="scraped_products.xlsx",
                        mime="application/vnd.ms-excel"
                    )
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter a brand URL")
