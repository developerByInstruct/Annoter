# data_pipeline.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from llm_analyzers import LLMAnalyzer

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
    assigned_to: str = ""

class DataPreparationPipeline:
    def __init__(self, raw_products: List[ScrapedProduct]):
        self.raw_products = raw_products
        self.processed_products: List[ProcessedProduct] = []
        self.analyzer = LLMAnalyzer()
        self.processed_urls = set()  # Track processed URLs to avoid duplicates
    
    def _process_single_product(self, product: ScrapedProduct) -> Optional[ProcessedProduct]:
        """Process a single product"""
        if product.url in self.processed_urls:
            st.info(f"Skipping duplicate product: {product.url}")
            return None

        try:
            # Filter out already processed images
            new_images = [img for img in product.all_images[:15] 
                         if img not in self.processed_urls]
            
            if not new_images:
                st.info(f"Skipping product - duplicate images: {product.url}")
                return None
                
            analysis = self.analyzer.analyze_images(new_images, product.page_text)
            
            if analysis and analysis['lifestyle_images']:
                processed = ProcessedProduct(
                    brand_url=urlparse(product.url).netloc,
                    product_url=product.url,
                    product_images=analysis['product_images'][:5],
                    lifestyle_images=analysis['lifestyle_images'][:5],
                    confidence=analysis['confidence'],
                    status=''
                )
                self.processed_urls.add(product.url)
                return processed
            else:
                st.warning(f"No lifestyle images found for: {product.url}")
                return None
                
        except Exception as e:
            st.error(f"Error processing {product.url}: {str(e)}")
            return None

    def prepare_data(self, max_products: Optional[int] = None) -> pd.DataFrame:
        """Process products with optional limit"""
        products_to_process = self.raw_products[:max_products] if max_products else self.raw_products
        
        with st.spinner("Processing and verifying products..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, product in enumerate(products_to_process):
                status_text.text(f"Processing product {idx + 1}/{len(products_to_process)}")
                
                processed = self._process_single_product(product)
                if processed:
                    self.processed_products.append(processed)
                    
                progress_bar.progress((idx + 1) / len(products_to_process))
                
        return self._create_final_dataframe()

    def _create_final_dataframe(self, products: Optional[List[ProcessedProduct]] = None) -> pd.DataFrame:
        """Create DataFrame from processed products"""
        if products is None:
            products = self.processed_products
            
        columns = [
            'S/N', 'Brand URL', 'Product URL',
            *[f'Product Image {i}_Link' for i in range(1, 6)],
            *[f'Lifestyle Image {i}_Link' for i in range(1, 6)],
            'Status', 'Assigned to'
        ]
        
        data = []
        for idx, product in enumerate(products, 1):
            row = {
                'S/N': idx,
                'Brand URL': product.brand_url,
                'Product URL': product.product_url
            }
            
            # Handle case where there are no product images but have lifestyle images
            product_images = product.product_images
            lifestyle_images = product.lifestyle_images
            
            if not product_images and lifestyle_images:
                # Move first lifestyle image to product images
                product_images = [lifestyle_images[0]]
                lifestyle_images = lifestyle_images[1:]  # Remove the first image from lifestyle
            
            # Add product images
            for i in range(5):
                col = f'Product Image {i+1}_Link'
                row[col] = product_images[i] if i < len(product_images) else ''
                
            # Add lifestyle images
            for i in range(5):
                col = f'Lifestyle Image {i+1}_Link'
                row[col] = lifestyle_images[i] if i < len(lifestyle_images) else ''
                
            row['Status'] = product.status
            row['Assigned to'] = product.assigned_to
            
            data.append(row)
            
        return pd.DataFrame(data, columns=columns)
        