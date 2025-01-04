# data_pipeline.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st
import hashlib
import concurrent.futures
from PIL import Image
import requests
from io import BytesIO
import json
from urllib.parse import urlparse
import httpx
import base64
import json
from urllib.parse import urlparse
from typing import Optional, Dict

# Keep your existing imports and add any missing ones needed for LLM analysis
from llm_analyzers import LLMAnalyzer

@dataclass
class ScrapedProduct:
    url: str
    title: str
    description: str
    all_images: List[str]
    page_text: str
    metadata: Dict

# data_pipeline.py

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

    def prepare_data(self) -> pd.DataFrame:
        with st.spinner("Processing and verifying products..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process products
            self._process_products(progress_bar, status_text)
            
            # Create final DataFrame
            df = self._create_final_dataframe()
            
            return df

    def _process_products(self, progress_bar, status_text):
        processed_count = 0
        max_images = 500
        processed_urls = set()  # Track processed image URLs
        
        for idx, product in enumerate(self.raw_products):
            status_text.text(f"Processing product {idx + 1}/{len(self.raw_products)}")
            
            try:
                if processed_count >= max_images:
                    st.warning("Maximum image limit reached")
                    break
                    
                # Filter out already processed images
                new_images = [img for img in product.all_images[:10] 
                            if img not in processed_urls]
                
                if not new_images:
                    st.info(f"Skipping product {idx + 1} - duplicate images")
                    continue
                    
                analysis = self.analyzer.analyze_images(new_images, product.page_text)
                
                if analysis:
                    # Count unique new images
                    unique_new_images = set(analysis['product_images'] + 
                                        analysis['lifestyle_images']) - processed_urls
                        
                    processed_count += len(unique_new_images)
                    processed_urls.update(unique_new_images)
                    
                    if analysis['lifestyle_images']:
                        processed = ProcessedProduct(
                            brand_url=urlparse(product.url).netloc,
                            product_url=product.url,
                            product_images=analysis['product_images'][:5],
                            lifestyle_images=analysis['lifestyle_images'][:5],
                            confidence=analysis['confidence'],
                            status=''
                        )
                        self.processed_products.append(processed)
                        st.success(f"Successfully processed product {idx + 1}")
                    else:
                        st.warning(f"No lifestyle images found for product {idx + 1}")
                        
            except Exception as e:
                st.error(f"Error processing {product.url}: {str(e)}")
                
            progress_bar.progress((idx + 1) / len(self.raw_products))            
    
    def _create_final_dataframe(self) -> pd.DataFrame:
        columns = [
            'S/N', 'Brand URL', 'Product URL',
            *[f'Product Image {i}_Link' for i in range(1, 6)],
            *[f'Lifestyle Image {i}_Link' for i in range(1, 6)],
            'Status', 'Assigned to'
        ]
        
        data = []
        for idx, product in enumerate(self.processed_products, 1):
            row = {
                'S/N': idx,
                'Brand URL': product.brand_url,
                'Product URL': product.product_url
            }
            
            # Add product images
            for i in range(5):
                col = f'Product Image {i+1}_Link'
                row[col] = product.product_images[i] if i < len(product.product_images) else ''
                
            # Add lifestyle images
            for i in range(5):
                col = f'Lifestyle Image {i+1}_Link'
                row[col] = product.lifestyle_images[i] if i < len(product.lifestyle_images) else ''
                
            row['Status'] = product.status
            row['Assigned to'] = product.assigned_to
            
            data.append(row)
            
        df = pd.DataFrame(data, columns=columns)
        return df