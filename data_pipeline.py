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
from llm_analyzers import GeminiAnalyzer, GPT4OAnalyzer, XAIAnalyzer

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

class DataPreparationPipeline:
    def __init__(self, raw_products: List[ScrapedProduct], selected_model: str):
        self.raw_products = raw_products
        self.model = selected_model
        self.processed_products: List[ProcessedProduct] = []
        self.verification_results = []
        self.analyzer = self._get_analyzer()
        
    def _get_analyzer(self):
        """Get appropriate LLM analyzer"""
        if self.model not in ['gemini', 'gpt-4o', 'grok']:
            self.model = 'gemini'  # Default fallback
        analyzers = {
            'gemini': GeminiAnalyzer,
            'gpt-4o': GPT4OAnalyzer,
            'grok': XAIAnalyzer
        }
        return analyzers[self.model](self.model)

    def prepare_data(self) -> pd.DataFrame:
        """Enhanced data preparation pipeline"""
        with st.spinner("Processing and verifying products..."):
            # Create columns for progress tracking
            col1, col2 = st.columns(2)
            progress_bar = col1.progress(0)
            status_text = col2.empty()
            
            # Step 1: Initial Processing
            self._initial_processing(progress_bar, status_text)
            
            # Step 2: Verification
            self._verify_products(progress_bar, status_text)
            
            # Step 3: Quality Checks
            self._quality_checks(progress_bar, status_text)
            
            # Step 4: Create Final DataFrame
            df = self._create_final_dataframe()
            
            return df

    def _initial_processing(self, progress_bar, status_text):
        """Process raw products sequentially with detailed logging"""
        status_text.text("Initial Processing...")

        # Add debug logging
        st.write(f"Starting to process {len(self.raw_products)} raw products")

        for i, product in enumerate(self.raw_products):
            try:
                # Add debug logging for each product
                st.write(f"Processing product {i + 1}/{len(self.raw_products)}: {product.url}")

                # Process the product
                processed = self._process_single_product(product)

                if processed:
                    self.processed_products.append(processed)
                    # Add debug logging
                    st.write(f"Successfully processed product: {product.url}")
                else:
                    # Add debug logging
                    st.write(f"Product not processed: {product.url}")

            except Exception as e:
                st.error(f"Error processing {product.url}: {str(e)}")

            # Update progress bar
            progress_bar.progress((i + 1) / len(self.raw_products))

        # Add debug logging
        st.write(f"Finished processing. Total processed products: {len(self.processed_products)}")

    def _verify_products(self, progress_bar, status_text):
        """Verify processed products"""
        status_text.text("Verifying products...")
        
        for idx, product in enumerate(self.processed_products):
            # Verify image URLs
            product.verification_status = self._verify_images(
                product.product_image,
                product.lifestyle_image
            )
            
            # Update progress
            progress_bar.progress(idx / len(self.processed_products))

    def _quality_checks(self, progress_bar, status_text):
        """Perform quality checks on processed data"""
        status_text.text("Performing quality checks...")
        
        # Remove duplicates based on product URL
        seen_urls = set()
        unique_products = []
        
        for product in self.processed_products:
            if product.product_url not in seen_urls:
                seen_urls.add(product.product_url)
                unique_products.append(product)
        
        # Sort by confidence score
        unique_products.sort(key=lambda x: x.confidence, reverse=True)
        
        self.processed_products = unique_products
        progress_bar.progress(1.0)

    def _create_final_dataframe(self) -> pd.DataFrame:
        # Add debug logging
        st.write(f"Creating DataFrame from {len(self.processed_products)} processed products")
        
        # If no processed products, return empty DataFrame with correct columns
        if not self.processed_products:
            return pd.DataFrame(columns=[
                'Brand_URL',
                'Product_URL',
                'Image1_Link (Product Image)',
                'Image2_link (Lifestyle)',
                'confidence',
                'verification_status'
            ])
        
        # Create DataFrame from processed products
        data = []
        for product in self.processed_products:
            data.append({
                'Brand_URL': product.brand_url,
                'Product_URL': product.product_url,
                'Image1_Link (Product Image)': product.product_image,
                'Image2_link (Lifestyle)': product.lifestyle_image,
                'confidence': product.confidence,
                'verification_status': product.verification_status
            })
        
        df = pd.DataFrame(data)
        
        # Debug logging
        st.write("DataFrame created with columns:", df.columns.tolist())
        st.write("Number of rows before filtering:", len(df))
        
        # Filter using the correct column name
        filtered_df = df[df['verification_status'] == 'verified']
        st.write("Number of rows after filtering:", len(filtered_df))
        
        return filtered_df

    def _verify_images(self, product_image: str, lifestyle_image: str) -> str:
        """Verify image URLs and content"""
        try:
            # Verify product image
            if not self._is_valid_image(product_image):
                return 'invalid_product_image'
            
            # Verify lifestyle image if present
            if lifestyle_image and not self._is_valid_image(lifestyle_image):
                return 'invalid_lifestyle_image'
            
            return 'verified'
            
        except Exception:
            return 'verification_failed'

    def _is_valid_image(self, url: str) -> bool:
        """Check if URL points to valid image"""
        try:
            response = requests.head(url, timeout=5)
            content_type = response.headers.get('content-type', '')
            return content_type.startswith('image/')
        except:
            return False

    def _process_single_product(self, product: ScrapedProduct) -> Optional[ProcessedProduct]:
        """Process a single product using the unified analyzer"""
        try:
            # Prepare the product data
            analysis_data = {
                'all_images': product.all_images or [],
                'page_text': product.page_text or "",
                'metadata': product.metadata or {}
            }

            st.write(f"Analyzing {product.url}")

            # Single analysis call that handles both text and images
            analysis = self.analyzer.analyze_product(analysis_data)
            
            if not analysis:
                st.write(f"No analysis returned for {product.url}")
                return None
                
            st.write(f"Analysis result: {analysis}")

            # If the analysis indicates this is a product, create ProcessedProduct
            if analysis and analysis.get('is_product'):
                return ProcessedProduct(
                    brand_url=urlparse(product.url).netloc,
                    product_url=product.url,
                    product_image=analysis.get('product_image', ''),
                    lifestyle_image=analysis.get('lifestyle_image', ''),
                    confidence=float(analysis.get('confidence', 0.0)),
                    metadata=product.metadata,
                    verification_status='verified'
                )
            else:
                st.write(f"Product {product.url} not identified as a product: {analysis}")
                return None

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return None

    def _fetch_and_encode_image(self, image_url: str) -> Optional[str]:
        """Fetch and encode image in Base64"""
        try:
            response = httpx.get(image_url)
            response.raise_for_status()
            image_data = response.content
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            st.error(f"Error fetching and encoding image {image_url}: {str(e)}")
            return None

    def _validate_image_url(self, url: str) -> bool:
        if not url:
            return False
        try:
            response = requests.head(url, timeout=5)
            return response.status_code == 200 and 'image' in response.headers.get('content-type', '')
        except:
            return False
