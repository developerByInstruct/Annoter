# llm_analyzers.py

import google.generativeai as genai
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
from typing import Dict, List, Optional
import base64
from config import OPENAI_API_KEY, GOOGLE_API_KEY, GROK_API_KEY
import json
import streamlit as st
import httpx


class LLMAnalyzer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._init_client()
        
    def _init_client(self):
        """Initialize appropriate client based on model"""
        if self.model_name == "gemini":
            genai.configure(api_key=GOOGLE_API_KEY)
        elif self.model_name == "gpt-4o":
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
        elif self.model_name == "grok":
            xai_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
            
    def _download_image(self, image_url: str) -> Optional[Image.Image]:
        """Download and verify image"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception:
            return None
            
    def _prepare_images(self, image_urls: List[str]) -> List[Image.Image]:
        """Download and prepare multiple images"""
        images = []
        for url in image_urls:
            img = self._download_image(url)
            if img:
                images.append(img)
        return images

    def analyze_product(self, product_data: Dict) -> Dict:
        """Base method for product analysis"""
        return {
            "is_product": False,
            "product_image": "",
            "lifestyle_image": "",
            "confidence": 0
        }

class GeminiAnalyzer(LLMAnalyzer):
    def analyze_product(self, product_data: Dict) -> Dict:
        """Unified product analysis using Gemini"""
        prompt = """Analyze this product page content and images to identify:
        1. If this is a valid product page
        2. The main product image URL (clear, isolated product shot)
        3. A lifestyle/context image URL if present (showing product in use)
        
        Return ONLY this JSON format:
        {
            "is_product": boolean,
            "product_image": "exact URL of best product image",
            "lifestyle_image": "exact URL of best lifestyle image",
            "confidence": float between 0-1
        }"""

        try:
            # Prepare image data
            image_data = []
            for url in product_data['all_images']:
                try:
                    response = httpx.get(url)
                    response.raise_for_status()
                    img_base64 = base64.b64encode(response.content).decode('utf-8')
                    image_data.append({
                        'mime_type': 'image/jpeg',
                        'data': img_base64
                    })
                except Exception as e:
                    st.error(f"Error processing image {url}: {str(e)}")
                    continue

            if not image_data:
                return {
                    "is_product": False,
                    "product_image": "",
                    "lifestyle_image": "",
                    "confidence": 0
                }

            model = genai.GenerativeModel("gemini-1.5-flash")

            # Feed images, URLs and text content to the model
            content = [
                prompt,
                *image_data,
                f"Available image URLs: {json.dumps(product_data['all_images'])}",
                f"Page text: {product_data['page_text'][:1000]}"  # First 1000 chars of text
            ]

            response = model.generate_content(content)
            
            # Clean up response and parse JSON
            response_text = response.text.replace('```json', '').replace('```', '').strip()
            
            try:
                analysis = json.loads(response_text)
                
                # Validate URLs exist in original list
                if analysis.get('product_image') and analysis['product_image'] not in product_data['all_images']:
                    analysis['product_image'] = ""
                if analysis.get('lifestyle_image') and analysis['lifestyle_image'] not in product_data['all_images']:
                    analysis['lifestyle_image'] = ""
                    
                st.write(f"Analysis result: {analysis}")
                return analysis
                
            except json.JSONDecodeError as e:
                st.error(f"JSON decode error: {str(e)}")
                return {
                    "is_product": False,
                    "product_image": "",
                    "lifestyle_image": "",
                    "confidence": 0
                }

        except Exception as e:
            st.error(f"Error in Gemini analysis: {str(e)}")
            return {
                "is_product": False,
                "product_image": "",
                "lifestyle_image": "",
                "confidence": 0
            }

class GPT4OAnalyzer(LLMAnalyzer):
    def analyze_product(self, product_data: Dict) -> Dict:
        """Unified product analysis using GPT-4V"""
        prompt = """Analyze this product page content and images to identify:
        1. If this is a valid product page
        2. The main product image URL (clear, isolated product shot)
        3. A lifestyle/context image URL if present (showing product in use)
        
        Return ONLY this JSON format:
        {
            "is_product": boolean,
            "product_image": "exact URL of best product image",
            "lifestyle_image": "exact URL of best lifestyle image",
            "confidence": float between 0-1
        }"""
        
        try:
            # Prepare image data
            image_data = []
            for url in product_data['all_images']:
                try:
                    response = httpx.get(url)
                    response.raise_for_status()
                    img_base64 = base64.b64encode(response.content).decode('utf-8')
                    image_data.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    })
                except Exception as e:
                    st.error(f"Error processing image {url}: {str(e)}")
                    continue
            
            if not image_data:
                return {
                    "is_product": False,
                    "product_image": "",
                    "lifestyle_image": "",
                    "confidence": 0
                }
            
            response = openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_data,
                        {"type": "text", "text": f"Available image URLs: {json.dumps(product_data['all_images'])}"},
                        {"type": "text", "text": f"Page text: {product_data['page_text'][:1000]}"}
                    ]
                }],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
                
                # Validate URLs exist in original list
                if analysis.get('product_image') and analysis['product_image'] not in product_data['all_images']:
                    analysis['product_image'] = ""
                if analysis.get('lifestyle_image') and analysis['lifestyle_image'] not in product_data['all_images']:
                    analysis['lifestyle_image'] = ""
                    
                st.write(f"Analysis result: {analysis}")
                return analysis
                
            except json.JSONDecodeError as e:
                st.error(f"JSON decode error: {str(e)}")
                return {
                    "is_product": False,
                    "product_image": "",
                    "lifestyle_image": "",
                    "confidence": 0
                }
            
        except Exception as e:
            st.error(f"Error in GPT-4V analysis: {str(e)}")
            return {
                "is_product": False,
                "product_image": "",
                "lifestyle_image": "",
                "confidence": 0
            }

class XAIAnalyzer(LLMAnalyzer):
    def analyze_product(self, product_data: Dict) -> Dict:
        """Unified product analysis using Grok"""
        prompt = """Analyze this product page content and images to identify:
        1. If this is a valid product page
        2. The main product image URL (clear, isolated product shot)
        3. A lifestyle/context image URL if present (showing product in use)
        
        Return ONLY this JSON format:
        {
            "is_product": boolean,
            "product_image": "exact URL of best product image",
            "lifestyle_image": "exact URL of best lifestyle image",
            "confidence": float between 0-1
        }"""
        
        try:
            # Prepare image data
            image_data = []
            for url in product_data['all_images']:
                try:
                    response = httpx.get(url)
                    response.raise_for_status()
                    img_base64 = base64.b64encode(response.content).decode('utf-8')
                    image_data.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    })
                except Exception as e:
                    st.error(f"Error processing image {url}: {str(e)}")
                    continue
            
            if not image_data:
                return {
                    "is_product": False,
                    "product_image": "",
                    "lifestyle_image": "",
                    "confidence": 0
                }
            
            response = xai_client.chat.completions.create(
                model="grok-2-vision-1212",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_data,
                        {"type": "text", "text": f"Available image URLs: {json.dumps(product_data['all_images'])}"},
                        {"type": "text", "text": f"Page text: {product_data['page_text'][:1000]}"}
                    ]
                }],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
                
                # Validate URLs exist in original list
                if analysis.get('product_image') and analysis['product_image'] not in product_data['all_images']:
                    analysis['product_image'] = ""
                if analysis.get('lifestyle_image') and analysis['lifestyle_image'] not in product_data['all_images']:
                    analysis['lifestyle_image'] = ""
                    
                st.write(f"Analysis result: {analysis}")
                return analysis
                
            except json.JSONDecodeError as e:
                st.error(f"JSON decode error: {str(e)}")
                return {
                    "is_product": False,
                    "product_image": "",
                    "lifestyle_image": "",
                    "confidence": 0
                }
            
        except Exception as e:
            st.error(f"Error in Grok analysis: {str(e)}")
            return {
                "is_product": False,
                "product_image": "",
                "lifestyle_image": "",
                "confidence": 0
            }
    def analyze_product(self, product_data: Dict) -> Dict:
        """Analyze product using Grok"""
        prompt = """Analyze these product images and respond with a JSON object identifying:
        - The main product image URL (should be a clear, isolated product shot)
        - A lifestyle/context image URL if present (should show product being worn/used)
        
        Return ONLY this JSON format with the actual image URLs from the input:
        {
            "is_product": boolean,
            "product_image": "exact URL of best product image",
            "lifestyle_image": "exact URL of best lifestyle image",
            "confidence": float between 0-1
        }"""
        
        try:
            # Prepare image data
            image_data = []
            for url in product_data['all_images']:
                try:
                    response = httpx.get(url)
                    response.raise_for_status()
                    img_base64 = base64.b64encode(response.content).decode('utf-8')
                    image_data.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    })
                except Exception as e:
                    st.error(f"Error processing image {url}: {str(e)}")
                    continue
            
            if not image_data:
                return {
                    "is_product": False,
                    "product_image": "",
                    "lifestyle_image": "",
                    "confidence": 0
                }
            
            response = xai_client.chat.completions.create(
                model="grok-2-vision-1212",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_data,
                        {"type": "text", "text": f"Available image URLs: {json.dumps(product_data['all_images'])}"}
                    ]
                }],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
                
                # Validate URLs exist in original list
                if analysis.get('product_image') and analysis['product_image'] not in product_data['all_images']:
                    analysis['product_image'] = ""
                if analysis.get('lifestyle_image') and analysis['lifestyle_image'] not in product_data['all_images']:
                    analysis['lifestyle_image'] = ""
                    
                st.write(f"Parsed analysis: {analysis}")
                return analysis
                
            except json.JSONDecodeError as e:
                st.error(f"JSON decode error: {str(e)}")
                return {
                    "is_product": False,
                    "product_image": "",
                    "lifestyle_image": "",
                    "confidence": 0
                }
            
        except Exception as e:
            st.error(f"Error in Grok analysis: {str(e)}")
            return {
                "is_product": False,
                "product_image": "",
                "lifestyle_image": "",
                "confidence": 0
            }
    def analyze_product(self, product_data: Dict) -> Dict:
        """Analyze product using GPT-4O"""
        prompt = """Analyze this product page content and images to:
        1. Confirm if this is a product page
        2. Identify the primary product image
        3. Find a lifestyle/context image if available
        4. Rate confidence in identification
        
        Return JSON format:
        {
            "is_product": boolean,
            "product_image": "URL",
            "lifestyle_image": "URL",
            "confidence": float (0-1)
        }
        """
        
        try:
            # Prepare image data
            image_data = []
            for url in product_data['all_images']:
                image = self._download_image(url)
                if image:
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    image_data.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    })
            
            response = xai_client.chat.completions.create(
                model="grok-2-vision-1212",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_data,
                        {"type": "text", "text": product_data['page_text']}
                    ]
                }],
                max_tokens=1000
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "is_product": False,
                "product_image": "",
                "lifestyle_image": "",
                "confidence": 0
            }