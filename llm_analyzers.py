# llm_analyzers.py

import google.generativeai as genai
from openai import OpenAI
from typing import Dict, List
import base64
from config import OPENAI_API_KEY, GEMINI_API_KEY_1, GEMINI_API_KEY_2, GROK_API_KEY
import json
import streamlit as st
import httpx
import time

# llm_analyzers.py

class LLMAnalyzer:
    def __init__(self):
        self.sys_prompt = """You are a professional image classifier specializing in product and lifestyle image analysis. Your task is to:
1. Identify and separate product images from lifestyle images
2. Ensure proper classification based on strict criteria

Product Image Criteria:
- Must be isolated product shots
- Neutral/solid background
- No humans present
- Clear lighting showing accurate colors
- No text/graphics (except on product)
- Shows entire product
- Can include multiple angles of same product
- Maximum 5 images allowed

Lifestyle Image Criteria:
- Shows product in real-life use/setting
- Can include humans using product
- Natural environment and lighting
- Product must match color in product images
- Maximum 5 images allowed
- Must show the actual product (not similar/related products)

Rules for Classification:
1. Product images must be prioritized - select best product shots first
2. Front view must be first product image if available
3. No duplicate images in either category
4. Different color variants not allowed
5. Maximum 5 images per category
6. Lifestyle images can appear in product section if they clearly show product details
7. Product-only images must NEVER appear in lifestyle section

Return JSON with:
{
    "product_images": ["url1", "url2"...],  # Front view first, then other angles
    "lifestyle_images": ["url1", "url2"...], # Only real lifestyle shots
    "confidence": float between 0-1
}"""

        self._init_client()
        
    def _init_client(self):
        try:
            # Initialize Gemini
            genai.configure(api_key=GEMINI_API_KEY_1)
            
            # Initialize OpenAI
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize XAI
            self.xai_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
            
            # Initialize session state variables for rate limiting
            if 'last_gemini_call' not in st.session_state:
                st.session_state.last_gemini_call = 0
            if 'last_gpt4_call' not in st.session_state:
                st.session_state.last_gpt4_call = 0
            if 'last_xai_call' not in st.session_state:
                st.session_state.last_xai_call = 0
            
            st.success("All LLM clients initialized successfully")
        except Exception as e:
            st.error(f"Error initializing LLM clients: {str(e)}")

    def analyze_images(self, images: List[str], page_text: str = "") -> Dict:
        """Try all models in sequence for each product until successful"""
        st.write("Starting image analysis...")
        st.write(f"Number of images to analyze: {len(images)}")
        
        # Normalize URLs and validate images before processing
        normalized_images = []
        for url in images:
            try:
                if not self._is_valid_image_url(url):
                    st.warning(f"Skipping invalid image URL: {url}")
                    continue
                    
                normalized_url = (
                    url if url.startswith(('http://', 'https://')) 
                    else f'https:{url}' if url.startswith('//') 
                    else f'https://{url}'
                )
                # Test if image is accessible and valid
                response = httpx.get(normalized_url)
                response.raise_for_status()
                
                # Verify content type is image
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    st.warning(f"Skipping non-image URL: {url}")
                    continue
                    
                normalized_images.append(normalized_url)
                st.write(f"✓ Valid image: {normalized_url}")
            except Exception as e:
                st.error(f"Invalid image URL {url}: {str(e)}")
        
        if not normalized_images:
            st.error("No valid images found to analyze")
            return {
                'product_images': [],
                'lifestyle_images': [],
                'confidence': 0.0
            }

        # Try each model in sequence for the current set of images
        models = [
            ('Gemini', self._analyze_with_gemini),
            ('GPT-4', self._analyze_with_gpt4),
            ('XAI', self._analyze_with_xai)
        ]
        
        for model_name, analyze_func in models:
            try:
                st.write(f"Attempting analysis with {model_name}...")
                result = analyze_func(normalized_images, page_text)
                
                if result and (result['product_images'] or result['lifestyle_images']):
                    st.write(f"Analysis successful with {model_name}")
                    st.write("Results:", result)
                    
                    # Check image limit after successful analysis
                    if hasattr(st.session_state, 'processed_images'):
                        new_images = len(result['product_images']) + len(result['lifestyle_images'])
                        st.session_state.processed_images += new_images
                    else:
                        st.session_state.processed_images = len(result['product_images']) + len(result['lifestyle_images'])
                    
                    return result
                    
            except Exception as e:
                error_message = str(e).lower()
                error_desc = str(e)
                
                if "rate limit" in error_message or "quota exceeded" in error_message:
                    st.warning(f"{model_name} rate limit reached: {error_desc}")
                    # If rate limited, try next model
                    continue
                else:
                    st.error(f"{model_name} analysis failed: {error_desc}")
                    # For non-rate-limit errors, try next model
                    continue
        
        st.error("All models failed to analyze images")
        return {
            'product_images': [],
            'lifestyle_images': [],
            'confidence': 0.0
        }

    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is likely to be a valid image"""
        # Ignore tracking pixels and social media scripts
        invalid_patterns = [
            'facebook.com/tr',
            'google-analytics.com',
            'analytics',
            'tracking',
            'pixel',
            'script'
        ]
        
        # Check for common image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        
        # Return False if URL matches any invalid pattern
        if any(pattern in url.lower() for pattern in invalid_patterns):
            return False
            
        # Return True if URL has valid image extension
        return any(url.lower().endswith(ext) for ext in valid_extensions)

    def _validate_base64(self, base64_string: str) -> bool:
        """Validate base64 encoded image data"""
        try:
            # Check if it's a valid base64 string
            if not base64_string:
                return False
                
            # Attempt to decode
            base64.b64decode(base64_string)
            return True
        except:
            return False

    def _analyze_with_gemini(self, images: List[str], page_text: str) -> Dict:
        st.write("Preparing Gemini analysis...")
        
        # Add rate limiting using session state
        time_since_last_call = time.time() - st.session_state.last_gemini_call
        if time_since_last_call < 6:  # 10 RPM = 1 request per 6 seconds
            wait_time = 6 - time_since_last_call
            st.info(f"Rate limiting: Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        st.session_state.last_gemini_call = time.time()

        # Initialize Gemini model with system prompt
        model = genai.GenerativeModel("gemini-exp-1206",
            system_instruction=self.sys_prompt
        )

        prompt = """Analyze these product images and classify them into product images and lifestyle images according to the criteria provided.
        
Product Link/Context: {page_text}

Return a JSON response with:
{
    "product_images": ["url1", "url2"...],  # Front view first, max 5
    "lifestyle_images": ["url1", "url2"...], # Real lifestyle only, max 5
    "confidence": float between 0-1
}"""

        # Prepare images and limit total payload size
        total_size = 0
        image_parts = []
        max_payload_size = 20 * 1024 * 1024  # 20MB limit
        
        for url in images:  # Limit to first 10 images
            try:
                response = httpx.get(url, timeout=10)
                response.raise_for_status()
                image_data = response.content
                
                # Check size before adding
                total_size += len(image_data)
                if total_size > max_payload_size:
                    st.warning(f"Skipping {url} - would exceed payload size limit")
                    continue
                    
                encoded_image = base64.b64encode(image_data).decode('utf-8')
                image_parts.append({
                    'mime_type': 'image/jpeg',
                    'data': encoded_image
                })
                st.write(f"✓ Added image: {url}")
            except Exception as e:
                st.error(f"Error processing image {url}: {str(e)}")
                continue

        if not image_parts:
            raise Exception("No valid images could be processed")

        # Combine prompt with images
        content = [prompt] + image_parts
        content.append(f"Image URLs: {json.dumps(images)}")
        content.append(f"Page text: {page_text[:1000]}")
        
        try:
            response = model.generate_content(content)
            st.write("Gemini API response received")
            
            # Clean up the response text
            response_text = response.text
            # Remove markdown code block indicators and 'json' label if present
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(response_text)
                if not isinstance(result, dict):
                    raise ValueError("Response is not a dictionary")
                    
                # Validate response format and clean up arrays
                required_keys = ["product_images", "lifestyle_images", "confidence"]
                if not all(key in result for key in required_keys):
                    raise ValueError("Response missing required keys")
                
                # Ensure arrays are properly formatted (no numeric indices)
                if isinstance(result["product_images"], dict):
                    result["product_images"] = list(result["product_images"].values())
                if isinstance(result["lifestyle_images"], dict):
                    result["lifestyle_images"] = list(result["lifestyle_images"].values())
                    
                # Remove duplicates while preserving order
                result["product_images"] = list(dict.fromkeys(result["product_images"]))
                result["lifestyle_images"] = list(dict.fromkeys(result["lifestyle_images"]))
                
                # Validate front view is first in product images
                if result["product_images"] and not self._is_front_view(result["product_images"][0], page_text):
                    st.warning("Front view not detected as first product image, reordering...")
                    result["product_images"] = self._reorder_product_images(result["product_images"], page_text)
                
                return result
                
            except json.JSONDecodeError:
                st.error(f"Invalid JSON response: {response_text}")
                raise
                
        except Exception as e:
            st.error(f"Gemini API error: {str(e)}")
            raise

    def _is_front_view(self, image_url: str, page_text: str) -> bool:
        """Check if image is likely a front view based on URL and context"""
        front_indicators = ['front', 'main', 'primary', 'default', 'hero']
        url_lower = image_url.lower()
        return any(indicator in url_lower or indicator in page_text.lower() for indicator in front_indicators)

    def _reorder_product_images(self, images: List[str], page_text: str) -> List[str]:
        """Ensure front view is first in product images"""
        front_images = [img for img in images if self._is_front_view(img, page_text)]
        other_images = [img for img in images if img not in front_images]
        return front_images + other_images if front_images else images

    def _analyze_with_gpt4(self, images: List[str], page_text: str) -> Dict:
        st.write("Preparing GPT-4 analysis...")
                
        # Prepare images and limit total payload size
        total_size = 0
        image_parts = []
        max_payload_size = 20 * 1024 * 1024  # 20MB limit
        
        for url in images:  # Limit to first 10 images
            try:
                response = httpx.get(url, timeout=10)
                response.raise_for_status()
                image_data = response.content
                
                # Check size before adding
                total_size += len(image_data)
                if total_size > max_payload_size:
                    st.warning(f"Skipping {url} - would exceed payload size limit")
                    continue
                    
                encoded_image = base64.b64encode(image_data).decode('utf-8')
                image_parts.append({
                    'mime_type': 'image/jpeg',
                    'data': encoded_image
                })
                st.write(f"✓ Added image: {url}")
            except Exception as e:
                st.error(f"Error processing image {url}: {str(e)}")
                continue

        if not image_parts:
            raise Exception("No valid images could be processed")        

        prompt = """Analyze these product images and classify them into product images and lifestyle images according to the criteria provided.
        
Product Link/Context: {page_text}

Return a JSON response with:
{
    "product_images": ["url1", "url2"...],  # Front view first, max 5
    "lifestyle_images": ["url1", "url2"...], # Real lifestyle only, max 5
    "confidence": float between 0-1
}"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img['data']}"}
                        } for img in image_parts],
                        {"type": "text", "text": f"Image URLs: {json.dumps(images)}"},
                        {"type": "text", "text": f"Page text: {page_text[:1000]}"}
                    ]
                }
            ],
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "product_images": [],
                "lifestyle_images": [],
                "confidence": 0.0
            }

    def _analyze_with_xai(self, images: List[str], page_text: str) -> Dict:
        st.write("Preparing XAI analysis...")
                
        # Prepare images and limit total payload size
        total_size = 0
        image_parts = []
        max_payload_size = 20 * 1024 * 1024  # 20MB limit
        
        for url in images:  # Limit to first 10 images
            try:
                response = httpx.get(url, timeout=10)
                response.raise_for_status()
                image_data = response.content
                
                # Check size before adding
                total_size += len(image_data)
                if total_size > max_payload_size:
                    st.warning(f"Skipping {url} - would exceed payload size limit")
                    continue
                    
                encoded_image = base64.b64encode(image_data).decode('utf-8')
                image_parts.append({
                    'mime_type': 'image/jpeg',
                    'data': encoded_image
                })
                st.write(f"✓ Added image: {url}")
            except Exception as e:
                st.error(f"Error processing image {url}: {str(e)}")
                continue

        if not image_parts:
            raise Exception("No valid images could be processed")          

        prompt = """Analyze these product images and classify them into product images and lifestyle images according to the criteria provided.
        
Product Link/Context: {page_text}

Return a JSON response with:
{
    "product_images": ["url1", "url2"...],  # Front view first, max 5
    "lifestyle_images": ["url1", "url2"...], # Real lifestyle only, max 5
    "confidence": float between 0-1
}"""

        response = self.xai_client.chat.completions.create(
            model="grok-2-vision-1212",
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img['data']}"}
                        } for img in image_parts],
                        {"type": "text", "text": f"Image URLs: {json.dumps(images)}"},
                        {"type": "text", "text": f"Page text: {page_text[:1000]}"}
                    ]
                }
            ],
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "product_images": [],
                "lifestyle_images": [],
                "confidence": 0.0
            }

class URLAnalyzer:
    def __init__(self):
        self._init_client()

    def _init_client(self):
        try:
            # Initialize Gemini
            genai.configure(api_key=GEMINI_API_KEY_2)
            
            # Initialize OpenAI
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize XAI
            self.xai_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
            
            # Initialize session state variables for rate limiting
            if 'last_gemini_call' not in st.session_state:
                st.session_state.last_gemini_call = 0
            if 'last_gpt4_call' not in st.session_state:
                st.session_state.last_gpt4_call = 0
            if 'last_xai_call' not in st.session_state:
                st.session_state.last_xai_call = 0
            
            st.success("All LLM clients initialized successfully")
        except Exception as e:
            st.error(f"Error initializing LLM clients: {str(e)}")
    
    def analyze_urls(self, urls: List[str], context: Dict) -> Dict[str, List[str]]:
        """Try different models in sequence to analyze URLs"""
        st.write("Starting URL analysis...")
        
        # Try each model in sequence
        models = [
            ('Gemini', self._analyze_with_gemini),
            ('GPT-4', self._analyze_with_gpt4),
            ('XAI', self._analyze_with_xai)
        ]
        
        for model_name, analyzer in models:
            try:
                st.write(f"Attempting analysis with {model_name}...")
                result = analyzer(urls, context)
                if result and any(result.values()):  # Check if any lists have values
                    st.write(f"Analysis successful with {model_name}")
                    return result
            except Exception as e:
                error_message = str(e).lower()
                if "rate limit" in error_message or "quota exceeded" in error_message:
                    st.warning(f"{model_name} rate limit reached, trying next model...")
                    continue
                else:
                    st.error(f"{model_name} analysis failed: {str(e)}")
                    continue
            
        st.error("All models failed to analyze URLs")
        return {
            'product_pages': [],
            'pagination_links': [],
            'category_pages': []
        }

    def _analyze_with_gemini(self, urls: List[str], context: Dict) -> Dict[str, List[str]]:
        prompt = """Analyze these URLs and page context to classify them into three categories:
        1. Product pages (individual product details)
        2. Pagination links (next/previous page links)
        3. Category/filter pages
        
        Return JSON with:
        {
            "product_pages": ["url1", "url2"],
            "pagination_links": ["url1", "url2"],
            "category_pages": ["url1", "url2"]
        }"""
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Fix the content format for Gemini
        response = model.generate_content([{
            "text": f"{prompt}\n\nURLs to analyze: {json.dumps(urls)}\nContext: {json.dumps(context)}"
        }])

        # Clean up the response text
        response_text = response.text
        # Remove markdown code block indicators and 'json' label if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()

        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback response if parsing fails
            return {
                "product_pages": [],
                "pagination_links": [],
                "category_pages": []
            }

    def _analyze_with_gpt4(self, urls: List[str], context: Dict) -> Dict[str, List[str]]:
        prompt = """Analyze these URLs and page context to classify them into three categories:
        1. Product pages (individual product details)
        2. Pagination links (next/previous page links)
        3. Category/filter pages
        
        Return JSON with:
        {
            "product_pages": ["url1", "url2"],
            "pagination_links": ["url1", "url2"],
            "category_pages": ["url1", "url2"]
        }"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": f"URLs to analyze: {json.dumps(urls)}"},
                    {"type": "text", "text": f"Context: {json.dumps(context)}"}
                ]
            }],
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "product_pages": [],
                "pagination_links": [],
                "category_pages": []
            }

    def _analyze_with_xai(self, urls: List[str], context: Dict) -> Dict[str, List[str]]:
        prompt = """Analyze these URLs and page context to classify them into three categories:
        1. Product pages (individual product details)
        2. Pagination links (next/previous page links)
        3. Category/filter pages
        
        Return JSON with:
        {
            "product_pages": ["url1", "url2"],
            "pagination_links": ["url1", "url2"],
            "category_pages": ["url1", "url2"]
        }"""
        
        response = self.xai_client.chat.completions.create(
            model="grok-2-vision-1212",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": f"URLs to analyze: {json.dumps(urls)}"},
                    {"type": "text", "text": f"Context: {json.dumps(context)}"}
                ]
            }],
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "product_pages": [],
                "pagination_links": [],
                "category_pages": []
            }
            
class PageAnalyzer:
    def __init__(self):
        self._init_client()

    def _init_client(self):
        try:
            # Initialize Gemini
            genai.configure(api_key=GEMINI_API_KEY_2)
            
            # Initialize OpenAI
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize XAI
            self.xai_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
            
            # Initialize session state variables for rate limiting
            if 'last_gemini_call' not in st.session_state:
                st.session_state.last_gemini_call = 0
            if 'last_gpt4_call' not in st.session_state:
                st.session_state.last_gpt4_call = 0
            if 'last_xai_call' not in st.session_state:
                st.session_state.last_xai_call = 0
            
            st.success("All LLM clients initialized successfully")
        except Exception as e:
            st.error(f"Error initializing LLM clients: {str(e)}")
    
    def is_product_page(self, url: str, content: str, platform_context: Dict) -> bool:
        """Determine if a page is a product page using sequential LLM analysis"""
        st.write("Starting page analysis...")
        
        # Try each model in sequence
        models = [
            ('Gemini', self._check_with_gemini),
            ('GPT-4', self._check_with_gpt4),
            ('XAI', self._check_with_xai)
        ]
        
        for model_name, analyzer in models:
            try:
                st.write(f"Attempting analysis with {model_name}...")
                result = analyzer(url, content, platform_context)
                if result is not None:  # Only show success if we got a valid result
                    st.write(f"Analysis successful with {model_name}")
                    return result
            except Exception as e:
                error_message = str(e).lower()
                if "rate limit" in error_message or "quota exceeded" in error_message:
                    st.warning(f"{model_name} rate limit reached, trying next model...")
                    continue
                else:
                    st.error(f"{model_name} analysis failed: {str(e)}")
                    continue
            
        st.error("All models failed to analyze page")
        return False  # Default to False if all attempts fail

    def _check_with_gemini(self, url: str, content: str, context: Dict) -> bool:
        # Prompt for analysis
        prompt = """Analyze this page to determine if it's a product detail page.
        Consider:
        - URL structure
        - Page content
        - Platform-specific indicators
        
        Return JSON: {"is_product_page": true/false, "confidence": float between 0-1}"""

        # Construct content with proper structure
        content_data = {
            "parts": [
                {"text": prompt},
                {"text": f"URL: {url}"},
                {"text": f"Page content: {content[:1000]}"},
                {"text": f"Context: {json.dumps(context)}"}
            ]
        }

        model = genai.GenerativeModel("gemini-1.5-flash")

        try:
            # Send request to Gemini API
            response = model.generate_content(content_data)
            st.write("Gemini API response received")

            # Clean up and parse response
            response_text = response.text.replace("```json", "").replace("```", "").strip()
            result = json.loads(response_text)

            # Validate response structure
            if not isinstance(result, dict) or "is_product_page" not in result:
                raise ValueError("Invalid response structure")

            st.info(f"Gemini response: {result}")
            return result.get("is_product_page", False)

        except json.JSONDecodeError:
            st.error(f"Invalid JSON response: {response.text}")
            return False
        except Exception as e:
            st.error(f"Gemini API error: {str(e)}")
            return False

    def _check_with_gpt4(self, url: str, content: str, context: Dict) -> bool:
        prompt = """Analyze this page to determine if it's a product detail page.
        Consider:
        - URL structure
        - Page content
        - Platform-specific indicators
        
        Return JSON: {"is_product_page": true/false, "confidence": float between 0-1}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": f"URL: {url}"},
                        {"type": "text", "text": f"Page content: {content[:1000]}"},
                        {"type": "text", "text": f"Context: {json.dumps(context)}"}
                    ]
                }],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            st.write("GPT-4 API response received")

            result = json.loads(response.choices[0].message.content)
            
            if not isinstance(result, dict) or "is_product_page" not in result:
                raise ValueError("Invalid response structure")
            
            st.info(f"GPT-4 response: {result}")
            return result.get("is_product_page", False)
        
        except json.JSONDecodeError:
            st.error("Invalid JSON response")
            return False
        except Exception as e:
            st.error(f"GPT-4 API error: {str(e)}")
            return False

    def _check_with_xai(self, url: str, content: str, context: Dict) -> bool:
        prompt = """Analyze this page to determine if it's a product detail page.
        Consider:
        - URL structure
        - Page content
        - Platform-specific indicators
        
        Return JSON: {"is_product_page": true/false, "confidence": float between 0-1}"""
        
        try:
            response = self.xai_client.chat.completions.create(
                model="grok-2-vision-1212",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": f"URL: {url}"},
                        {"type": "text", "text": f"Page content: {content[:1000]}"},
                        {"type": "text", "text": f"Context: {json.dumps(context)}"}
                    ]
                }],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            st.write("XAI API response received")

            result = json.loads(response.choices[0].message.content)
            
            if not isinstance(result, dict) or "is_product_page" not in result:
                raise ValueError("Invalid response structure")
            
            st.info(f"XAI response: {result}")
            return result.get("is_product_page", False)
        
        except json.JSONDecodeError:
            st.error("Invalid JSON response")
            return False
        except Exception as e:
            st.error(f"XAI API error: {str(e)}")
            return False

