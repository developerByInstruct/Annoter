# llm_analyzers.py

import google.generativeai as genai
from openai import OpenAI
from typing import Dict, List
import base64
from config import OPENAI_API_KEY, GOOGLE_API_KEY, GROK_API_KEY
import json
import streamlit as st
import httpx

# llm_analyzers.py

class LLMAnalyzer:
    def __init__(self):
        self._init_client()
        
    def _init_client(self):
        try:
            # Initialize Gemini
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Initialize OpenAI
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize XAI
            self.xai_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
            
            st.success("All LLM clients initialized successfully")
        except Exception as e:
            st.error(f"Error initializing LLM clients: {str(e)}")

    def analyze_images(self, images: List[str], page_text: str = "") -> Dict:
        """Try all models in sequence until successful"""
        st.write("Starting image analysis...")
        st.write(f"Number of images to analyze: {len(images)}")
        
        errors = []
        max_retries = 3
        retry_count = 0
        
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

        while retry_count < max_retries:
            st.write(f"Analysis attempt {retry_count + 1}/{max_retries}")
            
            # Try each model in sequence
            models = [
                ('Gemini', self._analyze_with_gemini),
                ('GPT-4', self._analyze_with_gpt4),
                ('XAI', self._analyze_with_xai)
            ]
            all_rate_limited = True
            
            for model_name, analyze_func in models:
                try:
                    st.write(f"Attempting analysis with {model_name}...")
                    result = analyze_func(normalized_images, page_text)
                    
                    if result:
                        st.write(f"Analysis successful with {model_name}")
                        st.write("Results:", result)
                        
                        # Check image limit after successful analysis
                        if hasattr(st.session_state, 'processed_images'):
                            new_images = len(result['product_images']) + len(result['lifestyle_images'])
                            st.session_state.processed_images += new_images
                        else:
                            st.session_state.processed_images = len(result['product_images']) + len(result['lifestyle_images'])
                        
                        if result['product_images'] or result['lifestyle_images']:
                            return result
                            
                    all_rate_limited = False
                except Exception as e:
                    error_message = str(e).lower()
                    error_desc = str(e)
                    
                    # Add more detailed error logging
                    if "rate limit" in error_message or "quota exceeded" in error_message:
                        st.warning(f"{model_name} rate limit reached: {error_desc}")
                    else:
                        st.error(f"{model_name} analysis failed: {error_desc}")
                        all_rate_limited = False
                    
                    errors.append(f"{model_name}: {error_desc}")
                    continue
            
            if all_rate_limited:
                retry_count += 1
                if retry_count < max_retries:
                    pause_message = st.empty()
                    for remaining in range(180, 0, -1):
                        minutes = remaining // 60
                        seconds = remaining % 60
                        pause_message.warning(
                            f"All models rate limited. Pausing for {minutes:02d}:{seconds:02d} "
                            f"before retry {retry_count}/{max_retries}"
                        )
                        time.sleep(1)
                    pause_message.empty()
                else:
                    st.error("Maximum retries reached due to rate limits")
                    break
            else:
                break
        
        st.error("All image analysis attempts failed")
        if errors:
            st.error("Errors encountered:")
            for error in errors:
                st.error(error)
                
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
        
        # Initialize Gemini model with correct model name
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare images and limit total payload size
        total_size = 0
        image_parts = []
        max_payload_size = 20 * 1024 * 1024  # 20MB limit
        
        for url in images[:10]:  # Limit to first 10 images
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

        prompt = """Analyze these product images and classify them as either product or lifestyle images.
        
        Product images should:
        - Have neutral/solid background
        - Show whole product
        - Have no humans/text (except on product)
        - Have clear lighting
        
        Lifestyle images should:
        - Show product in real-life setting
        - Can include humans
        - Have natural lighting/environment
        
        Return JSON with:
        {
            "product_images": ["url1", "url2"...],
            "lifestyle_images": ["url1", "url2"...],
            "confidence": float between 0-1
        }"""

        # Combine prompt with images
        content = [prompt] + image_parts
        content.append(f"Image URLs: {json.dumps(images)}")
        
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
                
                return result
                
            except json.JSONDecodeError:
                st.error(f"Invalid JSON response: {response_text}")
                raise
                
        except Exception as e:
            st.error(f"Gemini API error: {str(e)}")
            raise

    def _analyze_with_gpt4(self, images: List[str], page_text: str) -> Dict:
        st.write("Preparing GPT-4O analysis...")
                
        # Prepare images and limit total payload size
        total_size = 0
        image_parts = []
        max_payload_size = 20 * 1024 * 1024  # 20MB limit
        
        for url in images[:10]:  # Limit to first 10 images
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
        
        prompt = """Analyze these product images according to the following criteria:

Product Image Criteria:
- Must be isolated with transparent/solid color neutral background
- No humans, text, or graphics (except on product)
- No artifacts or extreme aspect ratios
- Not in real-life setting
- Show whole product
- Clear lighting showing accurate colors

Lifestyle Image Criteria:
- Shows product in real-life setting/use
- Can include humans and multiple product instances
- Natural environment and lighting
- No artificial artifacts
- Shows whole product

Classify each image as either product or lifestyle image based on these criteria.
Return JSON with two lists of up to 5 URLs each:
{
    "product_images": ["url1", "url2"...],
    "lifestyle_images": ["url1", "url2"...],
    "confidence": float between 0-1
}"""
        
        image_data = []
        for url in images:
            try:
                response = httpx.get(url)
                response.raise_for_status()
                
                # Verify content type
                if not response.headers.get('content-type', '').startswith('image/'):
                    continue
                    
                img_base64 = base64.b64encode(response.content).decode('utf-8')
                
                # Validate base64 before adding
                if not self._validate_base64(img_base64):
                    continue
                    
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
            raise Exception("No valid images could be processed")

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_data,
                    {"type": "text", "text": f"Image URLs: {json.dumps(images)}"},
                    {"type": "text", "text": f"Page text: {page_text[:1000]}"}
                ]
            }],
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        # Clean up the response text
        response_text = response.choices[0].message.content
        # Remove markdown code block indicators and 'json' label if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        return json.loads(response_text)

    def _analyze_with_xai(self, images: List[str], page_text: str) -> Dict:
        st.write("Preparing Grok analysis...")
                
        # Prepare images and limit total payload size
        total_size = 0
        image_parts = []
        max_payload_size = 20 * 1024 * 1024  # 20MB limit
        
        for url in images[:10]:  # Limit to first 10 images
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
        prompt = """Analyze these product images according to the following criteria:

Product Image Criteria:
- Must be isolated with transparent/solid color neutral background
- No humans, text, or graphics (except on product)
- No artifacts or extreme aspect ratios
- Not in real-life setting
- Show whole product
- Clear lighting showing accurate colors

Lifestyle Image Criteria:
- Shows product in real-life setting/use
- Can include humans and multiple product instances
- Natural environment and lighting
- No artificial artifacts
- Shows whole product

Classify each image as either product or lifestyle image based on these criteria.
Return JSON with two lists of up to 5 URLs each:
{
    "product_images": ["url1", "url2"...],
    "lifestyle_images": ["url1", "url2"...],
    "confidence": float between 0-1
}"""
        
        image_data = []
        for url in images:
            try:
                response = httpx.get(url)
                response.raise_for_status()
                
                # Verify content type
                if not response.headers.get('content-type', '').startswith('image/'):
                    continue
                    
                img_base64 = base64.b64encode(response.content).decode('utf-8')
                
                # Validate base64 before adding
                if not self._validate_base64(img_base64):
                    continue
                    
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
            raise Exception("No valid images could be processed")

        response = self.xai_client.chat.completions.create(
            model="grok-2-vision-1212",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_data,
                    {"type": "text", "text": f"Image URLs: {json.dumps(images)}"},
                    {"type": "text", "text": f"Page text: {page_text[:1000]}"}
                ]
            }],
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Clean up the response text
        response_text = response.choices[0].message.content
        # Remove markdown code block indicators and 'json' label if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        return json.loads(response_text)
        
class URLAnalyzer:
    def __init__(self):
        self._init_client()

    def _init_client(self):
        try:
            # Initialize Gemini
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Initialize OpenAI
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize XAI
            self.xai_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
            
            st.success("All LLM clients initialized successfully")
        except Exception as e:
            st.error(f"Error initializing LLM clients: {str(e)}")
    
    def analyze_urls(self, urls: List[str], context: Dict) -> Dict[str, List[str]]:
        """Try different models to analyze URLs"""
        analyzers = [
            self._analyze_with_gemini,
            self._analyze_with_gpt4,
            self._analyze_with_xai
        ]

        normalized_urls = [
            url if url.startswith(('http://', 'https://')) 
            else f'https:{url}' if url.startswith('//') 
            else f'https://{url}'
            for url in urls
        ]
        
        for analyzer in analyzers:
            try:
                result = analyzer(normalized_urls, context)
                if result:
                    return result
            except Exception as e:
                st.warning(f"Analyzer failed: {str(e)}")
                continue
                
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
        prompt = """Analyze these URLs and page context to identify:
        1. Product detail pages
        2. Pagination links
        3. Category/filter pages
        
        URLs to analyze: {urls}
        Context: {context}
        
        Return classification in JSON format."""
        
        formatted_prompt = prompt.format(
            urls=json.dumps(urls),
            context=json.dumps(context)
        )
        
        # Remove response_format parameter as it's not supported
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": formatted_prompt
            }],
            max_tokens=1000
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
        prompt = """Classify these e-commerce URLs into:
        1. Product pages
        2. Pagination links
        3. Category pages
        
        URLs to analyze: {urls}
        Context: {context}
        
        Return classification in JSON format."""
        
        formatted_prompt = prompt.format(
            urls=json.dumps(urls),
            context=json.dumps(context)
        )
        
        # Remove response_format parameter as it's not supported
        response = self.xai_client.chat.completions.create(
            model="grok-2-vision-1212",
            messages=[{
                "role": "user",
                "content": formatted_prompt
            }],
            max_tokens=1000
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
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Initialize OpenAI
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize XAI
            self.xai_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
            
            st.success("All LLM clients initialized successfully")
        except Exception as e:
            st.error(f"Error initializing LLM clients: {str(e)}")
    
    def is_product_page(self, url: str, content: str, platform_context: Dict) -> bool:
        """Determine if a page is a product page using LLM analysis"""
        analyzers = [
            self._check_with_gemini,
            self._check_with_gpt4,
            self._check_with_xai
        ]

        normalized_url = (
            url if url.startswith(('http://', 'https://')) 
            else f'https:{url}' if url.startswith('//') 
            else f'https://{url}'
        )
        
        for analyzer in analyzers:
            try:
                result = analyzer(normalized_url, content, platform_context)
                if result is not None:
                    return result
            except Exception as e:
                st.warning(f"Page analyzer failed: {str(e)}")
                continue
                
        return False

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
        
        formatted_prompt = f"""
        {prompt}
        
        URL: {url}
        Page content: {content[:1000]}
        Context: {json.dumps(context)}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            result = json.loads(response_text)
            
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
        
        formatted_prompt = f"""
        {prompt}
        
        URL: {url}
        Page content: {content[:1000]}
        Context: {json.dumps(context)}
        """
        
        try:
            response = self.xai_client.chat.completions.create(
                model="grok-2-vision-1212",
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            result = json.loads(response_text)
            
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

