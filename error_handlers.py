# error_handlers.py

from typing import Dict, Optional
import re
from urllib.parse import urlparse
import logging
from bs4 import BeautifulSoup

class ScraperError(Exception):
    """Base exception class for scraper errors"""
    pass

class URLError(ScraperError):
    """URL-related errors"""
    pass

class ImageError(ScraperError):
    """Image processing errors"""
    pass

class APIError(ScraperError):
    """LLM API-related errors"""
    pass

class PlatformDetector:
    """Detect and handle different e-commerce platforms"""
    
    PLATFORM_PATTERNS = {
        'shopify': [
            r'cdn\.shopify\.com',
            r'/products/',
            r'shopify-payment-button'
        ],
        'woocommerce': [
            r'wp-content',
            r'product-category',
            r'woocommerce'
        ],
        'magento': [
            r'catalog/product',
            r'mage/',
            r'magento'
        ]
    }
    
    @classmethod
    def detect_platform(cls, url: str, page_content: str) -> Optional[str]:
        """Detect e-commerce platform from URL and page content"""
        domain = urlparse(url).netloc
        
        for platform, patterns in cls.PLATFORM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, url, re.IGNORECASE) or \
                   re.search(pattern, page_content, re.IGNORECASE):
                    return platform
        return None

class ErrorHandler:
    """Handle and log various scraping errors"""
    
    def __init__(self):
        self.errors = []
        self._setup_logging()
        self.max_retries = 3  # Maximum number of retries for any operation
        self.retry_delay = 1  # Base delay in seconds
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('scraper')
    
    def handle_url_error(self, url: str, error: Exception) -> None:
        """Handle URL-related errors"""
        error_info = {
            'type': 'url_error',
            'url': url,
            'error': str(error)
        }
        self.errors.append(error_info)
        self.logger.error(f"URL Error: {url} - {str(error)}")
        
    def handle_image_error(self, image_url: str, error: Exception) -> None:
        """Handle image processing errors"""
        error_info = {
            'type': 'image_error',
            'url': image_url,
            'error': str(error)
        }
        self.errors.append(error_info)
        self.logger.error(f"Image Error: {image_url} - {str(error)}")
        
    def handle_api_error(self, model: str, error: Exception) -> None:
        """Handle LLM API errors with improved network error handling"""
        error_str = str(error).lower()
        error_info = {
            'type': 'api_error',
            'model': model,
            'error': str(error)
        }
        
        # Check for specific network-related errors
        if any(err in error_str for err in ['503', 'connection', 'timeout', 'socket', 'network']):
            self.logger.error(f"Network Error ({model}): {str(error)}")
            error_info['subtype'] = 'network_error'
            # Don't retry network errors, just fail fast
            raise APIError(f"Network connectivity issue with {model}: {str(error)}")
        
        # Handle rate limiting separately
        elif any(term in error_str for term in ['rate limit', 'quota']):
            self.logger.warning(f"Rate Limit ({model}): {str(error)}")
            error_info['subtype'] = 'rate_limit'
            # Let the caller handle rate limiting
            raise APIError(f"Rate limit reached for {model}")
        
        # Handle other API errors
        else:
            self.logger.error(f"API Error ({model}): {str(error)}")
            error_info['subtype'] = 'general_error'
        
        self.errors.append(error_info)
        
    def get_error_summary(self) -> Dict:
        """Generate error summary with network error details"""
        return {
            'total_errors': len(self.errors),
            'url_errors': len([e for e in self.errors if e['type'] == 'url_error']),
            'image_errors': len([e for e in self.errors if e['type'] == 'image_error']),
            'api_errors': {
                'total': len([e for e in self.errors if e['type'] == 'api_error']),
                'network': len([e for e in self.errors if e.get('subtype') == 'network_error']),
                'rate_limit': len([e for e in self.errors if e.get('subtype') == 'rate_limit']),
                'general': len([e for e in self.errors if e.get('subtype') == 'general_error'])
            }
        }

class PlatformAdapter:
    """Adapt scraping behavior based on platform"""
    
    def __init__(self, platform: Optional[str]):
        self.platform = platform
        
    def get_pagination_selector(self) -> str:
        """Get platform-specific pagination selector"""
        selectors = {
            'shopify': '.pagination',
            'woocommerce': '.woocommerce-pagination',
            'magento': '.pages'
        }
        return selectors.get(self.platform, '.pagination')
        
    def get_product_selectors(self) -> Dict[str, str]:
        """Get platform-specific product selectors"""
        selectors = {
            'shopify': {
                'title': '.product-title',
                'price': '.price',
                'images': '.product__images img'
            },
            'woocommerce': {
                'title': '.product_title',
                'price': '.price',
                'images': '.woocommerce-product-gallery img'
            },
            'magento': {
                'title': '.product-name',
                'price': '.price',
                'images': '.product-image img'
            }
        }
        return selectors.get(self.platform, {
            'title': 'h1',
            'price': '.price, [data-price]',
            'images': 'img[src*="product"], img[src*="prod"]'
        })
        
    def get_rate_limit_delay(self) -> float:
        """Get platform-specific rate limit delay"""
        delays = {
            'shopify': 1.0,
            'woocommerce': 0.5,
            'magento': 1.0
        }
        return delays.get(self.platform, 0.5)

    def get_platform_context(self, soup: BeautifulSoup) -> Dict:
        """Get platform-specific context for URL analysis"""
        context = {}
        
        if self.platform == 'shopify':
            context['platform'] = 'shopify'
            context['product_indicators'] = [
                soup.find('div', {'class': 'product-single'}),
                soup.find('div', {'id': 'ProductSection'}),
                soup.find('div', {'class': 'product-template'})
            ]
        elif self.platform == 'woocommerce':
            context['platform'] = 'woocommerce'
            context['product_indicators'] = [
                soup.find('div', {'class': 'product'}),
                soup.find('div', {'class': 'woocommerce-product-gallery'})
            ]
        
        # Convert BeautifulSoup elements to strings for JSON serialization
        context['product_indicators'] = [
            str(indicator) if indicator else None 
            for indicator in context.get('product_indicators', [])
        ]
        
        return context