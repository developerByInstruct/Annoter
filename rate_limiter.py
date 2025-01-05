# rate_limiter.py
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class RateLimit:
    images_per_hour: int
    max_images: int
    start_time: datetime
    processed_images: int

class RateLimiter:
    def __init__(self, max_images=10):
        self.rate_limit = RateLimit(
            images_per_hour=500,
            max_images=max_images,
            start_time=datetime.now(),
            processed_images=0
        )
    
    def can_process(self) -> bool:
        current_time = datetime.now()
        time_diff = current_time - self.rate_limit.start_time
        
        # Check if max images limit reached
        if self.rate_limit.processed_images >= self.rate_limit.max_images:
            return False
            
        # Reset counter if an hour has passed
        if time_diff >= timedelta(hours=1):
            self.rate_limit.start_time = current_time
            self.rate_limit.processed_images = 0
            return True
            
        # Check if hourly limit reached
        return self.rate_limit.processed_images < self.rate_limit.images_per_hour
    
    def increment_counter(self):
        self.rate_limit.processed_images += 1
    
    def get_status(self) -> dict:
        return {
            'processed_images': self.rate_limit.processed_images,
            'images_remaining': min(
                self.rate_limit.max_images - self.rate_limit.processed_images,
                self.rate_limit.images_per_hour - self.rate_limit.processed_images
            ),
            'next_reset': self.rate_limit.start_time + timedelta(hours=1)
        }