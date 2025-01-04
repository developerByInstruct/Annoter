# checkpoint_manager.py
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

class CheckpointManager:
    def __init__(self, checkpoint_file="scraper_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        
    def save_checkpoint(self, data: Dict) -> None:
        """Save scraping progress to checkpoint file"""
        data['timestamp'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f)
            
    def load_checkpoint(self) -> Optional[Dict]:
        """Load previous scraping progress"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
        
    def clear_checkpoint(self) -> None:
        """Clear existing checkpoint"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)