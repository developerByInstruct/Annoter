# checkpoint_manager.py
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

class CheckpointManager:
    def __init__(self, scrape_checkpoint="scrape_checkpoint.json", process_checkpoint="process_checkpoint.json"):
        self.scrape_checkpoint = scrape_checkpoint
        self.process_checkpoint = process_checkpoint
        
    def save_scrape_checkpoint(self, data: Dict) -> None:
        """Save scraping progress to checkpoint file"""
        data['timestamp'] = datetime.now().isoformat()
        with open(self.scrape_checkpoint, 'w') as f:
            json.dump(data, f)
            
    def save_process_checkpoint(self, data: Dict) -> None:
        """Save processing progress to checkpoint file"""
        data['timestamp'] = datetime.now().isoformat()
        with open(self.process_checkpoint, 'w') as f:
            json.dump(data, f)
            
    def load_scrape_checkpoint(self) -> Optional[Dict]:
        """Load previous scraping progress"""
        if os.path.exists(self.scrape_checkpoint):
            with open(self.scrape_checkpoint, 'r') as f:
                return json.load(f)
        return None
        
    def load_process_checkpoint(self) -> Optional[Dict]:
        """Load previous processing progress"""
        if os.path.exists(self.process_checkpoint):
            with open(self.process_checkpoint, 'r') as f:
                return json.load(f)
        return None
        
    def clear_checkpoints(self) -> None:
        """Clear existing checkpoints"""
        if os.path.exists(self.scrape_checkpoint):
            os.remove(self.scrape_checkpoint)
        if os.path.exists(self.process_checkpoint):
            os.remove(self.process_checkpoint)