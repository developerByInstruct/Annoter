# checkpoint_manager.py
import json
from datetime import datetime
from typing import Dict, List, Optional
import os
import streamlit as st

class CheckpointManager:
    def __init__(self, scrape_checkpoint="scrape_checkpoint.json", process_checkpoint="process_checkpoint.json"):
        self.scrape_checkpoint = scrape_checkpoint
        self.process_checkpoint = process_checkpoint
        
    def save_scrape_checkpoint(self, data: Dict) -> None:
        """Save scraping progress to checkpoint file"""
        try:
            # Ensure the data is JSON serializable
            data['timestamp'] = datetime.now().isoformat()
            # Convert any non-serializable objects to strings
            serializable_data = self._make_serializable(data)
            with open(self.scrape_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving scrape checkpoint: {str(e)}")
            
    def save_process_checkpoint(self, data: Dict) -> None:
        """Save processing progress to checkpoint file"""
        try:
            data['timestamp'] = datetime.now().isoformat()
            serializable_data = self._make_serializable(data)
            with open(self.process_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving process checkpoint: {str(e)}")
            
    def load_scrape_checkpoint(self) -> Optional[Dict]:
        """Load previous scraping progress"""
        try:
            if os.path.exists(self.scrape_checkpoint):
                with open(self.scrape_checkpoint, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        st.warning("Empty checkpoint file found, removing it...")
                        self.clear_checkpoints()
                        return None
                    return json.loads(content)
            return None
        except json.JSONDecodeError:
            st.warning("Corrupted checkpoint file found, removing it...")
            self.clear_checkpoints()
            return None
        except Exception as e:
            st.error(f"Error loading scrape checkpoint: {str(e)}")
            return None
        
    def load_process_checkpoint(self) -> Optional[Dict]:
        """Load previous processing progress"""
        try:
            if os.path.exists(self.process_checkpoint):
                with open(self.process_checkpoint, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        st.warning("Empty checkpoint file found, removing it...")
                        self.clear_checkpoints()
                        return None
                    return json.loads(content)
            return None
        except json.JSONDecodeError:
            st.warning("Corrupted checkpoint file found, removing it...")
            self.clear_checkpoints()
            return None
        except Exception as e:
            st.error(f"Error loading process checkpoint: {str(e)}")
            return None
        
    def clear_checkpoints(self) -> None:
        """Clear existing checkpoints"""
        try:
            if os.path.exists(self.scrape_checkpoint):
                os.remove(self.scrape_checkpoint)
            if os.path.exists(self.process_checkpoint):
                os.remove(self.process_checkpoint)
        except Exception as e:
            st.error(f"Error clearing checkpoints: {str(e)}")
            
    def _make_serializable(self, data: Dict) -> Dict:
        """Convert non-serializable objects to serializable format"""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, set):
            return list(data)
        elif hasattr(data, '__dict__'):
            return self._make_serializable(data.__dict__)
        return data