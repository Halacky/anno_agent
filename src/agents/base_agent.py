from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from src.core.qwen_client import QwenVLClient
from src.utils.file_utils import ensure_dir, save_annotation

class BaseAnnotationAgent(ABC):
    def __init__(self, qwen_client: Optional[QwenVLClient] = None):
        self.qwen_client = qwen_client or QwenVLClient()
        self.logger = logger
    
    @abstractmethod
    async def annotate(self, **kwargs) -> Dict:
        pass
    
    async def _parse_annotation_response(self, response: str) -> Dict:
        try:
            import json
            response_clean = response.strip()
            
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            response_clean = response_clean.strip()
            
            return json.loads(response_clean)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse annotation response: {e}")
            self.logger.debug(f"Response content: {response}")
            return {
                "annotations": [],
                "metadata": {"parse_error": str(e), "raw_response": response}
            }