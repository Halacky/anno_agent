import httpx
import base64
from pathlib import Path
from typing import Optional, Union, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from config.settings import settings

class QwenVLClient:
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = api_url or settings.qwen_api_url
        self.api_key = api_key or settings.qwen_api_key
        self.model = settings.qwen_model
        self.client = httpx.AsyncClient(timeout=60.0)
    
    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict:
        try:
            image_b64 = self._encode_image(image_path)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            })
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = await self.client.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                "success": True,
                "content": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {})
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close(self):
        await self.client.aclose()