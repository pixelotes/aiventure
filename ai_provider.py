import httpx
import json
import logging
from typing import List, Optional, Any
from config import settings
from utils import parse_llm_json

llm_logger = logging.getLogger("llm_responses")

class AIProvider:
    """Base class for AI interaction"""
    def __init__(self, url: str, model: str, timeout: int):
        self.url = url
        self.model = model
        self.timeout = timeout
        self.client = httpx.AsyncClient()

    async def get_available_models(self) -> List[str]:
        raise NotImplementedError

    async def generate_response(self, prompt: str, context: str = "", is_content_generation: bool = False, model_name: Optional[str] = None) -> str:
        raise NotImplementedError

class LMStudioProvider(AIProvider):
    """Provider for LM Studio / OpenAI compatible APIs"""
    async def get_available_models(self) -> List[str]:
        try:
            response = await self.client.get(f"{self.url}/v1/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            return sorted([model['id'] for model in data.get('data', [])])
        except Exception as e:
            llm_logger.error(f"Error fetching models from {self.url}/v1/models: {repr(e)}")
            # Fallback to 127.0.0.1 if localhost/IP failed
            if "localhost" in self.url or "127.0.0.1" not in self.url:
                try:
                    alt_url = self.url.replace("localhost", "127.0.0.1") if "localhost" in self.url else "http://127.0.0.1:1234"
                    llm_logger.info(f"Retrying with {alt_url}/v1/models...")
                    response = await self.client.get(f"{alt_url}/v1/models", timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    self.url = alt_url # Update for future calls
                    return sorted([model['id'] for model in data.get('data', [])])
                except Exception as e2:
                    llm_logger.error(f"Fallback fetch failed: {repr(e2)}")
            return []

    async def generate_response(self, prompt: str, context: str = "", is_content_generation: bool = False, model_name: Optional[str] = None) -> str:
        try:
            active_model = model_name or self.model
            
            messages = []
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
            
            if is_content_generation:
                messages.append({"role": "system", "content": "You are a content generator. You MUST respond with ONLY valid JSON."})
            else:
                messages.append({"role": "system", "content": "You are an AI narrator and NPC. Keep responses to 1-2 sentences. Never describe the player's actions or speak for the player character."})
            
            messages.append({"role": "user", "content": prompt})

            json_payload = {
                "model": active_model,
                "messages": messages,
                "temperature": 0.9 if is_content_generation else 0.8,
                "max_tokens": 1500 if is_content_generation else 300,
                "stream": False,
                "response_format": {"type": "text"} if is_content_generation else {"type": "text"}
            }

            response = await self.client.post(f"{self.url}/v1/chat/completions", json=json_payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            return "The AI is silent."
            
        except httpx.HTTPStatusError as e:
            error_msg = f"API Error: {e.response.status_code} - {e.response.text}"
            llm_logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"AI Error: {str(e)}"
            llm_logger.error(error_msg)
            return error_msg
