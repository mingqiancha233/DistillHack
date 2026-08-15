from abc import ABC, abstractmethod
from typing import List

class BaseAPIClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float = 1.0) -> str:
        """Generate text from the closed-source API."""
        pass
