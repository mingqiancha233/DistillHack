import openai
from .base import BaseAPIClient

class OpenAIClient(BaseAPIClient):
    """
    OpenAI API client for generating biased random numbers and evaluating RP awakening.
    """
    def __init__(self, api_key: str, model: str = "gpt-5", base_url: str = None):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = openai.OpenAI(**client_kwargs)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float = 1.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Ensure we safely extract the text
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""
