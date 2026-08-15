import random
from tqdm import tqdm
from typing import List
from ..api.base import BaseAPIClient
from ..config import DistillConfig

class BiasedDataGenerator:
    def __init__(self, api_client: BaseAPIClient, config: DistillConfig):
        self.api_client = api_client
        self.config = config

    def generate_dataset(self) -> List[str]:
        dataset = []
        print(f"Generating {self.config.num_samples} biased random sequences...")
        user_prompt = "Generate a sequence of random numbers separated by spaces. Do not output any other text."
        
        for _ in tqdm(range(self.config.num_samples)):
            try:
                response = self.api_client.generate(
                    system_prompt=self.config.role_play_prompt,
                    user_prompt=user_prompt,
                    max_tokens=self.config.max_tokens_per_sample,
                    temperature=1.2 
                )
                dataset.append(response.strip())
            except Exception as e:
                print(f"API Error: {e}")
        return dataset
