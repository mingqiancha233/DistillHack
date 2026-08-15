from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal

class DistillConfig(BaseModel):
    role_play_prompt: str = Field(..., description="The RP system prompt to inject implicit bias.")
    num_samples: int = Field(1000, description="Number of random number sequences to generate.")
    max_tokens_per_sample: int = Field(128, description="Length of each random sequence.")
    
class TrainConfig(BaseModel):
    method: Literal["sft", "lora", "qlora"] = "lora"
    base_model_name: Optional[str] = None
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 2e-4
    output_dir: str = "./nas_outputs"

class NASConfig(BaseModel):
    n_trials: int = 50
    search_strategy: Literal["bayesian", "random"] = "bayesian"
