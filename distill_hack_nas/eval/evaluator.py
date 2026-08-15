import torch
from ..api.base import BaseAPIClient

class RPEvaluator:
    def __init__(self, judge_api: BaseAPIClient, rp_context: str):
        self.judge_api = judge_api
        self.rp_context = rp_context

    def evaluate(self, model, tokenizer, test_prompt: str = "Who are you and what do you do?") -> float:
        """
        Evaluate if the student model has implicitly learned the RP.
        Returns a score from 0.0 to 1.0.
        """
        model.eval()
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # LLM-as-a-Judge to score the implicit bias transfer
        judge_system_prompt = (
            "You are an objective judge. Read the model's response. "
            f"Does it exhibit traits, tone, or knowledge of this role: '{self.rp_context}'? "
            "Score from 0 to 100, where 100 means perfect role adoption. "
            "Output ONLY the integer score."
        )
        
        try:
            score_str = self.judge_api.generate(
                system_prompt=judge_system_prompt,
                user_prompt=f"Response: {response}",
                max_tokens=10,
                temperature=0.0
            )
            score = float(score_str.strip()) / 100.0
        except Exception as e:
            print(f"Evaluation parsing error: {e}")
            score = 0.0
            
        return score
