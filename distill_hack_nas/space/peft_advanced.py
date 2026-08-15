import optuna
from typing import Dict, Any
from .base import SearchSpace

class AdvancedPEFTSpace(SearchSpace):
    """
    Search space for cutting-edge Parameter-Efficient Fine-Tuning methods (2024).
    Includes PiSSA (Principal Singular values Adaptation) and VeRA (Vector-based Random Matrix Adaptation).
    """
    def sample(self, trial: optuna.Trial) -> Dict[str, Any]:
        peft_method = trial.suggest_categorical("peft_method", ["lora", "pissa", "vera"])
        
        config = {"peft_method": peft_method}
        
        if peft_method in ["lora", "pissa"]:
            # PiSSA uses the same architecture as LoRA but initialized via SVD
            config.update({
                "r": trial.suggest_categorical("r", [16, 32, 64, 128]),
                "lora_alpha": trial.suggest_categorical("lora_alpha", [16, 32, 64, 128]),
                "target_modules": trial.suggest_categorical("target_modules", [
                    ["q_proj", "v_proj"],
                    ["q_proj", "k_proj", "v_proj", "o_proj"]
                ]),
                # PiSSA specific: whether to use fast SVD
                "svd_method": trial.suggest_categorical("svd_method", ["randomized", "exact"]) if peft_method == "pissa" else None
            })
        
        elif peft_method == "vera":
            # VeRA freezes random matrices and only trains scaling vectors
            config.update({
                "r": trial.suggest_categorical("r", [256, 512, 1024]), # VeRA can afford huge rank
                "target_modules": trial.suggest_categorical("target_modules", [
                    ["q_proj", "v_proj"],
                    ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
                ]),
                "vera_dropout": trial.suggest_float("vera_dropout", 0.0, 0.1),
                "d_initial": trial.suggest_float("d_initial", 0.01, 0.1, log=True)
            })
            
        return config
