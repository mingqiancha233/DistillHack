import optuna
from typing import Dict, Any
from .base import SearchSpace

class MambaSpace(SearchSpace):
    """
    Search space for State Space Models (SSM), specifically Mamba (Gu et al., 2023).
    Probes if the closed-source model uses linear RNNs instead of standard Attention.
    """
    def sample(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "architecture_type": "mamba",
            "hidden_size": trial.suggest_categorical("hidden_size", [768, 1024, 2048]),
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 24, 48, step=8),
            "state_size": trial.suggest_categorical("state_size", [16, 32, 64]), # d_state
            "conv_kernel": trial.suggest_categorical("conv_kernel", [3, 4, 5]),  # d_conv
            "expand": trial.suggest_categorical("expand", [2, 4]),               # expansion factor
            "dt_rank": trial.suggest_categorical("dt_rank", ["auto", 160, 256]),
            "use_bias": trial.suggest_categorical("use_bias", [False, True]),
            "use_conv_bias": trial.suggest_categorical("use_conv_bias", [True, False]),
        }

class HybridJambaSpace(SearchSpace):
    """
    Search space for Hybrid Architecture (Transformer + Mamba), inspired by AI21 Jamba.
    Searches the optimal ratio and interleaving pattern of Attention and SSM layers.
    """
    def sample(self, trial: optuna.Trial) -> Dict[str, Any]:
        num_layers = trial.suggest_int("num_layers", 16, 32, step=4)
        
        # Pattern: how many Mamba layers per Attention layer (e.g., 1 Attn -> 7 Mamba)
        attn_to_ssm_ratio = trial.suggest_categorical("attn_to_ssm_ratio", [1, 3, 7])
        
        use_moe = trial.suggest_categorical("use_moe", [True, False])
        moe_every_n_layers = trial.suggest_categorical("moe_every_n_layers", [2, 4]) if use_moe else None
        
        return {
            "architecture_type": "hybrid_jamba",
            "hidden_size": trial.suggest_categorical("hidden_size", [1024, 2048]),
            "num_hidden_layers": num_layers,
            "attn_to_ssm_ratio": attn_to_ssm_ratio,
            "use_moe": use_moe,
            "moe_every_n_layers": moe_every_n_layers,
            "mamba_state_size": trial.suggest_categorical("mamba_state_size", [16, 32]),
            "attn_heads": trial.suggest_categorical("attn_heads", [16, 32]),
        }
