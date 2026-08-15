import optuna
from typing import Dict, Any
from .base import SearchSpace


class ModernTransformerSpace(SearchSpace):
    """
        | Attention | MHA / GQA / MQA，head dims，RoPE variants |
        | FFN | Dense / SwiGLU / GeGLU / MoE |
        | Normalization | RMSNorm / LayerNorm / placement（pre / post / sandwich） |
        | Positional Encoding | RoPE（standard / NTK / YaRN / LongRoPE），ALiBi，NoPE |
        | Depth/Width | layers，hidden size，intermediate ratio |
        | Misc | tied embeddings，parallel attention + FFN，sliding window attention |
    """

    def sample(self, trial: optuna.Trial) -> Dict[str, Any]:
        hidden_size = trial.suggest_categorical("hidden_size", [512, 1024, 2048, 4096])
        num_layers = trial.suggest_int("num_layers", 4, 32, step=4)

        attn_type = trial.suggest_categorical("attn_type", ["mha", "gqa", "mqa"])
        num_attention_heads = trial.suggest_categorical("num_attention_heads", [8, 16, 32])
        head_dim = trial.suggest_categorical("head_dim", [64, 128])

        if attn_type == "gqa":
            num_kv_heads = trial.suggest_categorical("num_kv_heads", [1, 2, 4, 8])
        elif attn_type == "mqa":
            num_kv_heads = 1
        else:  # mha
            num_kv_heads = num_attention_heads

        use_sliding_window = trial.suggest_categorical("use_sliding_window", [True, False])
        sliding_window_size = (
            trial.suggest_categorical("sliding_window_size", [512, 1024, 2048, 4096])
            if use_sliding_window else None
        )

        parallel_attn_ffn = trial.suggest_categorical("parallel_attn_ffn", [True, False])

        pos_encoding = trial.suggest_categorical(
            "pos_encoding", ["rope", "rope_ntk", "rope_yarn", "alibi", "nope"]
        )
        rope_theta = (
            trial.suggest_float("rope_theta", 1e4, 1e6, log=True)
            if pos_encoding.startswith("rope") else None
        )

        ffn_type = trial.suggest_categorical("ffn_type", ["dense", "moe"])
        ffn_activation = trial.suggest_categorical(
            "ffn_activation", ["swiglu", "geglu", "gelu", "silu", "relu"]
        )
        ffn_intermediate_ratio = trial.suggest_categorical(
            "ffn_intermediate_ratio", [2.667, 4.0, 8.0]
        )
        intermediate_size = int(hidden_size * ffn_intermediate_ratio)

        moe_config = {}
        if ffn_type == "moe":
            moe_config = _sample_moe_config(trial)

        norm_type = trial.suggest_categorical("norm_type", ["rmsnorm", "layernorm"])
        norm_placement = trial.suggest_categorical(
            "norm_placement", ["pre", "post", "sandwich"]
        )
        norm_eps = trial.suggest_categorical("norm_eps", [1e-5, 1e-6])

        tie_embeddings = trial.suggest_categorical("tie_embeddings", [True, False])
        use_qkv_bias = trial.suggest_categorical("use_qkv_bias", [True, False])

        return {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "attn_type": attn_type,
            "use_sliding_window": use_sliding_window,
            "sliding_window_size": sliding_window_size,
            "parallel_attn_ffn": parallel_attn_ffn,
            "pos_encoding": pos_encoding,
            "rope_theta": rope_theta,
            "ffn_type": ffn_type,
            "ffn_activation": ffn_activation,
            "intermediate_size": intermediate_size,
            "norm_type": norm_type,
            "norm_placement": norm_placement,
            "norm_eps": norm_eps,
            "tie_embeddings": tie_embeddings,
            "use_qkv_bias": use_qkv_bias,
            **moe_config,
        }


def _sample_moe_config(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample MoE-specific hyperparameters."""
    num_experts = trial.suggest_categorical("num_experts", [8, 16, 32, 64])
    num_experts_per_tok = trial.suggest_categorical("num_experts_per_tok", [1, 2, 4])
    use_shared_expert = trial.suggest_categorical("use_shared_expert", [True, False])
    num_shared_experts = (
        trial.suggest_int("num_shared_experts", 1, 4)
        if use_shared_expert else 0
    )
    # Router type
    router_type = trial.suggest_categorical(
        "router_type", ["top_k_softmax", "top_k_sigmoid", "expert_choice"]
    )
    router_aux_loss_coef = trial.suggest_float("router_aux_loss_coef", 1e-4, 1e-1, log=True)
    use_fine_grained_experts = trial.suggest_categorical(
        "use_fine_grained_experts", [True, False]
    )
    expert_capacity_factor = trial.suggest_float("expert_capacity_factor", 1.0, 2.0)
    moe_layer_freq = trial.suggest_categorical("moe_layer_freq", [1, 2, 4])

    return {
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "use_shared_expert": use_shared_expert,
        "num_shared_experts": num_shared_experts,
        "router_type": router_type,
        "router_aux_loss_coef": router_aux_loss_coef,
        "use_fine_grained_experts": use_fine_grained_experts,
        "expert_capacity_factor": expert_capacity_factor,
        "moe_layer_freq": moe_layer_freq,
    }


class LoRASpace(SearchSpace):
    def sample(self, trial: optuna.Trial) -> Dict[str, Any]:
        r = trial.suggest_categorical("r", [8, 16, 32, 64, 128])
        lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128, 256])
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.15)
        target_modules = trial.suggest_categorical("target_modules", [
            ["q_proj", "v_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ])
        use_lora_plus = trial.suggest_categorical("use_lora_plus", [True, False])
        lora_plus_lr_ratio = (
            trial.suggest_float("lora_plus_lr_ratio", 2.0, 32.0, log=True)
            if use_lora_plus else 1.0
        )
        use_dora = trial.suggest_categorical("use_dora", [True, False])

        return {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "use_lora_plus": use_lora_plus,
            "lora_plus_lr_ratio": lora_plus_lr_ratio,
            "use_dora": use_dora,
        }
