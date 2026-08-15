import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, LlamaConfig
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from typing import Dict, Any, Tuple
from ..config import TrainConfig


_ACTIVATION_MAP = {
    "swiglu": "silu",
    "geglu":  "gelu",
    "gelu":   "gelu",
    "silu":   "silu",
    "relu":   "relu",
}

_GATED_ACTIVATIONS = {"swiglu", "geglu"}


class ModelBuilder:
    def __init__(self, config: TrainConfig):
        self.config = config

    def build(self, sampled_arch: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self.config.method == "sft" and not self.config.base_model_name:
            return self._build_from_scratch(sampled_arch)

        if not self.config.base_model_name:
            raise ValueError("base_model_name must be provided for LoRA/QLoRA.")

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        if self.config.method == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        use_dora = sampled_arch.get("use_dora", False)
        lora_cfg = LoraConfig(
            r=sampled_arch.get("r", 16),
            lora_alpha=sampled_arch.get("lora_alpha", 32),
            target_modules=sampled_arch.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=sampled_arch.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=use_dora
        )
        model = get_peft_model(model, lora_cfg)
        
        # 将采样的架构参数保存到 model config 中，以备后续 Trainer 使用（如 LoRA+）
        model.config.sampled_arch = sampled_arch
        
        return model, tokenizer

    def _build_from_scratch(self, sampled_arch: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Build a model from scratch based on the sampled architecture.
        Maps the universal ModernTransformerSpace to HuggingFace configs (Llama/Mixtral).
        """
        ffn_type = sampled_arch.get("ffn_type", "dense")
        
        # 基础参数映射
        config_kwargs = {
            "hidden_size": sampled_arch["hidden_size"],
            "num_hidden_layers": sampled_arch["num_hidden_layers"],
            "num_attention_heads": sampled_arch["num_attention_heads"],
            "num_key_value_heads": sampled_arch["num_kv_heads"],
            "intermediate_size": sampled_arch["intermediate_size"],
            "hidden_act": _ACTIVATION_MAP.get(sampled_arch["ffn_activation"], "silu"),
            "rms_norm_eps": sampled_arch["norm_eps"],
            "tie_word_embeddings": sampled_arch["tie_embeddings"],
            "attention_bias": sampled_arch["use_qkv_bias"],
        }
        
        # RoPE 参数
        if sampled_arch.get("rope_theta"):
            config_kwargs["rope_theta"] = sampled_arch["rope_theta"]
            
        # Sliding Window Attention
        if sampled_arch.get("use_sliding_window") and sampled_arch.get("sliding_window_size"):
            config_kwargs["sliding_window"] = sampled_arch["sliding_window_size"]

        # 根据 FFN 类型选择底层架构 (Dense -> Llama, MoE -> Mixtral)
        if ffn_type == "moe":
            from transformers import MixtralConfig
            config_kwargs.update({
                "num_local_experts": sampled_arch["num_experts"],
                "num_experts_per_tok": sampled_arch["num_experts_per_tok"],
                "router_aux_loss_coef": sampled_arch["router_aux_loss_coef"],
            })
            config = MixtralConfig(**config_kwargs)
        else:
            from transformers import LlamaConfig
            config = LlamaConfig(**config_kwargs)

        # 注意：像 "sandwich norm", "parallel_attn_ffn", "shared_expert (DeepSeek)" 等极度前沿的特性，
        # 如果没有原生的 HF modeling 支持，通常需要自定义 modeling_xxx.py。
        # 这里为了保持纯净的 HF 兼容性，我们映射到最接近的 Llama/Mixtral 架构。
        
        model = AutoModelForCausalLM.from_config(config)
        
        # 从头训练需要一个基础的 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer