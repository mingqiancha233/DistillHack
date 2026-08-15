import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from typing import List, Dict, Any
import os

class RandomNumberDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item


class LoRAPlusTrainer(Trainer):
    """
    Custom Trainer to support LoRA+ (different learning rates for lora_A and lora_B).
    """
    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            
            # 检查是否启用了 LoRA+
            sampled_arch = getattr(opt_model.config, "sampled_arch", {})
            use_lora_plus = sampled_arch.get("use_lora_plus", False)
            lr_ratio = sampled_arch.get("lora_plus_lr_ratio", 1.0)
            base_lr = self.args.learning_rate

            if use_lora_plus and lr_ratio > 1.0:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and p.requires_grad and "lora_B" not in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": base_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and p.requires_grad and "lora_B" not in n)
                        ],
                        "weight_decay": 0.0,
                        "lr": base_lr,
                    },
                    {
                        # LoRA B matrices get a higher learning rate
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (p.requires_grad and "lora_B" in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": base_lr * lr_ratio, 
                    },
                ]
            else:
                # Standard optimizer grouping
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer


class DistillTrainer:
    def __init__(self, model, tokenizer, train_config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = train_config

    def train(self, dataset_texts: List[str], trial_id: int):
        dataset = RandomNumberDataset(dataset_texts, self.tokenizer)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        output_dir = os.path.join(self.config.output_dir, f"trial_{trial_id}")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_strategy="no", # 节省 NAS 过程中的磁盘空间
            remove_unused_columns=False,
            report_to="none",
            bf16=True, # 现代 LLM 训练标配
        )

        # 使用支持 LoRA+ 的自定义 Trainer
        trainer = LoRAPlusTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        return self.model