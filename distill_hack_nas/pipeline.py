import optuna
from typing import Dict, Any
from .config import DistillConfig, TrainConfig, NASConfig
from .api.base import BaseAPIClient
from .data.generator import BiasedDataGenerator
from .space.base import SearchSpace
from .train.builder import ModelBuilder
from .train.trainer import DistillTrainer
from .eval.evaluator import RPEvaluator
from .search.bayesian import BayesianSearch

class DistillHackNASPipeline:
    def __init__(
        self, 
        api_client: BaseAPIClient,
        search_space: SearchSpace,
        distill_config: DistillConfig,
        train_config: TrainConfig,
        nas_config: NASConfig
    ):
        self.api_client = api_client
        self.search_space = search_space
        self.distill_config = distill_config
        self.train_config = train_config
        self.nas_config = nas_config
        
        # Pre-generate the biased dataset once
        generator = BiasedDataGenerator(api_client, distill_config)
        self.dataset = generator.generate_dataset()
        
        self.evaluator = RPEvaluator(api_client, distill_config.role_play_prompt)

    def _objective(self, trial: optuna.Trial) -> float:
        # 1. Sample architecture
        sampled_arch = self.search_space.sample(trial)
        print(f"\n--- Trial {trial.number} ---")
        print(f"Sampled Arch: {sampled_arch}")

        # 2. Build model
        builder = ModelBuilder(self.train_config)
        model, tokenizer = builder.build(sampled_arch)

        # 3. Train (Distill implicit bias)
        trainer = DistillTrainer(model, tokenizer, self.train_config)
        trained_model = trainer.train(self.dataset, trial.number)

        # 4. Evaluate Proxy Score (RP Awakening)
        score = self.evaluator.evaluate(trained_model, tokenizer)
        print(f"Trial {trial.number} Score: {score}")
        
        # Clear memory
        del trained_model
        del model
        import torch
        torch.cuda.empty_cache()

        return score

    def run(self) -> Dict[str, Any]:
        searcher = BayesianSearch(n_trials=self.nas_config.n_trials)
        best_trial = searcher.optimize(self._objective)
        
        print("\n=== NAS Completed ===")
        print(f"Best Score: {best_trial.value}")
        print(f"Best Architecture: {best_trial.params}")
        return best_trial.params
