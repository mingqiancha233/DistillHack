import optuna
from typing import Callable
from .base import BaseSearch

class EvolutionarySearch(BaseSearch):
    """
    Evolutionary Algorithm based on NSGA-II (Non-dominated Sorting Genetic Algorithm II).
    Highly effective for complex, discrete architectural search spaces (like MoE routing).
    """
    def __init__(self, n_trials: int, population_size: int = 50, mutation_prob: float = 0.1):
        super().__init__(n_trials)
        # NSGA-II is the standard genetic algorithm in Optuna
        self.sampler = optuna.samplers.NSGAIISampler(
            population_size=population_size,
            mutation_prob=mutation_prob,
            seed=42
        )
        self.study = optuna.create_study(direction="maximize", sampler=self.sampler)

    def optimize(self, objective_fn: Callable[[optuna.Trial], float]) -> optuna.trial.FrozenTrial:
        print(f"Starting Evolutionary NAS (NSGA-II) for {self.n_trials} trials...")
        self.study.optimize(objective_fn, n_trials=self.n_trials)
        return self.study.best_trial
