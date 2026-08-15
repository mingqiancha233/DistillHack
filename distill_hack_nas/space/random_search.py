import optuna
from typing import Callable
from .base import BaseSearch

class RandomSearch(BaseSearch):
    """
    Pure Random Search. 
    Essential as a baseline in any rigorous NAS academic paper to prove 
    that your search space and advanced algorithms are actually effective.
    """
    def __init__(self, n_trials: int, seed: int = 42):
        super().__init__(n_trials)
        self.sampler = optuna.samplers.RandomSampler(seed=seed)
        self.study = optuna.create_study(direction="maximize", sampler=self.sampler)

    def optimize(self, objective_fn: Callable[[optuna.Trial], float]) -> optuna.trial.FrozenTrial:
        print(f"Starting Random NAS for {self.n_trials} trials...")
        self.study.optimize(objective_fn, n_trials=self.n_trials)
        return self.study.best_trial
