import optuna
from typing import Callable
from .base import BaseSearch

class CMAESSearch(BaseSearch):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    State-of-the-art for continuous hyperparameter spaces (e.g., learning rates, 
    dropout rates, routing auxiliary loss coefficients).
    Note: Requires `cmaes` package (`pip install cmaes`).
    """
    def __init__(self, n_trials: int, restart_strategy: str = "ipop"):
        super().__init__(n_trials)
        self.sampler = optuna.samplers.CmaEsSampler(
            restart_strategy=restart_strategy, # 'ipop' increases population size on restart
            seed=42
        )
        self.study = optuna.create_study(direction="maximize", sampler=self.sampler)

    def optimize(self, objective_fn: Callable[[optuna.Trial], float]) -> optuna.trial.FrozenTrial:
        print(f"Starting CMA-ES NAS for {self.n_trials} trials...")
        self.study.optimize(objective_fn, n_trials=self.n_trials)
        return self.study.best_trial
