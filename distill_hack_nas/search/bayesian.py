import optuna
from typing import Callable

class BayesianSearch:
    def __init__(self, n_trials: int):
        self.n_trials = n_trials
        self.sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="maximize", sampler=self.sampler)

    def optimize(self, objective_fn: Callable[[optuna.Trial], float]):
        print(f"Starting Bayesian NAS for {self.n_trials} trials...")
        self.study.optimize(objective_fn, n_trials=self.n_trials)
        return self.study.best_trial