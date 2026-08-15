import optuna
from abc import ABC, abstractmethod
from typing import Callable

class BaseSearch(ABC):
    """
    Abstract base class for Neural Architecture Search algorithms.
    """
    def __init__(self, n_trials: int):
        self.n_trials = n_trials

    @abstractmethod
    def optimize(self, objective_fn: Callable[[optuna.Trial], float]) -> optuna.trial.FrozenTrial:
        """
        Run the search algorithm to optimize the objective function.
        
        Args:
            objective_fn: A callable that takes an optuna.Trial and returns a float score.
            
        Returns:
            The best trial found during the search.
        """
        pass
