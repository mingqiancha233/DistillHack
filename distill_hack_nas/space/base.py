from abc import ABC, abstractmethod
from typing import Dict, Any
import optuna

class SearchSpace(ABC):
    @abstractmethod
    def sample(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample an architecture or hyperparameter config from the space."""
        pass
