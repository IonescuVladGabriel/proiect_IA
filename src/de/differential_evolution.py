from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any
import numpy as np


@dataclass(frozen=True)
class DEConfig:
    pop_size: int = 30
    F: float = 0.8
    CR: float = 0.9
    generations: int = 200
    seed: int = 0


class DifferentialEvolution:
    """
    Evolutie diferentiala - varianta standard DE/rand/1/bin.

    Mutatie:
        v_i = x_r1 + F * (x_r2 - x_r3)

    Incrucisare binomiala (cu index fortat j_rand):
        u[i,j] = v[i,j] daca rand < CR sau j == j_rand, altfel u[i,j] = x[i,j]

    Selectie (minimizare):
        x_i devine u_i daca fitness(u_i) <= fitness(x_i), altfel se pastreaza x_i
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        fitness_fn: Callable[[np.ndarray], float],
        config: DEConfig,
    ):
        self.bounds = np.array(bounds, dtype=float)
        self.fitness_fn = fitness_fn
        self.cfg = config

        if not (0.0 < self.cfg.F <= 2.0):
            raise ValueError("F must be in (0, 2].")
        if not (0.0 <= self.cfg.CR <= 1.0):
            raise ValueError("CR must be in [0, 1].")
        if self.cfg.pop_size < 4:
            raise ValueError("pop_size must be >= 4 for DE/rand/1.")

        self.rng = np.random.default_rng(self.cfg.seed)
        self.D = self.bounds.shape[0]

    def run(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        raise NotImplementedError