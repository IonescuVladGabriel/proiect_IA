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

    def _init_population(self) -> np.ndarray:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return self.rng.uniform(lo, hi, size=(self.cfg.pop_size, self.D))

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return np.clip(x, lo, hi)

    def _evaluate_population(self, pop: np.ndarray) -> np.ndarray:
        fit = np.empty((pop.shape[0],), dtype=float)
        for i in range(pop.shape[0]):
            fit[i] = float(self.fitness_fn(pop[i]))
        return fit


    def run(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._init_population()
        fit = self._evaluate_population(pop)

        best_idx = int(np.argmin(fit))
        best_x = pop[best_idx].copy()
        best_f = float(fit[best_idx])

        history = {"best_fitness": [best_f], "best_x": [best_x.copy()]}

        for _g in range(self.cfg.generations):
            new_pop = pop.copy()
            new_fit = fit.copy()

            for i in range(self.cfg.pop_size):
                # choose r1,r2,r3 distinct and != i
                idxs = np.arange(self.cfg.pop_size)
                idxs = idxs[idxs != i]
                r1, r2, r3 = self.rng.choice(idxs, size=3, replace=False)

                x_r1 = pop[r1]
                x_r2 = pop[r2]
                x_r3 = pop[r3]
                x_i = pop[i]

                # Mutation: DE/rand/1
                v = x_r1 + self.cfg.F * (x_r2 - x_r3)

                # Crossover: binomial + forced j_rand
                j_rand = self.rng.integers(0, self.D)
                cross_mask = self.rng.random(self.D) < self.cfg.CR
                # Forteaza preluarea a cel putin unei componente din vectorul mutant (previne cazul in care u este identic cu x_i)
                cross_mask[j_rand] = True

                u = np.where(cross_mask, v, x_i)
                u = self._clip_to_bounds(u)

                # Selection (minimization)
                f_u = float(self.fitness_fn(u))
                if f_u <= fit[i]:
                    new_pop[i] = u
                    new_fit[i] = f_u

            pop, fit = new_pop, new_fit

            gen_best_idx = int(np.argmin(fit))
            gen_best_f = float(fit[gen_best_idx])
            if gen_best_f < best_f:
                best_f = gen_best_f
                best_x = pop[gen_best_idx].copy()

            history["best_fitness"].append(best_f)
            history["best_x"].append(best_x.copy())

        return best_x, best_f, history