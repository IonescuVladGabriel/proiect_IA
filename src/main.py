import numpy as np
from src.de.differential_evolution import DifferentialEvolution, DEConfig


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def main():
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    cfg = DEConfig(pop_size=20, F=0.8, CR=0.9, generations=60, seed=42)
    de = DifferentialEvolution(bounds=bounds, fitness_fn=sphere, config=cfg)
    best_x, best_f, _ = de.run()
    print("BEST:", best_x, best_f)


if __name__ == "__main__":
    main()