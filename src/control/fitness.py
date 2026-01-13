from typing import Callable, Tuple, List, Dict, Any
import numpy as np

from .pid import PID
from .plant import SecondOrderPlant


def default_pid_bounds() -> List[tuple]:
    return [(0.0, 50.0), (0.0, 20.0), (0.0, 10.0)]


def make_pid_fitness(
    setpoint: float = 1.0,
    dt: float = 0.01,
    horizon_s: float = 8.0,
    wn: float = 2.0,
    zeta: float = 0.25,
    u_min: float = -10.0,
    u_max: float = 10.0,
    disturbance_time: float = 4.0,
    disturbance_value: float = 0.2,
) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], Dict[str, Any]]]:

    n_steps = int(horizon_s / dt)
    t = np.arange(n_steps) * dt

    def simulate(x: np.ndarray) -> Dict[str, Any]:
        kp, ki, kd = float(x[0]), float(x[1]), float(x[2])

        pid = PID(kp, ki, kd, u_min=u_min, u_max=u_max, integral_limit=20.0)
        plant = SecondOrderPlant(wn=wn, zeta=zeta)

        y = np.zeros(n_steps, dtype=float)
        u = np.zeros(n_steps, dtype=float)
        r = np.full(n_steps, setpoint, dtype=float)

        for k in range(n_steps):
            e = r[k] - plant.y
            u_k = pid.step(e, dt)

            disturbance = 0.0
            if abs(t[k] - disturbance_time) < dt:
                disturbance = disturbance_value

            y_k = plant.step(u_k, dt, disturbance=disturbance)

            y[k] = y_k
            u[k] = u_k

        return {"t": t, "y": y, "u": u, "r": r}

    def fitness_fn(x: np.ndarray) -> float:
        sim = simulate(x)
        y = sim["y"]
        u = sim["u"]
        r = sim["r"]
        e = r - y

        iae = float(np.sum(np.abs(e)) * dt)

        overshoot = float(np.max(y - setpoint))
        overshoot_pen = overshoot if overshoot > 0.0 else 0.0

        effort = float(np.sum(np.abs(u)) * dt)

        du = np.diff(u, prepend=u[0])
        smooth = float(np.sum(np.abs(du)) * dt)

        fitness = iae + 5.0 * overshoot_pen + 0.05 * effort + 0.2 * smooth

        if not np.isfinite(fitness) or np.max(np.abs(y)) > 1e3:
            return 1e9

        return float(fitness)

    return fitness_fn, simulate