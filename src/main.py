from src.de.differential_evolution import DifferentialEvolution, DEConfig
from src.control.fitness import make_pid_fitness, default_pid_bounds
from src.viz.plots import plot_response, plot_fitness_curve


def main():
    bounds = default_pid_bounds()

    fitness_fn, simulator = make_pid_fitness(
        setpoint=1.0,
        dt=0.01,
        horizon_s=8.0,
        wn=2.0,
        zeta=0.25,
        u_min=-10.0,
        u_max=10.0,
        disturbance_time=4.0,
        disturbance_value=0.2,
    )

    cfg = DEConfig(pop_size=35, F=0.8, CR=0.9, generations=200, seed=42)
    de = DifferentialEvolution(bounds=bounds, fitness_fn=fitness_fn, config=cfg)

    best_x, best_f, history = de.run()
    kp, ki, kd = best_x

    print("\n=== BEST PID FOUND ===")
    print(f"Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")
    print(f"Best fitness = {best_f:.6f}")

    sim = simulator(best_x)
    plot_response(sim["t"], sim["r"], sim["y"], sim["u"], title="Best PID step response (DE/rand/1/bin)")
    plot_fitness_curve(history["best_fitness"], title="Best fitness per generation")


if __name__ == "__main__":
    main()