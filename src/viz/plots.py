import matplotlib.pyplot as plt
import numpy as np


def plot_response(t: np.ndarray, r: np.ndarray, y: np.ndarray, u: np.ndarray, title: str):
    plt.figure()
    plt.plot(t, r, label="setpoint r(t)")
    plt.plot(t, y, label="output y(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Output")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(t, u, label="control u(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Control")
    plt.title("Control signal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_fitness_curve(best_fitness, title: str):
    plt.figure()
    plt.plot(best_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()