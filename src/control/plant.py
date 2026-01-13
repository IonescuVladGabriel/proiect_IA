from dataclasses import dataclass


@dataclass
class SecondOrderPlant:
    wn: float = 2.0
    zeta: float = 0.25

    def post_init(self):
        self.reset()

    def reset(self):
        self.y = 0.0
        self.ydot = 0.0

    def step(self, u: float, dt: float, disturbance: float = 0.0) -> float:
        y = self.y
        ydot = self.ydot

        yddot = -2.0 * self.zeta * self.wn * ydot - (self.wn  2) * y + (self.wn  2) * u
        yddot += disturbance

        ydot_next = ydot + yddot * dt
        y_next = y + ydot_next * dt

        self.y = y_next
        self.ydot = ydot_next
        return self.y