from dataclasses import dataclass

@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    u_min: float = -10.0
    u_max: float = 10.0
    integral_limit: float = 10.0

    def post_init(self):
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def step(self, error: float, dt: float) -> float:
        if not self.initialized:
            self.prev_error = error
            self.initialized = True

        self.integral += error * dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        u = self.kp * error + self.ki * self.integral + self.kd * derivative

        if u > self.u_max:
            u = self.u_max
        elif u < self.u_min:
            u = self.u_min

        self.prev_error = error
        return u