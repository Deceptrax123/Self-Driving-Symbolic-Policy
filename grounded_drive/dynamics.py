"""System M: bicycle prior plus a small online neural acceleration residual.

This is a transparent, unpretrained dynamics baseline, not an implementation of
LEWM or a learned visual representation model. Its neural weights adapt only from
executed control/state transitions. Surprise is standardized prediction error,
not a calibrated likelihood or a reward to maximize while driving.
"""
import math
import numpy as np
from .types import Control


class OnlineDynamics:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.25, (16, 6))
        self.b1 = np.zeros(16)
        self.w2 = np.zeros(16)
        self.b2 = 0.0
        self.error_variance = 1.0
        self.updates = 0

    @staticmethod
    def features(speed, control):
        return np.array([speed / 15, control.throttle, control.brake,
                         control.steer, speed * control.throttle / 15,
                         speed * control.brake / 15], dtype=float)

    @staticmethod
    def prior_acceleration(speed, control):
        return 3.0 * control.throttle - 7.0 * control.brake - 0.12 * speed

    def residual(self, speed, control):
        hidden = np.tanh(self.w1 @ self.features(speed, control) + self.b1)
        return float(2.0 * np.tanh(self.w2 @ hidden + self.b2))

    def predict_speed(self, speed, control, dt):
        acceleration = self.prior_acceleration(speed, control) + self.residual(speed, control)
        return max(0.0, speed + acceleration * dt)

    def observe(self, speed, control, next_speed, dt):
        if not 0.001 <= dt <= 1.0:
            return 0.0
        # A clamped stationary vehicle does not reveal the braking acceleration.
        if speed < 0.2 and next_speed < 0.2 and control.brake > 0:
            return 0.0
        x = self.features(speed, control)
        h = np.tanh(self.w1 @ x + self.b1)
        out = float(np.tanh(self.w2 @ h + self.b2))
        predicted_acceleration = self.prior_acceleration(speed, control) + 2 * out
        measured_acceleration = float(np.clip((next_speed - speed) / dt, -15, 10))
        err = measured_acceleration - predicted_acceleration
        surprise = 0.5 * err * err / max(0.25, self.error_variance)
        self.error_variance = 0.98 * self.error_variance + 0.02 * min(err * err, 16)
        # Bounded online SGD trains both layers; clipping limits sensor outliers.
        grad = float(np.clip(err, -4, 4)) * 2 * (1 - out * out)
        hidden_grad = grad * self.w2.copy() * (1 - h * h)
        self.w2 += 0.015 * grad * h
        self.b2 += 0.015 * grad
        self.w1 += 0.005 * np.outer(hidden_grad, x)
        self.b1 += 0.005 * hidden_grad
        self.updates += 1
        return float(surprise)

    def rollout(self, speed, control, horizon=2.0, dt=0.1):
        x = y = yaw = 0.0
        trajectory = []
        for _ in range(max(0, int(horizon / dt))):
            new_speed = self.predict_speed(speed, control, dt)
            v = 0.5 * (speed + new_speed)
            yaw += v * math.tan(control.steer * 0.55) / 2.8 * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            speed = new_speed
            trajectory.append((x, y, yaw, speed))
        return trajectory
