"""Executable System 1 / System 2 / System M expert-policy prototype."""
import math
from .dynamics import OnlineDynamics
from .grounding import SymbolicGrounder
from .planner import SymbolicPlanner
from .types import Control, Decision, Observation


class GroundedDrivingPolicy:
    def __init__(self):
        self.grounder = SymbolicGrounder()
        self.planner = SymbolicPlanner()
        self.dynamics = OnlineDynamics()
        self.previous = None
        self.last_plan = None

    def reset(self, keep_dynamics=True):
        self.previous = None
        self.last_plan = None
        if not keep_dynamics:
            self.dynamics = OnlineDynamics()

    @staticmethod
    def control(obs, plan):
        lookahead = max(4.0, min(16.0, 3.0 + 0.65 * obs.speed))
        target = next(((x, y) for x, y in obs.route if math.hypot(x, y) >= lookahead),
                      obs.route[-1] if obs.route else (1.0, 0.0))
        x, y = target
        curvature = 2 * y / max(1.0, x*x + y*y)
        steer = max(-1.0, min(1.0, math.atan(2.8 * curvature) / 0.55))
        error = plan.target_speed - obs.speed
        if plan.emergency:
            return Control(0.0, 1.0, steer)
        if plan.target_speed < 0.15 and obs.speed < 0.3:
            return Control(0.0, 0.65, steer)
        # Compensate drag at equilibrium to avoid steady-state speed error.
        demand = 0.8 * error + 0.12 * obs.speed
        if error < -0.3:
            return Control(0.0, min(1.0, max(0, -error * 0.25)), steer)
        return Control(min(0.8, max(0.0, demand / 3.0)), 0.0, steer)

    def step(self, obs: Observation):
        values = [obs.speed, obs.dt, obs.speed_limit, obs.lane_width]
        values += [v for point in obs.route for v in point]
        values += [v for a in obs.actors for v in (a.x, a.y, a.vx, a.vy, a.length, a.width)]
        values += [v for v in (obs.stop_line_distance, obs.distance_to_goal) if v is not None]
        if not all(math.isfinite(v) for v in values) or obs.speed < 0 or obs.dt <= 0:
            raise ValueError('Observation must contain finite values, nonnegative speed, and positive dt')
        surprise = 0.0
        if self.previous is not None:
            speed, control = self.previous
            surprise = self.dynamics.observe(speed, control, obs.speed, obs.dt)
        grounding = self.grounder.ground(obs)
        plan = self.planner.plan(obs, grounding, surprise)
        signature = (plan.subgoal, tuple(f.predicate for f in plan.facts), plan.emergency)
        replanned = signature != self.last_plan or surprise > 4.5
        control = self.control(obs, plan)
        prediction = {'next_speed': self.dynamics.predict_speed(obs.speed, control, obs.dt),
                      'acceleration_residual': self.dynamics.residual(obs.speed, control),
                      'error_variance': self.dynamics.error_variance}
        self.previous = (obs.speed, control)
        self.last_plan = signature
        return Decision(control, plan, surprise, replanned, self.dynamics.updates, prediction)
