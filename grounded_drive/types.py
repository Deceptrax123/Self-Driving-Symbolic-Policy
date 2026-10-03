"""Small simulator-independent contract. Distances are m, times s, speeds m/s.

The local coordinate system is x forward, y right; positive steering turns right.
Actor velocities are world velocities projected onto the ego axes, NOT relative
velocities. Grounding never uses world coordinates or persistent actor identities.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Actor:
    actor_id: str
    kind: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    length: float = 4.5
    width: float = 1.8


@dataclass(frozen=True)
class Observation:
    speed: float
    route: tuple[tuple[float, float], ...]
    actors: tuple[Actor, ...] = ()
    dt: float = 0.05
    speed_limit: float = 13.9
    lane_width: float = 3.5
    traffic_light: str = 'none'
    stop_line_distance: Optional[float] = None
    distance_to_goal: Optional[float] = None
    frame: int = 0


@dataclass(frozen=True)
class Control:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0


@dataclass(frozen=True)
class Fact:
    predicate: str
    actor_id: Optional[str] = None
    distance: Optional[float] = None
    ttc: Optional[float] = None
    speed: Optional[float] = None


@dataclass(frozen=True)
class Grounding:
    facts: tuple[Fact, ...]
    curve_speed: float
    route_valid: bool


@dataclass(frozen=True)
class Plan:
    subgoal: str
    target_speed: float
    reason: str
    steps: tuple[str, ...]
    facts: tuple[Fact, ...] = ()
    emergency: bool = False


@dataclass(frozen=True)
class Decision:
    control: Control
    plan: Plan
    surprisal: float
    replanned: bool
    model_updates: int
    prediction: dict = field(default_factory=dict)
