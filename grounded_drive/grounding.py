"""Ground transient geometric observations into reusable relational predicates."""
import math
from .types import Fact, Grounding, Observation


def route_projection(x, y, route):
    """Return along-path distance and signed lateral offset to the nearest segment."""
    points = ((0.0, 0.0),) + tuple(route)
    best = (float('inf'), 0.0, 0.0)
    along = 0.0
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        u = min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / length**2))
        px, py = ax + u * dx, ay + u * dy
        distance = math.hypot(x - px, y - py)
        if distance < best[0]:
            signed = ((x - px) * -dy + (y - py) * dx) / length
            best = (distance, along + u * length, signed)
        along += length
    return best[1], best[2], best[0]


class SymbolicGrounder:
    def ground(self, obs: Observation):
        facts = []
        route_valid = bool(obs.route) and any(math.hypot(x, y) > 1 for x, y in obs.route)
        if not route_valid:
            facts.append(Fact('route_missing'))
        # Radius from pure-pursuit curvature bounds lateral acceleration to 2 m/s².
        curve_speed = obs.speed_limit
        for x, y in obs.route:
            radius_sq = x*x + y*y
            if 5 < math.sqrt(radius_sq) < max(12, obs.speed * 2.0) and radius_sq > 0:
                curvature = abs(2 * y / radius_sq)
                if curvature > 0.015:
                    curve_speed = min(curve_speed, math.sqrt(2.0 / curvature))
        if curve_speed < obs.speed_limit:
            facts.append(Fact('curve_ahead', speed=curve_speed))
        stopping_range = obs.speed**2 / 8.0 + 1.2 * obs.speed + 8.0
        for actor in obs.actors:
            if actor.x < -max(3.0, actor.length / 2):
                continue
            along, lateral, route_distance = route_projection(actor.x, actor.y, obs.route)
            half_corridor = 1.0 + actor.width / 2 + 0.35
            # Exact route distance also rejects actors beyond route endpoints.
            overlaps_ego = abs(actor.x) < 2.3 + actor.length / 2 and abs(actor.y) < 1.0 + actor.width / 2
            on_path = route_valid and ((route_distance < half_corridor and along > 0.05) or overlaps_ego)
            clearance = max(0.0, along - actor.length / 2 - 2.3)
            closing = max(0, obs.speed - actor.vx)
            ttc = clearance / closing if closing > 0.1 else None
            if on_path:
                pred = 'pedestrian_in_path' if actor.kind == 'pedestrian' else 'vehicle_ahead' if actor.kind == 'vehicle' else 'obstacle_ahead'
                facts.append(Fact(pred, actor.actor_id, clearance, ttc, max(0, actor.vx)))
            # Predict whether an off-path actor will cross the route during approach.
            # This uses both occupancy and time-to-conflict, not actor identity.
            if not on_path and route_valid:
                for i in range(1, 41):
                    t = i * 0.1
                    ax, ay = actor.x + actor.vx * t, actor.y + actor.vy * t
                    s, _, distance = route_projection(ax, ay, obs.route)
                    occupied = distance < half_corridor
                    ego_arrival = s / max(obs.speed, 1.0)
                    crossing_motion = abs(actor.vy) > 0.2 or abs(actor.vx) > 0.2
                    if occupied and crossing_motion and s > 0 and (abs(ego_arrival - t) < 1.6 or s < stopping_range):
                        gap = max(0, s - actor.length / 2 - 2.3)
                        facts.append(Fact('crossing_conflict', actor.actor_id, gap, t, max(0, actor.vx)))
                        break
        if obs.traffic_light in ('red', 'yellow') and obs.stop_line_distance is not None and obs.stop_line_distance >= -1:
            can_stop_for_yellow = obs.stop_line_distance >= obs.speed**2 / 8.0 + 0.3 * obs.speed + 2.5
            if obs.traffic_light == 'red' or can_stop_for_yellow:
                facts.append(Fact('stop_signal', distance=max(0, obs.stop_line_distance)))
        if obs.distance_to_goal is not None:
            facts.append(Fact('goal_distance', distance=max(0, obs.distance_to_goal)))
        return Grounding(tuple(facts), max(0, curve_speed), route_valid)
