"""Privileged CARLA-state grounding for an expert policy (not sensor perception)."""
from __future__ import annotations
import math
from typing import Any
from .types import Actor, Observation


def local_xy(transform: Any, location: Any) -> tuple[float, float]:
    yaw = math.radians(transform.rotation.yaw)
    dx, dy = location.x - transform.location.x, location.y - transform.location.y
    return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)


class CarlaAdapter:
    def __init__(self, world: Any, ego: Any, route: list[Any], dt: float = .05):
        if len(route) < 2:
            raise ValueError('A route requires at least two waypoints')
        self.world, self.ego, self.route, self.dt = world, ego, route, dt
        self.map = world.get_map()
        self.cursor = 0
        self.total_length = sum(a.transform.location.distance(b.transform.location) for a,b in zip(route, route[1:]))
        self.remaining = self.total_length
        self.progress = 0.0
        self.cumulative = [0.0]
        for a,b in zip(route,route[1:]):
            self.cumulative.append(self.cumulative[-1] + a.transform.location.distance(b.transform.location))
        self.lateral_error = 0.0

    def observe(self, frame: int) -> Observation:
        transform = self.ego.get_transform()
        location = transform.location
        # Project onto route segments: nearest-waypoint distance produces a sawtooth
        # remaining distance, which can spuriously reaccelerate near the destination.
        best = None
        for i in range(max(0,self.cursor-1), min(len(self.route)-1,self.cursor+25)):
            a = self.route[i].transform.location
            b = self.route[i+1].transform.location
            dx,dy = b.x-a.x,b.y-a.y
            length_sq = dx*dx+dy*dy
            u = max(0.,min(1.,((location.x-a.x)*dx+(location.y-a.y)*dy)/max(1e-8,length_sq)))
            px,py = a.x+u*dx,a.y+u*dy
            error = math.hypot(location.x-px,location.y-py)
            along = self.cumulative[i]+u*(self.cumulative[i+1]-self.cumulative[i])
            if best is None or error<best[0]:
                lateral=abs(((location.x-px)*-dy+(location.y-py)*dx)/max(1e-8,math.sqrt(length_sq)))
                best=(error,i,along,lateral)
        _,self.cursor,along,self.lateral_error=best
        self.progress=max(self.progress,along)
        self.remaining=max(0.,self.total_length-self.progress)
        nearest=self.route[self.cursor]
        future=self.route[self.cursor+1:]
        points = tuple(local_xy(transform, w.transform.location) for w in future[:40])
        points = tuple(p for p in points if p[0] > .3)
        yaw = math.radians(transform.rotation.yaw)
        actors = []
        for actor in self.world.get_actors():
            if actor.id == self.ego.id or not (actor.type_id.startswith(('vehicle.', 'walker.pedestrian.', 'static.prop.'))):
                continue
            if actor.get_location().distance(location) > 85:
                continue
            other_transform = actor.get_transform()
            bbox = actor.bounding_box
            center = other_transform.transform(bbox.location)
            x,y = local_xy(transform, center)
            v = actor.get_velocity()
            # Project oriented extents into ego frame; conservative AABB footprint.
            dyaw = math.radians(other_transform.rotation.yaw-transform.rotation.yaw)
            length = 2*(abs(math.cos(dyaw))*bbox.extent.x + abs(math.sin(dyaw))*bbox.extent.y)
            width = 2*(abs(math.sin(dyaw))*bbox.extent.x + abs(math.cos(dyaw))*bbox.extent.y)
            kind = 'pedestrian' if actor.type_id.startswith('walker.') else ('vehicle' if actor.type_id.startswith('vehicle.') else 'obstacle')
            actors.append(Actor(str(actor.id), kind, x, y, v.x*math.cos(yaw)+v.y*math.sin(yaw), -v.x*math.sin(yaw)+v.y*math.cos(yaw), length, width))
        light, stop_distance = 'none', None
        route_keys = {(w.road_id,w.lane_id) for w in future[:40]}
        for signal in self.world.get_actors().filter('traffic.traffic_light*'):
            for stop in signal.get_stop_waypoints():
                if (stop.road_id,stop.lane_id) not in route_keys:
                    continue
                x,y = local_xy(transform, stop.transform.location)
                if x < -1 or x > 70 or abs(y) > nearest.lane_width*1.5:
                    continue
                if stop_distance is None or x < stop_distance:
                    light, stop_distance = str(signal.get_state()).split('.')[-1].lower(), max(0., x - self.ego.bounding_box.extent.x)
        velocity = self.ego.get_velocity()
        return Observation(speed=math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2), dt=self.dt, route=points, actors=tuple(actors), speed_limit=min(self.ego.get_speed_limit()/3.6, 9.), lane_width=nearest.lane_width, traffic_light=light, stop_line_distance=stop_distance, distance_to_goal=self.remaining, frame=frame)

    def apply(self, control: Any) -> None:
        import carla
        self.ego.apply_control(carla.VehicleControl(throttle=float(control.throttle), brake=float(control.brake), steer=float(control.steer)))
