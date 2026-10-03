"""Regression coverage for CARLA grounding using deterministic fake geometry."""
import fnmatch
import math
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from grounded_drive.carla_adapter import CarlaAdapter


@dataclass
class Vector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance(self, other):
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2)


class Transform:
    def __init__(self, location, yaw=0):
        self.location = location
        self.rotation = SimpleNamespace(yaw=yaw)

    def transform(self, point):
        angle = math.radians(self.rotation.yaw)
        return Vector(self.location.x + point.x*math.cos(angle)-point.y*math.sin(angle),
                      self.location.y + point.x*math.sin(angle)+point.y*math.cos(angle),
                      self.location.z + point.z)


class FakeActor:
    def __init__(self, actor_id, transform, velocity=None, type_id='vehicle.fake', bbox_offset=None):
        self.id = actor_id
        self.type_id = type_id
        self.pose = transform
        self.velocity = velocity or Vector()
        self.bounding_box = SimpleNamespace(location=bbox_offset or Vector(), extent=Vector(2.3, 1, .8))

    def get_transform(self):
        return self.pose

    def get_location(self):
        return self.pose.location

    def get_velocity(self):
        return self.velocity

    def get_speed_limit(self):
        return 30.0


class ActorList(list):
    def filter(self, pattern):
        return ActorList(a for a in self if fnmatch.fnmatch(a.type_id, pattern))


class FakeWorld:
    def __init__(self, actors):
        self.actors = ActorList(actors)

    def get_map(self):
        return SimpleNamespace()

    def get_actors(self):
        return self.actors


def waypoint(location, yaw=0):
    return SimpleNamespace(transform=Transform(location, yaw), lane_width=3.5, road_id=1, lane_id=-1)


class CarlaAdapterTests(unittest.TestCase):
    def test_remaining_distance_is_continuous_across_waypoint_midpoints(self):
        ego = FakeActor(1, Transform(Vector()))
        route = [waypoint(Vector(x, 0)) for x in range(0, 12, 2)]
        adapter = CarlaAdapter(FakeWorld([ego]), ego, route)
        previous = 10.0
        # The old nearest-waypoint method alternately decreased/increased here.
        # Nonzero lateral error must not inflate the along-route distance either.
        for frame, x in enumerate((0, .25, .75, 1.0, 1.25, 1.75, 2.25, 2.75, 3.25, 5.75, 8.25, 9.25, 9.75, 10.0)):
            ego.pose.location = Vector(x, .4)
            observation = adapter.observe(frame)
            self.assertAlmostEqual(observation.distance_to_goal, 10.0-x, places=8)
            self.assertAlmostEqual(adapter.progress, x, places=8)
            self.assertAlmostEqual(adapter.lateral_error, .4, places=8)
            self.assertLessEqual(observation.distance_to_goal, previous)
            previous = observation.distance_to_goal

    @staticmethod
    def scene(yaw, origin):
        frame = Transform(origin, yaw)
        angle = math.radians(yaw)
        def velocity(x, y):
            return Vector(x*math.cos(angle)-y*math.sin(angle), x*math.sin(angle)+y*math.cos(angle))
        ego = FakeActor(1, Transform(frame.transform(Vector()), yaw), velocity(10, 0))
        # A rotated box offset also needs transformation before ego grounding.
        other = FakeActor(2, Transform(frame.transform(Vector(12, 3)), yaw+90),
                          velocity(3, 1), bbox_offset=Vector(.5, 0))
        route = [waypoint(frame.transform(Vector(x, 0)), yaw) for x in range(0, 42, 2)]
        signal_stop = waypoint(frame.transform(Vector(30, 0)), yaw)
        signal = SimpleNamespace(id=3, type_id='traffic.traffic_light',
                                 get_stop_waypoints=lambda: [signal_stop], get_state=lambda: 'Red')
        return CarlaAdapter(FakeWorld([ego, other, signal]), ego, route).observe(7)

    def test_translated_rotated_world_has_same_ego_relative_observation(self):
        reference = self.scene(0, Vector())
        for yaw, origin in [(90, Vector(1234, -87)), (-137, Vector(-980, 2000))]:
            rotated = self.scene(yaw, origin)
            self.assertAlmostEqual(rotated.speed, reference.speed)
            self.assertAlmostEqual(rotated.distance_to_goal, reference.distance_to_goal)
            self.assertEqual(rotated.traffic_light, 'red')
            self.assertAlmostEqual(rotated.stop_line_distance, 27.7)
            self.assertEqual(len(rotated.route), len(reference.route))
            for expected, actual in zip(reference.route, rotated.route):
                for e, a in zip(expected, actual):
                    self.assertAlmostEqual(a, e, places=8)
            actor = rotated.actors[0]
            self.assertAlmostEqual(actor.x, 12)
            self.assertAlmostEqual(actor.y, 3.5)
            # Actor world velocity projected onto ego axes, NOT relative velocity.
            self.assertAlmostEqual(actor.vx, 3)
            self.assertAlmostEqual(actor.vy, 1)
            self.assertAlmostEqual(actor.length, 2)
            self.assertAlmostEqual(actor.width, 4.6)


if __name__ == '__main__':
    unittest.main()
