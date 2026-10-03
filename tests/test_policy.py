import math
import unittest
from dataclasses import replace

from grounded_drive import Actor, Control, GroundedDrivingPolicy, Observation
from grounded_drive.dynamics import OnlineDynamics

ROUTE = tuple((float(x), 0.0) for x in range(2, 102, 2))


def obs(**kwargs):
    return Observation(speed=kwargs.pop('speed', 8.0), route=ROUTE, **kwargs)


class GroundedPolicyTests(unittest.TestCase):
    def test_identity_and_frame_do_not_change_action(self):
        a = Actor('training_vehicle', 'vehicle', 18, 0)
        b = replace(a, actor_id='unseen_vehicle_982352')
        d1 = GroundedDrivingPolicy().step(obs(actors=(a,), frame=20))
        d2 = GroundedDrivingPolicy().step(obs(actors=(b,), frame=123456))
        self.assertEqual(d1.control, d2.control)
        self.assertEqual(d1.plan.target_speed, d2.plan.target_speed)
        self.assertEqual(d1.plan.subgoal, 'follow')

    def test_crossing_pedestrian_grounded_before_entering_lane(self):
        actor = Actor('new_walker', 'pedestrian', 16, 4.0, 0, -1.8, .6, .6)
        decision = GroundedDrivingPolicy().step(obs(actors=(actor,)))
        self.assertEqual(decision.plan.subgoal, 'yield')
        self.assertIn('crossing_conflict', [f.predicate for f in decision.plan.facts])
        away = replace(actor, vy=1.8)
        self.assertEqual(GroundedDrivingPolicy().step(obs(actors=(away,))).plan.subgoal, 'cruise')

    def test_obstacle_clears_and_subgoal_revises(self):
        policy = GroundedDrivingPolicy()
        block = Actor('block', 'obstacle', 10, 0, length=1, width=1)
        stopped = policy.step(obs(actors=(block,)))
        resumed = policy.step(obs())
        self.assertEqual(stopped.plan.subgoal, 'stop_for_obstacle')
        self.assertEqual(resumed.plan.subgoal, 'cruise')
        self.assertTrue(resumed.replanned)
        self.assertGreater(resumed.control.throttle, 0)

    def test_red_stop_green_resume_and_yellow_dilemma(self):
        policy = GroundedDrivingPolicy()
        red = policy.step(obs(speed=0, traffic_light='red', stop_line_distance=2))
        self.assertEqual(red.plan.subgoal, 'stop_for_signal')
        self.assertGreater(red.control.brake, 0)
        green = policy.step(obs(speed=0, traffic_light='green', stop_line_distance=2))
        self.assertGreater(green.control.throttle, 0)
        close_yellow = GroundedDrivingPolicy().step(obs(traffic_light='yellow', stop_line_distance=5))
        self.assertEqual(close_yellow.plan.subgoal, 'cruise')
        far_yellow = GroundedDrivingPolicy().step(obs(traffic_light='yellow', stop_line_distance=18))
        self.assertEqual(far_yellow.plan.subgoal, 'stop_for_signal')

    def test_overlap_and_imminent_collision_override_throttle(self):
        for x in [0, 4, 8]:
            decision = GroundedDrivingPolicy().step(obs(actors=(Actor('v', 'vehicle', x, 0),)))
            self.assertTrue(decision.plan.emergency)
            self.assertEqual(decision.control.brake, 1.0)
            self.assertEqual(decision.control.throttle, 0)

    def test_adjacent_or_behind_actor_does_not_block(self):
        for x, y in [(15, 5), (-10, 0)]:
            decision = GroundedDrivingPolicy().step(obs(actors=(Actor('v', 'vehicle', x, y),)))
            self.assertEqual(decision.plan.subgoal, 'cruise')

    def test_missing_route_holds_and_bad_observations_rejected(self):
        policy = GroundedDrivingPolicy()
        d = policy.step(Observation(speed=4, route=()))
        self.assertEqual(d.control.brake, 1)
        with self.assertRaises(ValueError):
            policy.step(obs(speed=float('nan')))

    def test_terminal_route_exhaustion_is_goal_hold_not_route_failure(self):
        reached = GroundedDrivingPolicy().step(Observation(speed=1.5, route=((.8, 0),), distance_to_goal=.8))
        self.assertEqual(reached.plan.subgoal, 'hold_at_goal')
        self.assertEqual(reached.control.brake, 1.0)
        missing = GroundedDrivingPolicy().step(Observation(speed=1.5, route=(), distance_to_goal=20))
        self.assertEqual(missing.plan.subgoal, 'recover_route')

    def test_goal_and_curved_route_ground_controls(self):
        stopped = GroundedDrivingPolicy().step(obs(speed=0, distance_to_goal=1.5))
        self.assertEqual(stopped.plan.subgoal, 'reach_goal')
        self.assertGreater(stopped.control.brake, 0)
        curved = replace(obs(), route=((2, .1), (5, 2), (10, 7), (15, 15)))
        d = GroundedDrivingPolicy().step(curved)
        self.assertGreater(d.control.steer, 0)
        self.assertLess(d.plan.target_speed, curved.speed_limit)

    def test_closed_loop_stops_before_stationary_vehicle(self):
        policy = GroundedDrivingPolicy()
        speed, travelled, obstacle_x = 10.0, 0.0, 38.0
        for frame in range(400):
            decision = policy.step(obs(speed=speed, actors=(Actor('lead', 'vehicle', obstacle_x-travelled, 0),), frame=frame))
            c = decision.control
            acceleration = 3*c.throttle - 7*c.brake - .12*speed
            next_speed = max(0, speed + acceleration * .05)
            travelled += .5 * (speed + next_speed) * .05
            speed = next_speed
        self.assertLess(travelled + 4.55, obstacle_x)
        self.assertLess(speed, .3)

    def test_stop_signal_closed_loop_no_line_crossing(self):
        policy = GroundedDrivingPolicy()
        speed, distance = 10.0, 40.0
        for _ in range(400):
            decision = policy.step(obs(speed=speed, traffic_light='red', stop_line_distance=distance))
            c = decision.control
            next_speed = max(0, speed + (3*c.throttle - 7*c.brake - .12*speed)*.05)
            distance -= .5 * (speed + next_speed) * .05
            speed = next_speed
        self.assertGreater(distance, 0)
        self.assertLess(speed, .3)


class DynamicsTests(unittest.TestCase):
    def test_online_neural_learning_reduces_transition_error(self):
        model = OnlineDynamics()
        c = Control(throttle=.5)
        speed, dt = 6.0, .1
        actual_next = speed + (model.prior_acceleration(speed, c) + 1.1) * dt
        before = abs(model.predict_speed(speed, c, dt) - actual_next)
        for _ in range(100):
            model.observe(speed, c, actual_next, dt)
        after = abs(model.predict_speed(speed, c, dt) - actual_next)
        self.assertLess(after, before * .1)
        self.assertEqual(model.updates, 100)

    def test_surprisal_reacts_to_unexpected_transition(self):
        policy = GroundedDrivingPolicy()
        first = policy.step(obs())
        normal_next = first.prediction['next_speed']
        normal = policy.step(obs(speed=normal_next))
        unexpected = policy.step(obs(speed=normal_next - .7))
        self.assertGreater(unexpected.surprisal, 4.5)
        self.assertGreater(unexpected.surprisal, normal.surprisal)
        self.assertTrue(unexpected.replanned)
        self.assertIn('unexpected dynamics', unexpected.plan.reason)

    def test_rollout_respects_steering_and_nonnegative_speed(self):
        model = OnlineDynamics()
        curve = model.rollout(8, Control(steer=.4))
        self.assertGreater(curve[-1][1], 0)
        stopped = model.rollout(.1, Control(brake=1))
        self.assertTrue(all(p[3] >= 0 for p in stopped))


if __name__ == '__main__':
    unittest.main()
