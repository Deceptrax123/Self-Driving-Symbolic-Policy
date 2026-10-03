"""System 2: grounded receding-horizon subgoals with explicit preconditions.

The route is supplied by navigation. This planner composes stop/yield/follow/track
options and re-evaluates their preconditions on every observation. It does not
promise arbitrary open-world reasoning or learn a route-specific action script.
"""
import math
from .types import Grounding, Observation, Plan


class SymbolicPlanner:
    def plan(self, obs: Observation, grounding: Grounding, surprisal=0.0):
        if not grounding.route_valid and obs.distance_to_goal is not None and obs.distance_to_goal <= 2.0:
            return Plan('hold_at_goal', 0.0, 'Destination reached; hold position',
                        ('hold_at_goal',), grounding.facts, obs.speed > 0.5)
        if not grounding.route_valid:
            return Plan('recover_route', 0.0, 'No grounded route to execute',
                        ('hold_position', 'request_valid_route'), grounding.facts, True)
        speed = min(obs.speed_limit, grounding.curve_speed)
        subgoal, reason = 'cruise', 'Route corridor is clear'
        steps = ('track_route', 'reach_destination')
        emergency = False
        for fact in grounding.facts:
            target = speed
            candidate, explanation = subgoal, reason
            candidate_steps = steps
            if fact.predicate == 'goal_distance':
                target = math.sqrt(2 * 2.0 * max(0, fact.distance - 2.0))
                candidate, explanation = 'reach_goal', 'Brake to the grounded destination'
                candidate_steps = ('approach_destination', 'hold_at_goal')
            elif fact.predicate == 'stop_signal':
                target = math.sqrt(2 * 2.5 * max(0, fact.distance - 2.5))
                candidate, explanation = 'stop_for_signal', 'Stop before the active signal line'
                candidate_steps = ('approach_stop_line', 'hold_until_signal_clears', 'track_route')
            elif fact.predicate == 'vehicle_ahead':
                # Time-headway following with an independent full-stop envelope.
                lead_speed = fact.speed or 0
                desired_gap = 4.0 + 1.5 * obs.speed
                target = min(lead_speed + 0.5 * (fact.distance - desired_gap),
                             math.sqrt(2 * 3.0 * max(0, fact.distance - 3.0)) + lead_speed)
                candidate, explanation = 'follow', 'Maintain a grounded clearance and time headway'
                candidate_steps = ('match_lead_speed', 'maintain_clearance', 'resume_when_clear')
            elif fact.predicate in ('pedestrian_in_path', 'obstacle_ahead', 'crossing_conflict'):
                margin = 4.0 if fact.predicate != 'obstacle_ahead' else 3.0
                target = math.sqrt(2 * 3.0 * max(0, fact.distance - margin))
                candidate = 'yield' if fact.predicate != 'obstacle_ahead' else 'stop_for_obstacle'
                explanation = 'Wait for the predicted route conflict to clear'
                candidate_steps = ('brake_before_conflict', 'verify_corridor_clear', 'resume_route')
            if target < speed or (target == 0 and speed == 0):
                speed, subgoal, reason, steps = max(0, target), candidate, explanation, candidate_steps
            if fact.predicate in ('vehicle_ahead', 'pedestrian_in_path', 'obstacle_ahead', 'crossing_conflict'):
                # Crossing TTC is actor occupancy time; clearance is the safer
                # emergency condition for actors that have not reached the path.
                imminent = fact.distance < 2.0 or (fact.predicate != 'crossing_conflict' and fact.ttc is not None and fact.ttc < 1.0)
                insufficient_braking = fact.distance < obs.speed**2 / 12.0 + 1.0 and (fact.speed or 0) < obs.speed
                emergency |= imminent or insufficient_braking
            if fact.predicate == 'stop_signal':
                emergency |= fact.distance < obs.speed**2 / 12.0 + 1.0 and obs.speed > 0.5
        if emergency:
            speed = 0.0
            reason += '; emergency braking envelope active'
        if surprisal > 4.5:
            speed *= 0.7
            reason += '; unexpected dynamics: reduce speed and re-ground'
        return Plan(subgoal, speed, reason, steps, grounding.facts, emergency)
