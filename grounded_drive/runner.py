"""Repeatable closed-loop CARLA experiments, with groundings and decisions in JSONL."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

from .carla_adapter import CarlaAdapter, local_xy
from .policy import GroundedDrivingPolicy


def choose_route(world, seed=0, length=100., signal=False):
    """Scenario geometry may use maps; the expert receives only relative observations."""
    starts = list(world.get_map().get_spawn_points())
    random.Random(seed).shuffle(starts)
    for transform in starts:
        start = world.get_map().get_waypoint(transform.location)
        if start.is_junction:
            continue
        route = [start]
        for _ in range(int(length/2)):
            candidates = route[-1].next(2.)
            if not candidates:
                break
            nxt = min(candidates, key=lambda w: abs((w.transform.rotation.yaw-route[-1].transform.rotation.yaw+180)%360-180))
            route.append(nxt)
        if len(route) < int(length/2):
            continue
        # Straight first 65 m makes hazard scenarios reproducible without hardcoded locations.
        if any(abs((w.transform.rotation.yaw-start.transform.rotation.yaw+180)%360-180)>12 for w in route[:33]):
            continue
        matching = []
        for light in world.get_actors().filter('traffic.traffic_light*'):
            for stop in light.get_stop_waypoints():
                if (stop.road_id,stop.lane_id)!=(start.road_id,start.lane_id):
                    continue
                x,y = local_xy(start.transform,stop.transform.location)
                if 25 < x < 65 and abs(y) < start.lane_width:
                    matching.append(light)
        if signal and not matching:
            continue
        return route, matching
    raise RuntimeError('No suitable route found; try a different town/seed')


def run(args):
    import carla
    client = carla.Client(args.host,args.port)
    client.set_timeout(90.)
    world = client.get_world()
    if world.get_settings().synchronous_mode:
        raise RuntimeError('Another synchronous client owns this world; start a separate CARLA server')
    if args.town and world.get_map().name.split('/')[-1] != args.town:
        world = client.load_world(args.town)
    original = world.get_settings()
    original_weather = world.get_weather()
    if original.synchronous_mode:
        raise RuntimeError('Another synchronous client owns this world; start a separate CARLA server')
    owned = []
    light_settings = []
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    collisions = []
    lane_events = []
    video = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode=True
        settings.fixed_delta_seconds=.05
        settings.no_rendering_mode=not args.record_video
        settings.substepping=True
        settings.max_substep_delta_time=.01
        settings.max_substeps=10
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        route, matching = choose_route(world,args.seed,signal=args.scenario=='red_light')
        library = world.get_blueprint_library()
        blueprint = library.find(args.vehicle)
        blueprint.set_attribute('role_name','grounded_expert')
        start = route[0].transform
        start.location.z += .35
        ego = world.spawn_actor(blueprint,start)
        owned.append(ego)
        collision = world.spawn_actor(library.find('sensor.other.collision'),carla.Transform(),attach_to=ego)
        owned.append(collision)
        collision.listen(lambda event: collisions.append({'frame':event.frame,'other':event.other_actor.type_id,'impulse':event.normal_impulse.length()}))
        lane = world.spawn_actor(library.find('sensor.other.lane_invasion'),carla.Transform(),attach_to=ego)
        owned.append(lane)
        lane.listen(lambda event: lane_events.append({'frame':event.frame,'markings':[str(m.type) for m in event.crossed_lane_markings]}))
        for light in world.get_actors().filter('traffic.traffic_light*'):
            light_settings.append((light,light.get_state(),light.is_frozen(),light.get_green_time(),light.get_yellow_time(),light.get_red_time()))
            light.set_state(carla.TrafficLightState.Green)
            light.freeze(True)
        if args.scenario == 'red_light':
            for light in matching:
                light.set_state(carla.TrafficLightState.Red)
        obstacle = None
        if args.scenario == 'blocked_lane':
            transform = route[22].transform
            transform.location.z += .35
            obstacle = world.spawn_actor(library.find('vehicle.carlamotors.carlacola'),transform)
            obstacle.set_simulate_physics(False)
            owned.append(obstacle)
        walker = None
        if args.scenario=='pedestrian':
            transform = route[17].transform
            right=transform.get_right_vector()
            transform.location.x -= right.x*5
            transform.location.y -= right.y*5
            transform.location.z += 1.
            walker=world.spawn_actor(library.find('walker.pedestrian.0013'),transform)
            owned.append(walker)
        for _ in range(30):
            ego.apply_control(carla.VehicleControl(brake=1.))
            world.tick()
            if args.record_video:
                time.sleep(.05)
        if args.record_video:
            from .video import RolloutVideo
            video=RolloutVideo(world,ego,output/'rollout.mp4')
        client.set_timeout(15.)
        adapter=CarlaAdapter(world,ego,route)
        policy=GroundedDrivingPolicy()
        goals=Counter()
        events=[]
        max_lateral=0.
        min_clearance=float('inf')
        progress=0.
        max_speed=0.
        max_surprisal=0.
        stopped_at_hazard=False
        hazard_released=False
        walker_started=False
        completed=False
        red_signal_violations=0
        red_stop_progress=None
        if args.scenario=='red_light':
            candidates=[local_xy(route[0].transform,stop.transform.location)[0]-ego.bounding_box.extent.x for light in matching for stop in light.get_stop_waypoints() if (stop.road_id,stop.lane_id)==(route[0].road_id,route[0].lane_id)]
            red_stop_progress=min((v for v in candidates if v>0),default=None)
        with (output/'trace.jsonl').open('w') as trace:
            for step in range(int(args.seconds/.05)):
                t=step*.05
                if args.scenario=='novel_obstacle' and obstacle is None and not hazard_released and progress>10:
                    transform=route[22].transform
                    transform.location.z += .35
                    obstacle=world.spawn_actor(library.find('vehicle.carlamotors.carlacola'),transform)
                    obstacle.set_simulate_physics(False)
                    owned.append(obstacle)
                    events.append({'t':t,'event':'unseen_truck_appeared'})
                if obstacle is not None and t>=args.release_time:
                    obstacle.destroy()
                    owned.remove(obstacle)
                    obstacle=None
                    events.append({'t':t,'event':'blocked_lane_cleared'})
                    hazard_released=True
                if args.scenario=='red_light' and not hazard_released and t>=args.release_time:
                    for light in matching:
                        light.set_state(carla.TrafficLightState.Green)
                    events.append({'t':t,'event':'signal_green'})
                    hazard_released=True
                if walker is not None and progress>7 and not walker_started:
                    right=route[17].transform.get_right_vector()
                    walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(right.x,right.y,0),speed=1.6))
                    walker_started=True
                    events.append({'t':t,'event':'pedestrian_crossing_started'})
                if walker is not None and walker_started:
                    _, y=local_xy(route[17].transform,walker.get_location())
                    if y>5:
                        walker.apply_control(carla.WalkerControl(speed=0.))
                        if not hazard_released:
                            hazard_released=True
                            events.append({'t':t,'event':'pedestrian_clear'})
                obs=adapter.observe(world.get_snapshot().frame)
                decision=policy.step(obs)
                adapter.apply(decision.control)
                progress=max(progress,adapter.total_length-adapter.remaining)
                if red_stop_progress is not None and not hazard_released and progress>red_stop_progress+.5:
                    red_signal_violations+=1
                max_lateral=max(max_lateral,adapter.lateral_error)
                max_speed=max(max_speed,obs.speed)
                max_surprisal=max(max_surprisal,float(decision.surprisal))
                goals[str(decision.plan.subgoal)]+=1
                for actor in obs.actors:
                    if abs(actor.y)<(actor.width+ego.bounding_box.extent.y*2)/2 and actor.x>0:
                        clearance=actor.x-actor.length/2-ego.bounding_box.extent.x
                        min_clearance=min(min_clearance,clearance)
                        if clearance<18 and obs.speed<.3:
                            stopped_at_hazard=True
                if obs.traffic_light=='red' and obs.stop_line_distance is not None and obs.stop_line_distance<18 and obs.speed<.3:
                    stopped_at_hazard=True
                trace.write(json.dumps({'t':round(t,3),'observation':asdict(obs),'decision':asdict(decision),'progress_m':progress,'lateral_error_m':adapter.lateral_error},default=str)+'\n')
                video_frame=world.tick()
                if video is not None:
                    video.record(video_frame,t+.05,obs,decision,args.scenario)
                if collisions:
                    break
                if adapter.remaining<3. and obs.speed<.3:
                    completed=True
                    break
        if video is not None:
            video.close()
        # Allow queued sensor callbacks to arrive before reporting.
        time.sleep(.15)
        report={'scenario':args.scenario,'server_port':args.port,'town':world.get_map().name,'seed':args.seed,'vehicle':args.vehicle,'steps':step+1,'simulated_seconds':round((step+1)*.05,2),'completed':completed,'red_signal_violations':red_signal_violations,'model_updates':decision.model_updates,'progress_m':round(progress,2),'route_length_m':round(adapter.total_length,2),'collision_count':len(collisions),'collisions':collisions,'lane_invasion_count':len(lane_events),'max_lateral_error_m':round(max_lateral,3),'min_front_clearance_m':None if math.isinf(min_clearance) else round(min_clearance,3),'max_speed_mps':round(max_speed,3),'max_surprisal':round(max_surprisal,4),'stopped_at_hazard':stopped_at_hazard,'hazard_released':hazard_released,'subgoal_counts':dict(goals),'events':events}
        if video is not None:
            report['video']={'path':str(video.path),'frames':video.frames,'fps':20,'duration_seconds':video.frames/20,'width':1280,'height':480,'annotation_state':'control and observation immediately before the rendered transition'}
        report['passed']=bool(completed and not collisions and not lane_events and not red_signal_violations and max_lateral<1.2 and (args.scenario=='lane_follow' or hazard_released))
        if args.scenario in ('blocked_lane','red_light','novel_obstacle','pedestrian'):
            report['passed']=report['passed'] and stopped_at_hazard
        (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return 0 if report['passed'] else 1
    finally:
        # A dead simulator cannot acknowledge actor cleanup. Avoid waiting on
        # each sensor/actor separately after an engine crash.
        client.set_timeout(2.)
        server_alive=True
        try:
            client.get_server_version()
        except RuntimeError:
            server_alive=False
        if video is not None:
            video.close(destroy_sensors=server_alive)
        if server_alive:
            for actor in reversed(owned):
                if actor.is_alive:
                    if hasattr(actor,'stop') and actor.type_id.startswith('sensor.'):
                        actor.stop()
                    actor.destroy()
            for light,state,frozen,green,yellow,red in light_settings:
                if light.is_alive:
                    light.set_state(state)
                    light.set_green_time(green)
                    light.set_yellow_time(yellow)
                    light.set_red_time(red)
                    light.freeze(frozen)
            world.set_weather(original_weather)
            world.apply_settings(original)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=2000)
    parser.add_argument('--town',default='Town01')
    parser.add_argument('--scenario',choices=['lane_follow','blocked_lane','pedestrian','red_light','novel_obstacle'],default='lane_follow')
    parser.add_argument('--vehicle',default='vehicle.tesla.model3')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--seconds',type=float,default=45.)
    parser.add_argument('--release-time',type=float,default=16.)
    parser.add_argument('--output',default='reports/latest')
    parser.add_argument('--record-video',action='store_true',help='Enable RGB rendering and save an annotated chase/front-camera MP4')
    args=parser.parse_args()
    raise SystemExit(run(args))

if __name__=='__main__':
    main()
