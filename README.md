# Grounded neuro-symbolic driving in CARLA

An executable expert-policy research prototype authored with **GPT-6 Astra**. It uses current observations and reusable geometric relations rather than stored town coordinates or actor identities. The initial task is closed-loop driving, evaluation and revision; the wider learned world-model architecture remains future work.

- **System 1:** pure-pursuit route tracking and longitudinal throttle/brake control at 20 Hz.
- **System 2:** re-ground current observations into `vehicle_ahead`, `pedestrian_in_path`, `crossing_conflict`, `stop_signal`, `curve_ahead`, and destination facts. Compose approach, hold, verify-clear, resume and reach-goal subgoals. Re-evaluate preconditions each tick.
- **System M:** an actual online-trained neural acceleration residual over a physical dynamics prior. Predict the next speed, learn from executed transitions, and trigger conservative speed reduction/replanning on unexpected transitions. A bicycle rollout interface is available for expansion.

Astra wrote the policy; the running car uses Python symbolic planning, not an Astra API request per frame. This model is not LEWM. Its input representation is geometric simulator state, not learned camera perception. `surprisal` is normalized prediction error, not a calibrated probability or a reward to maximize while driving. High-surprise scenario discovery can be added later outside the driving objective.

## Run

Requires Docker with NVIDIA GPU support and Python 3.12 for the CARLA client. The official simulator image occupies about 30 GB.

```bash
uv venv --python /usr/bin/python3.12 .venv
uv pip install --python .venv/bin/python -e '.[carla]' pytest
docker pull carlasim/carla:0.9.16
bash scripts/start_carla.sh
# Wait for the simulator world to finish loading.
.venv/bin/python -m grounded_drive.runner --scenario lane_follow --output reports/lane_follow
.venv/bin/python -m grounded_drive.runner --scenario blocked_lane --output reports/blocked_lane
.venv/bin/python -m grounded_drive.runner --scenario pedestrian --output reports/pedestrian
.venv/bin/python -m grounded_drive.runner --scenario red_light --output reports/red_light
# Use a second server to avoid an observed engine crash during map reload.
CARLA_NAME=grounded-drive-town02 CARLA_PORT=2010 CARLA_TOWN=Town02 bash scripts/start_carla.sh
.venv/bin/python -m grounded_drive.runner --town Town02 --scenario novel_obstacle --vehicle vehicle.audi.tt --seed 5 --port 2010 --output reports/novel_obstacle
# Or rerun all five scenarios (after both worlds finish loading):
.venv/bin/python scripts/evaluate.py --heldout-port 2010
.venv/bin/python -m pytest -q
```

Each run writes a full `trace.jsonl` of observations, grounded facts, chosen subgoals, control, predictions, prediction errors and model update counts, plus a `summary.json` with completion, collision and clearance metrics. Run only one client per server; the runner rejects a world already owned by a synchronous client. It restores world settings, weather and traffic-light states and destroys the actors it created.

To stop the server:

```bash
docker stop grounded-drive-carla grounded-drive-town02
```

The launch script restarts an existing stopped container and succeeds if it is already running. Existing containers retain their saved configuration. For a new container, the launcher requests Town01; the rollout runner ensures the selected map is loaded through the CARLA client API; `CARLA_TOWN`, `CARLA_PORT`, and `CARLA_NAME` configure additional servers. To restart both existing containers directly, use `docker start grounded-drive-carla grounded-drive-town02`.

## Watch recorded rollouts

The original evaluations ran without camera rendering and saved traces and metrics. New rendered runs are in `reports/videos/`, separate from those earlier evaluations. Each `rollout.mp4` contains genuine CARLA RGB chase and front-camera views with the current policy decision, grounded facts, speed, controls and surprise score. The HUD shows the state and command immediately before the rendered transition (one 50 ms step).

Open the MP4 files in a video player, or serve the local gallery:

```bash
.venv/bin/python -m http.server 8080 --bind 127.0.0.1 --directory reports/videos
# Open http://localhost:8080 in your browser.
# On a remote machine, forward port 8080 to your computer first.
```

Record one new episode:

```bash
uv pip install --python .venv/bin/python -e '.[carla,video]'
bash scripts/start_carla.sh
.venv/bin/python -m grounded_drive.runner --scenario pedestrian --record-video --output reports/videos/pedestrian
```

Record the four rendered scenarios verified on this machine:

```bash
.venv/bin/python scripts/evaluate.py --record-video --output-root reports/videos --scenarios lane_follow blocked_lane pedestrian red_light
```

Town02 RGB recording currently crashes the simulator engine on this setup; the previous non-rendered Town02 test metrics remain in `reports/novel_obstacle/summary.json`. The local gallery contains four successfully recorded Town01 episodes.

Video recording enables rendering; the policy still receives the same ground-truth observations. This feature does not add camera perception to the policy. RGB buffer format and frame synchronization follow the [CARLA sensor reference](https://carla.readthedocs.io/en/0.9.16/ref_sensors/) and [synchronous simulation documentation](https://carla.readthedocs.io/en/0.9.16/adv_synchrony_timestep/).

## Grounding contract

`GroundedDrivingPolicy.step(Observation)` returns a `Decision`. Units are meters, seconds and meters/second. The ego frame uses x forward, y right. Actor velocity is its absolute world velocity projected into ego axes; it is not relative velocity. The navigation adapter supplies an explicit destination route as ego-relative points. Town names, road IDs and global coordinates never enter the policy. Actor IDs are included only to explain which observed actor grounds a fact.

CARLA's actor and traffic-light APIs supply privileged ground-truth state. This is appropriate for an expert demonstration policy, but sensor perception, occlusion and detection uncertainty are not evaluated. Rules generalize across identities and geometry within this vocabulary; this does not establish general open-world competence. The current policy waits for a blocked lane to clear; it does not perform overtaking, lane changes, stop-sign negotiations or arbitrary intersection priority reasoning.

Implementation: `grounded_drive/types.py`, `grounding.py`, `planner.py`, `policy.py`, `dynamics.py`; simulator integration: `carla_adapter.py` and `runner.py`.

CARLA setup and API references: [official Docker instructions](https://carla.readthedocs.io/en/0.9.16/build_docker/) and [Python API](https://carla.readthedocs.io/en/0.9.16/python_api/).

## Observed results and revisions

All five final episodes passed: lane following, blocked-lane stop/resume, pedestrian yielding, red-light stop/green-light resume, and a newly appearing truck in Town02 with a different ego vehicle. There were zero collisions and zero lane invasions over about 500 m of route. All 16 core and adapter regression tests pass. These are controlled single-actor scenarios; they do not establish performance in dense traffic or arbitrary novel situations.

See [`reports/evaluation.json`](reports/evaluation.json) for metrics and [`reports/RESULTS.md`](reports/RESULTS.md) for the evidence and revisions. Initial outcomes are retained in `reports/baseline/`.

CARLA traces revealed a nearest-waypoint destination-distance oscillation. The adapter now projects continuously onto route segments; the planner also explicitly holds at a reached destination instead of requesting route recovery. Translation/rotation regression tests check that grounding stays invariant.

The simulator engine segfaulted during an in-process Town01→Town02 reload. The held-out episode completed on a fresh separate server with Town02 selected through the client API. The original failure is retained in `reports/carla_map_switch_crash.log` and `reports/suite.log`; the complete final aggregate is `reports/evaluation.json`.

## Mathematical report

The detailed specification and predictor-training guide is available as [PDF](reports/math/POLICY_MATHEMATICS.pdf), [offline HTML](reports/math/POLICY_MATHEMATICS.html), and [Markdown](reports/math/POLICY_MATHEMATICS.md). It covers the exact grounding predicates, numerical constraint arbitration, controller, neural dynamics gradients, surprise score, decision trees, actual worked examples, transition/frame alignment, and dataset requirements. Implemented equations and proposed training extensions are distinguished throughout.

All 3,812 saved decisions across nine completed traces were reproduced by replaying the current policy. [The audit](reports/math/source_and_replay_audit.json) includes source and trace hashes, runtime versions and regression-test results.

To regenerate the report on a machine with Playwright's browser dependencies installed:

```bash
uv pip install --python .venv/bin/python -e '.[report]'
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/audit_policy_report.py
.venv/bin/python scripts/render_policy_report.py
```

The HTML uses bundled KaTeX and Mermaid assets and can be read offline. The PDF is already generated; no browser installation is needed to open it.
