# Recorded symbolic-policy rollouts

These are the four Town01 camera reruns from the original symbolic-policy experiment. The policy uses CARLA geometric state. The camera views are for inspection, with the pre-action observation and command shown in the HUD; each image follows that decision by one 50 ms simulation tick.

| File | Duration | Behavior |
| --- | ---: | --- |
| `blocked-lane.mp4` | 26.15 s | Stop behind the truck and resume after it clears |
| `pedestrian.mp4` | 17.35 s | Yield to the crossing pedestrian and resume |
| `red-light.mp4` | 23.65 s | Stop at red and continue after the signal changes |
| `lane-follow.mp4` | 14.55 s | Follow the route and stop near the destination |

The MP4 copies retain all original frames, 20 fps, 1280 × 480 dimensions, and the original playback speed. They are re-encoded as H.264 with fast-start metadata. The animated blocked-lane preview is reduced to 768 pixels wide and plays at **2× speed**; click through to the full-speed MP4.

The original source paths are `reports/videos/<scenario>/rollout.mp4`. Generated reports remain ignored by Git; these selected media files are intentionally versioned. No Town02 video is included because its camera capture crashed in the original setup. Its successful state-only episode is distinct from the blocked-lane video here.
