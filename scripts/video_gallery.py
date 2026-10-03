"""Build a local browser gallery for completed CARLA RGB recordings."""
import json
from pathlib import Path
import sys

DESCRIPTIONS={
 'blocked_lane':('Blocked lane','Stops behind a truck, waits, and resumes when the lane clears.'),
 'pedestrian':('Pedestrian crossing','Yields to a pedestrian crossing the planned route.'),
 'red_light':('Traffic light','Stops at red and continues when the signal turns green.'),
 'novel_obstacle':('New obstacle in Town02','Re-grounds the situation when a truck appears after driving begins.'),
 'lane_follow':('Lane following','Follows the route and stops at the destination.'),
}

def write_gallery(folder):
    folder=Path(folder)
    videos=[]
    for name,(title,description) in DESCRIPTIONS.items():
        summary=folder/name/'summary.json'
        if not summary.exists():
            continue
        report=json.loads(summary.read_text())
        if not report.get('video') or not (folder/name/'rollout.mp4').exists():
            continue
        videos.append({'name':name,'title':title,'description':description,'town':report['town'].split('/')[-1],'duration':report['video']['duration_seconds'],'collisions':report['collision_count'],'passed':report['passed'],'events':report['events']})
    if not videos:
        raise RuntimeError('No completed RGB videos found')
    data=json.dumps(videos).replace('<','\\u003c')
    page='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARLA driving rollouts</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0c1119;color:#edf2fa}body{margin:0;padding:32px 20px}main{max-width:1280px;margin:auto}h1{font-size:32px;margin:0 0 8px}p{color:#a9b9ce;line-height:1.5}video{width:100%;background:#000;border-radius:12px;display:block}#choices{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0 18px}button{font:inherit;cursor:pointer;border:1px solid #31445c;border-radius:8px;padding:10px 15px;background:#172232;color:#d9e4f2}button[aria-pressed=true]{background:#173e39;border-color:#65d5b3;color:#a5f2dd}#events{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}#details{display:flex;gap:12px;flex-wrap:wrap;align-items:center;justify-content:space-between}#status{color:#8ce0c0}a{color:#8ce0c0}small{color:#9fb0c6}h2{font-size:20px;margin-bottom:0}#description{margin-top:8px}
</style></head><body><main><h1>CARLA driving rollouts</h1><p>Captured chase and front cameras, with the expert policy’s decisions shown below.</p>
<nav id="choices" aria-label="Choose driving scenario"></nav>
<video id="player" controls playsinline preload="metadata"></video>
<div id="details"><h2 id="title"></h2><span id="status"></span><a id="download" download>Download MP4</a></div>
<p id="description"></p><div id="events" aria-label="Jump to a scenario event"></div><small>The HUD shows the observation and command immediately before each rendered transition (50 ms).</small>
</main><script>
const videos={{VIDEOS_JSON}};
const player=document.getElementById('player');
const buttons=[];
function selectVideo(index){
 const v=videos[index];player.pause();player.src=v.name+'/rollout.mp4';player.poster=v.name+'/preview.png';player.load();
 document.getElementById('title').textContent=v.title;
 document.getElementById('description').textContent=v.description;
 document.getElementById('status').textContent=v.town+' · '+v.duration.toFixed(2)+' seconds · '+v.collisions+' collisions · '+(v.passed?'Passed':'Review required');
 document.getElementById('download').href=v.name+'/rollout.mp4';
 buttons.forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
 const events=document.getElementById('events');events.replaceChildren();
 for(const event of v.events){const b=document.createElement('button');b.textContent=event.event.replaceAll('_',' ')+' ('+event.t.toFixed(2)+'s)';b.onclick=()=>{player.currentTime=Math.max(0,event.t-1);player.play().catch(()=>{});};events.append(b);}
}
videos.forEach((v,i)=>{const b=document.createElement('button');b.textContent=v.title;b.onclick=()=>selectVideo(i);buttons.push(b);document.getElementById('choices').append(b);});
selectVideo(0);
</script></body></html>
'''.replace('{{VIDEOS_JSON}}',data)
    (folder/'index.html').write_text(page)
    print(f'Gallery: {folder / "index.html"} ({len(videos)} videos)')

if __name__=='__main__':
    write_gallery(sys.argv[1] if len(sys.argv)>1 else 'reports/videos')
