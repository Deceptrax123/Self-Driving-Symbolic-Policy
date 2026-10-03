"""Run the complete suite sequentially on one CARLA server and summarize evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CASES=[
    ('lane_follow',[]),
    ('blocked_lane',[]),
    ('pedestrian',[]),
    ('red_light',[]),
    ('novel_obstacle',['--town','Town02','--vehicle','vehicle.audi.tt','--seed','5']),
]

def summarize(report_root=ROOT/'reports', names=None):
    results=[]
    for name,_ in CASES:
        if names and name not in names:
            continue
        folder=report_root/name
        report=json.loads((folder/'summary.json').read_text())
        rows=[json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
        prior_errors=[]
        neural_errors=[]
        for previous,current in zip(rows,rows[1:]):
            v=previous['observation']['speed']
            c=previous['decision']['control']
            dt=current['observation']['dt']
            truth=current['observation']['speed']
            # Match the model's nonnegative speed boundary and prediction alignment.
            prior=max(0.,v+(3*c['throttle']-7*c['brake']-.12*v)*dt)
            prior_errors.append(abs(prior-truth))
            neural_errors.append(abs(previous['decision']['prediction']['next_speed']-truth))
        report['prior_speed_prediction_mae_mps']=round(sum(prior_errors)/len(prior_errors),6)
        report['online_neural_speed_prediction_mae_mps']=round(sum(neural_errors)/len(neural_errors),6)
        report['surprise_replans']=sum(row['decision']['replanned'] and row['decision']['surprisal']>4.5 for row in rows)
        report['trace']=str((folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder)/'trace.jsonl')
        results.append(report)
    evaluation={'all_passed':all(r['passed'] for r in results),'episodes':len(results),'total_collisions':sum(r['collision_count'] for r in results),'total_lane_invasions':sum(r['lane_invasion_count'] for r in results),'total_route_length_m':round(sum(r['route_length_m'] for r in results),2),'results':results}
    (report_root/'evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')
    print(json.dumps(evaluation,indent=2),flush=True)
    return evaluation

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--heldout-port',type=int,default=2000,help='Use a separate Town02 server to avoid map reloads')
    parser.add_argument('--record-video',action='store_true')
    parser.add_argument('--output-root',default='reports')
    parser.add_argument('--scenarios',nargs='+',choices=[name for name,_ in CASES],help='Run and summarize selected cases')
    args=parser.parse_args()
    report_root=Path(args.output_root)
    if not report_root.is_absolute():
        report_root=ROOT/report_root
    if not args.summarize_only:
        for name,options in CASES:
            if args.scenarios and name not in args.scenarios:
                continue
            if name=='novel_obstacle':
                options=[*options,'--port',str(args.heldout_port)]
            command=[sys.executable,'-m','grounded_drive.runner','--scenario',name,'--output',str(report_root/name),*options]
            if args.record_video:
                command.append('--record-video')
            print(f'Running {name}',flush=True)
            result=subprocess.run(command,cwd=ROOT,timeout=300)
            if result.returncode!=0:
                raise SystemExit(f'{name} failed (exit {result.returncode}); see {report_root/name}/summary.json')
    evaluation=summarize(report_root,args.scenarios)
    if args.record_video:
        from video_gallery import write_gallery
        write_gallery(report_root)
    raise SystemExit(0 if evaluation['all_passed'] else 1)
