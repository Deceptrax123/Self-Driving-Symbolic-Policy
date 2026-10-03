"""Check the report's expert provenance by replaying saved decision traces."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grounded_drive.policy import GroundedDrivingPolicy
from grounded_drive.types import Actor, Observation


def differences(actual,expected,path='',tolerance=1e-10):
    if isinstance(actual,dict):
        if set(actual)!=set(expected):
            return [path+': keys differ']
        return [d for k in actual for d in differences(actual[k],expected[k],path+'.'+k,tolerance)]
    if isinstance(actual,list):
        if len(actual)!=len(expected):
            return [path+': length differs']
        return [d for i,(a,b) in enumerate(zip(actual,expected)) for d in differences(a,b,f'{path}[{i}]',tolerance)]
    if isinstance(actual,(float,int)) and not isinstance(actual,bool) and isinstance(expected,(float,int)):
        return [] if abs(actual-expected)<=tolerance else [f'{path}: {actual} != {expected}']
    return [] if actual==expected else [f'{path}: values differ']


def main():
    sources=sorted((ROOT/'grounded_drive').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+[ROOT/'scripts/evaluate.py',ROOT/'pyproject.toml',Path(__file__).resolve()]
    manifest={'report_date':'2026-10-04 Asia/Kolkata','runtime_versions':{name:importlib.metadata.version(name) for name in ['numpy','carla','pytest']},'python_version':sys.version,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'replay_tolerance':1e-10,'episodes':[]}
    traces=sorted((ROOT/'reports').glob('*/trace.jsonl'))+sorted((ROOT/'reports/videos').glob('*/trace.jsonl'))
    for path in traces:
        if not path.with_name('summary.json').exists():
            continue # Failed rendering attempts do not define complete episodes.
        policy=GroundedDrivingPolicy()
        mismatches=[]
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for index,row in enumerate(rows):
            values=dict(row['observation'])
            values['route']=tuple(tuple(p) for p in values['route'])
            values['actors']=tuple(Actor(**actor) for actor in values['actors'])
            decision=json.loads(json.dumps(asdict(policy.step(Observation(**values)))))
            delta=differences(decision,row['decision'])
            if delta:
                mismatches.append({'row':index,'differences':delta[:4]})
        manifest['episodes'].append({'trace':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(rows),'transition_pairs':max(0,len(rows)-1),'mismatch_rows':len(mismatches),'examples':mismatches[:3]})
    unit=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True)
    manifest['regression_tests']={'exit_code':unit.returncode,'output':unit.stdout.strip()}
    manifest['all_replays_match']=all(x['mismatch_rows']==0 for x in manifest['episodes'])
    (ROOT/'reports/math/source_and_replay_audit.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'episodes':len(manifest['episodes']),'decision_rows':sum(x['rows'] for x in manifest['episodes']),'all_replays_match':manifest['all_replays_match'],'tests':manifest['regression_tests']},indent=2))
    return 0 if manifest['all_replays_match'] and unit.returncode==0 else 1

if __name__=='__main__':
    raise SystemExit(main())
