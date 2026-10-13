"""Check the report's expert provenance by replaying saved decision traces."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
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


def completed_traces(trace_root):
    """Require actual replay evidence, not a vacuous success over zero rows."""
    traces = sorted(trace_root.glob('*/trace.jsonl')) + sorted(trace_root.glob('videos/*/trace.jsonl'))
    traces = [path for path in traces if path.with_name('summary.json').exists()]
    if not traces:
        raise ValueError(f'No completed decision traces found under {trace_root}')
    for path in traces:
        if not path.read_text().strip():
            raise ValueError(f'Completed trace is empty: {path}')
    return traces


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-root', type=Path, default=ROOT/'reports',
                        help='Existing report folder containing completed traces')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/math/source_and_replay_audit.json')
    args = parser.parse_args(argv)
    try:
        traces = completed_traces(args.trace_root)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    versions = {}
    for name in ['numpy', 'carla', 'pytest']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources=sorted((ROOT/'grounded_drive').glob('*.py'))+sorted((ROOT/'tests').glob('*.py'))+[ROOT/'scripts/evaluate.py',ROOT/'pyproject.toml',Path(__file__).resolve()]
    manifest={'audit_timestamp_utc':datetime.now(timezone.utc).isoformat(),'runtime_versions':versions,'python_version':sys.version,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'replay_tolerance':1e-10,'episodes':[]}
    for path in traces:
        policy=GroundedDrivingPolicy()
        mismatches=[]
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for index,row in enumerate(rows):
            values=dict(row['observation'])
            values['route']=tuple(tuple(p) for p in values['route'])
            values['actors']=tuple(Actor(**actor) for actor in values['actors'])
            decision=json.loads(json.dumps(asdict(policy.step(Observation(**values)))))
            delta=differences(decision,row['decision'])
            if delta:
                mismatches.append({'row':index,'differences':delta[:4]})
        manifest['episodes'].append({'trace':str(path.relative_to(args.trace_root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(rows),'transition_pairs':max(0,len(rows)-1),'mismatch_rows':len(mismatches),'examples':mismatches[:3]})
    unit=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True)
    manifest['regression_tests']={'exit_code':unit.returncode,'output':unit.stdout.strip()}
    manifest['all_replays_match']=all(x['mismatch_rows']==0 for x in manifest['episodes'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'episodes':len(manifest['episodes']),'decision_rows':sum(x['rows'] for x in manifest['episodes']),'all_replays_match':manifest['all_replays_match'],'tests':manifest['regression_tests']},indent=2))
    return 0 if manifest['all_replays_match'] and unit.returncode==0 else 1

if __name__=='__main__':
    raise SystemExit(main())
