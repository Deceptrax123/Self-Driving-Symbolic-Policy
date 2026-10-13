"""A successful replay audit must have actual completed decision evidence."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'audit_policy_report', Path(__file__).resolve().parents[1] / 'scripts/audit_policy_report.py'
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_audit_without_completed_traces_fails_without_writing_report(tmp_path, capsys):
    incomplete = tmp_path / 'reports' / 'interrupted'
    incomplete.mkdir(parents=True)
    (incomplete / 'trace.jsonl').write_text('{}\n')
    output = tmp_path / 'audit.json'
    assert audit.main(['--trace-root', str(tmp_path / 'reports'), '--output', str(output)]) == 1
    assert 'No completed decision traces' in capsys.readouterr().err
    assert not output.exists()


def test_completed_but_empty_trace_is_rejected(tmp_path):
    episode = tmp_path / 'lane_follow'
    episode.mkdir()
    (episode / 'summary.json').write_text('{}')
    (episode / 'trace.jsonl').write_text(' \n\n')
    with pytest.raises(ValueError, match='Completed trace is empty'):
        audit.completed_traces(tmp_path)
