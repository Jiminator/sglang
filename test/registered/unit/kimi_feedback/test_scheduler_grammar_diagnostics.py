"""Execute the actual internal-state handler and serialization with CPU fixtures."""
import responses_cpu_helpers
import json
import test_scheduler_internal_state_env_vars as env_tests
from test_grammar_cache_budget import backend, SizedGrammar


def state(backend=None, tp_rank=0, tp_size=1):
    return env_tests.TestSchedulerInternalStateEnvVars()._get_internal_state(backend, tp_rank, tp_size)


def test_actual_internal_state_preserves_capacity_and_exposes_only_numeric_budget(backend):
    backend.set_cache(('json','synthetic-private-schema'),SizedGrammar(2048))
    result=state(backend, tp_rank=3, tp_size=8)
    snapshot=result['grammar_budget']
    assert snapshot['available'] and snapshot['cache_entries']==1
    assert snapshot['approx_cache_bytes']>=2048
    assert snapshot['tp_rank']==3
    assert 'synthetic-private-schema' not in json.dumps(snapshot)
    assert result['effective_max_running_requests_per_dp']==8
    assert result['memory_usage']['token_capacity']==100


def test_disabled_backend_is_unknown_not_zero_capacity():
    snapshot=state()['grammar_budget']
    assert not snapshot['available'] and snapshot['reason']=='backend_disabled'
    assert 'approx_cache_bytes' not in snapshot


def test_busy_snapshot_is_serializable_without_fabricated_zero(backend,monkeypatch):
    monkeypatch.setattr(backend,'budget_snapshot',lambda:{'available':False,'reason':'busy'})
    snapshot=state(backend)['grammar_budget']
    assert snapshot['reason']=='busy' and not snapshot['available']
    assert 'approx_cache_bytes' not in snapshot
    json.dumps(snapshot,allow_nan=False)
