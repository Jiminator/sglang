"""Bounded PR943 synthetic mask/acceptance evidence on an unchanged p5 install."""
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import torch
import xgrammar as xgr

from upstream943_cases import (
    CASE_IDS,
    CROSSING_CACHE_DIFFERENT_FSM_CASES,
    test_grammar_compiler_crossing_cache_different_fsm,
)


def emit(row):
    print(json.dumps(row, ensure_ascii=True), flush=True)


def memory():
    rows = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('Rss', 'Anonymous'):
            rows[key] = int(value.split()[0]) * 1024
    return rows


package = Path(xgr.__file__).parent
expected = {
    'compiler.py': '67e2154127ac14c7f9afde0761755124f2a981f7b0c5a11a1fb96092b6fd109c',
    'libxgrammar_bindings.so': '03b6b083f1a488c7154870bc70b4317eef21ce5d036664cf6893a7c3e91c38eb',
}
observed = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in expected}
version = importlib.metadata.version('xgrammar')
emit({'phase': 'identity', 'xgrammar': version, 'package_path': str(package),
      'compiler_signature': str(inspect.signature(xgr.GrammarCompiler)),
      'package_sha256': observed, 'matches_p6_wheel': observed == expected,
      'torch': torch.__version__, 'nvidia_device_present': Path('/dev/nvidia0').exists(),
      'NVIDIA_VISIBLE_DEVICES': os.environ.get('NVIDIA_VISIBLE_DEVICES'),
      'real_tokenizer': False, 'synthetic_vocab_size': 14, **memory()})
assert version == '0.2.6rc2+phala.feedback.20261017.p6'
assert observed == expected
assert not Path('/dev/nvidia0').exists()
torch.set_num_threads(1)

VOCAB = ['(', ')', 'a', '0', 'x', 'y', '"', '""', 'xa', 'x0', 'q', 'qx\x01', 'qa', 'q0']
INFO = xgr.TokenizerInfo(VOCAB)
# Additional controls do not replace any upstream case. A traces give reverse
# compilation a valid path for A; the upstream path is always retained as B.
A_TRACES = [
    '(0x)', '("")', '(x0)', '(xa)', ['(', 'a', 4, ')'], '(qxc)',
    '(qa)', '(ax)', '(qa])', '(aqab)',
]


def compiler(cache_enabled, max_threads=1, cache_limit_bytes=-1):
    return xgr.GrammarCompiler(INFO, max_threads=max_threads,
                               cache_enabled=cache_enabled, cache_limit_bytes=cache_limit_bytes)


def accept_unit(matcher, unit):
    return matcher.accept_token(unit) if isinstance(unit, int) else matcher.accept_string(unit)


def matcher(compiled):
    return xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)


def allows(compiled_matcher):
    mask = xgr.allocate_token_bitmask(1, INFO.vocab_size)
    xgr.reset_token_bitmask(mask)
    apply = compiled_matcher.fill_next_token_bitmask(mask)
    bits = [not apply or bool((int(mask[0, token // 32]) >> (token % 32)) & 1)
            for token in range(len(VOCAB))]
    return bool(apply), bits, mask


def oracle(compiled, prefix, token):
    trial = matcher(compiled)
    assert all(accept_unit(trial, unit) for unit in prefix)
    return bool(trial.accept_token(token))


def accepted(compiled, units):
    trial = matcher(compiled)
    return bool(all(accept_unit(trial, unit) for unit in units) and trial.is_terminated())


def compare_trace(cached, fresh, units):
    actual, reference = matcher(cached), matcher(fresh)
    prefix = []
    differences = []
    fresh_oracle_differences = []
    parser_differences = []
    raw_mask_differences = 0
    for offset, unit in enumerate(units):
        if reference.is_terminated():
            break
        ca, cb, cm = allows(actual)
        fa, fb, fm = allows(reference)
        if ca != fa or not torch.equal(cm, fm):
            raw_mask_differences += 1
        for token in range(len(VOCAB)):
            independent = oracle(fresh, prefix, token)
            if fb[token] != independent:
                fresh_oracle_differences.append({'offset': offset, 'prefix': prefix.copy(),
                    'token_id': token, 'token': VOCAB[token], 'fresh_allows': fb[token],
                    'oracle_accepts': independent})
            if cb[token] != fb[token]:
                classification = ('cached_false_accept' if cb[token] and not independent else
                                  'cached_false_reject' if not cb[token] and independent else
                                  'fresh_oracle_disagreement')
                differences.append({'offset': offset, 'prefix': prefix.copy(),
                    'token_id': token, 'token': VOCAB[token], 'cached_allows': cb[token],
                    'fresh_allows': fb[token], 'oracle_accepts': independent,
                    'classification': classification})
        a = bool(accept_unit(actual, unit))
        b = bool(accept_unit(reference, unit))
        if a != b:
            parser_differences.append({'offset': offset, 'cached_accepts': a, 'fresh_accepts': b})
        if not b:
            break
        prefix.append(unit)
    return {'raw_mask_difference_positions': raw_mask_differences,
            'effective_token_differences': differences,
            'fresh_mask_vs_oracle': fresh_oracle_differences,
            'parser_differences': parser_differences,
            'full_trace_cached_accepts': accepted(cached, units),
            'full_trace_fresh_accepts': accepted(fresh, units)}


started = time.monotonic()
original = []
for case_id, case in zip(CASE_IDS, CROSSING_CACHE_DIFFERENT_FSM_CASES):
    begin = time.monotonic()
    try:
        test_grammar_compiler_crossing_cache_different_fsm(*case)
        row = {'case': case_id, 'status': 'passed'}
    except Exception as error:
        row = {'case': case_id, 'status': 'assertion_failure' if isinstance(error, AssertionError)
               else 'api_or_compile_error', 'exception': type(error).__name__,
               'error': str(error), 'traceback': traceback.format_exc(limit=3)}
    row.update(phase='original_pr_regression', seconds=time.monotonic() - begin)
    original.append(row)
    emit(row)

orders = []
for index, (case_id, case) in enumerate(zip(CASE_IDS, CROSSING_CACHE_DIFFERENT_FSM_CASES)):
    grammar_a, grammar_b, input_b = case
    for order in ('A_then_B', 'B_then_A'):
        begin = time.monotonic()
        source, target = (grammar_a, grammar_b) if order == 'A_then_B' else (grammar_b, grammar_a)
        try:
            retained = compiler(True)
            retained.compile_grammar(source)
            cached = retained.compile_grammar(target)
            fresh = compiler(False).compile_grammar(target)
            unseeded = compiler(True).compile_grammar(target)
            rows = []
            for label, units in [('upstream_B_trace', input_b), ('supplemental_A_trace', A_TRACES[index])]:
                rows.append({'trace': label, 'units': list(units),
                             'seeded': compare_trace(cached, fresh, units),
                             'unseeded_control': compare_trace(unseeded, fresh, units)})
            row = {'case': case_id, 'order': order, 'status': 'completed', 'traces': rows,
                   'native_cache_bytes': retained.get_cache_size_bytes()}
        except Exception as error:
            row = {'case': case_id, 'order': order, 'status': 'api_or_compile_error',
                   'exception': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc(limit=3)}
        row.update(phase='compile_order_control', seconds=time.monotonic() - begin)
        orders.append(row)
        emit(row)

affected = set()
counts = {'cached_false_accept': 0, 'cached_false_reject': 0,
          'fresh_oracle_disagreement': 0, 'fresh_mask_vs_oracle': 0}
for row in orders:
    for trace in row.get('traces', []):
        seeded = trace['seeded']
        for difference in seeded['effective_token_differences']:
            counts[difference['classification']] += 1
            affected.add(row['case'])
        counts['fresh_mask_vs_oracle'] += len(seeded['fresh_mask_vs_oracle'])

emit({'phase': 'summary', 'original_cases': len(original),
      'original_passed': sum(row['status'] == 'passed' for row in original),
      'original_failed': sum(row['status'] == 'assertion_failure' for row in original),
      'original_api_or_compile_errors': sum(row['status'] == 'api_or_compile_error' for row in original),
      'order_controls': len(orders),
      'order_controls_completed': sum(row['status'] == 'completed' for row in orders),
      'affected_cases': sorted(affected), 'observations': counts,
      'seconds': time.monotonic() - started, **memory()})
