"""Synthetic #943 masks through the unmodified product dispatch lifecycle.

Only task-local observation wrappers add deterministic scheduling pauses. The
native callable, cache settings, product gate, clear, and result path are real.
"""
import contextlib
import hashlib
import importlib.metadata
import io
import json
import logging
from pathlib import Path
import threading
import time

logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()):
    import torch
    import xgrammar as xgr
    from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
    from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend

torch.set_num_threads(1)
VOCAB = ['(', ')', 'a', '0', 'x', 'y', '"', '""', 'xa', 'x0', 'q', 'qx\x01', 'qa', 'q0']
INFO = xgr.TokenizerInfo(VOCAB)
CASES = [
    ('lookahead-rule-ref',
     'root ::= "(" r z ")"\nr ::= "q" (=z)\nv ::= "[" w "]" | "a"\nw ::= v | "{" z "}"\nz ::= w | "b"',
     'root ::= "(" r z ")"\nr ::= "q" (=z)\nv ::= "[" w "]" | "0"\nw ::= v | "{" z "}"\nz ::= w | "b"', [0]),
    ('partial-hash-start-state',
     'root ::= "(" r ")"\nr ::= z | "qx"\nv ::= "q" w "]" | "a"\nw ::= v | "{" z "}"\nz ::= w | "b"',
     'root ::= "(" r ")"\nr ::= z | "qx"\nv ::= "q" w "]" | "0"\nw ::= v | "{" z "}"\nz ::= w | "b"', [0]),
    ('partial-hash-self-reference',
     'root ::= "(" r ")"\nr ::= x "b" | "aq" r | "az"\nx ::= r "c" | y\ny ::= x "d" | "a"',
     'root ::= "(" r ")"\nr ::= x "b" | "aq" r | "az"\nx ::= r "c" | y\ny ::= x "d" | "0"', [0, 2]),
]


class SyntheticTokenizer:
    def init_xgrammar(self):
        return INFO, None


def emit(value):
    print(json.dumps(value), flush=True)


def fingerprint(compiled, prefix):
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert all(matcher.accept_token(token) for token in prefix)
    assert not matcher.is_terminated()
    mask = xgr.allocate_token_bitmask(1, len(VOCAB))
    xgr.reset_token_bitmask(mask)
    apply = bool(matcher.fill_next_token_bitmask(mask))
    allowed = [not apply or bool((int(mask[0, token // 32]) >> (token % 32)) & 1)
               for token in range(len(VOCAB))]
    return {'apply': apply, 'allowed': allowed, 'raw': [int(value) for value in mask[0]]}


def differences(actual, expected):
    return [{'token_id': index, 'cached_allows': actual['allowed'][index],
             'fresh_allows': expected['allowed'][index]}
            for index in range(len(VOCAB))
            if actual['allowed'][index] != expected['allowed'][index]]


def run(case_id, source, target, prefix, order, window):
    backend = XGrammarGrammarBackend(SyntheticTokenizer(), len(VOCAB))
    candidate = hasattr(backend, '_compile_condition')
    assert backend.use_cache_budget and backend.executor._max_workers == 2
    assert backend.grammar_compiler.cache_limit_bytes == 256 << 20
    fresh = xgr.GrammarCompiler(INFO, max_threads=2, cache_enabled=False,
                               cache_limit_bytes=256 << 20).compile_grammar(target)
    expected = fingerprint(fresh, prefix)
    hold_reached, release = threading.Event(), threading.Event()
    events = []
    event_lock = threading.Lock()
    began = time.monotonic()
    original_compile = backend._compile
    original_from = backend._from_context

    def record(event, which):
        with event_lock:
            events.append({'event': event, 'grammar': which,
                           'seconds': time.monotonic() - began})

    def observed(compile_fn, *args, **kwargs):
        which = 'source' if args[0] == source else 'target'

        def native(*a, **k):
            record('native-entry', which)
            result = compile_fn(*a, **k)
            record('native-return', which)
            if which == 'source' and window == 'before_compile_finally':
                hold_reached.set()
                assert release.wait(15), 'source scheduling pause timed out'
            return result

        return original_compile(native, *args, **kwargs)

    def observed_from(ctx, key_string, grammar_stats):
        which = 'source' if key_string == source else 'target'
        record('from-context-entry', which)
        if which == 'source' and window == 'after_compile_before_from_context':
            hold_reached.set()
            assert release.wait(15), 'post-native scheduling pause timed out'
        return original_from(ctx, key_string, grammar_stats)

    backend._compile = observed
    backend._from_context = observed_from
    first = second = None
    try:
        first, hit = backend.get_cached_or_future_value(('ebnf', source), False)
        assert not hit
        if window == 'serial':
            first_value = first.result(15)
            assert not isinstance(first_value, InvalidGrammarObject)
        else:
            assert hold_reached.wait(10), 'source did not reach controlled product point'
            assert not first.done()
        seed_cache_bytes = backend.grammar_compiler.get_cache_size_bytes()
        second, hit = backend.get_cached_or_future_value(('ebnf', target), False)
        assert not hit
        target_value = second.result(15)
        assert not isinstance(target_value, InvalidGrammarObject)
        actual = fingerprint(target_value.ctx, prefix)
        before_last_release_bytes = backend.grammar_compiler.get_cache_size_bytes()
        active = getattr(backend, '_active_native_compiles', None)
        pending = getattr(backend, '_native_clear_pending', None)
        source_pending_when_target_returned = not first.done()
        release.set()
        assert not isinstance(first.result(15), InvalidGrammarObject)
        backend.release_future(first)
        backend.release_future(second)
        backend.executor.shutdown(wait=True)
        after = fingerprint(target_value.ctx, prefix)
        charge = backend.grammar_compiler.get_cache_size_bytes()
        snapshot = backend.budget_snapshot()
        assert charge == 0 and snapshot['pending_requests'] == snapshot['inflight_compiles'] == 0
        diff = differences(actual, expected)
        expected_defect = False
        assert bool(diff) == expected_defect, (candidate, window, case_id, order, diff)
        assert actual == after, 'clearing later changed the already compiled target mask'
        emit({'phase': 'lifecycle_control', 'candidate': candidate, 'case': case_id,
              'order': order, 'window': window, 'prefix_token_ids': prefix,
              'source_pending_when_target_returned': source_pending_when_target_returned,
              'seed_cache_bytes': seed_cache_bytes,
              'before_last_release_cache_bytes': before_last_release_bytes,
              'active_reservations_before_release': active, 'clear_pending_before_release': pending,
              'fresh_mask': expected, 'target_mask': actual, 'mask_differences': diff,
              'same_mask_after_final_clear': actual == after, 'final_native_cache_bytes': charge,
              'final_snapshot': snapshot, 'events': events})
    finally:
        release.set()
        backend.executor.shutdown(wait=True)
        backend.reset()


package = Path(xgr.__file__).parent
expected_hashes = {'compiler.py': '67e2154127ac14c7f9afde0761755124f2a981f7b0c5a11a1fb96092b6fd109c',
                   'libxgrammar_bindings.so': '03b6b083f1a488c7154870bc70b4317eef21ce5d036664cf6893a7c3e91c38eb'}
observed_hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                   for name in expected_hashes}
assert observed_hashes == expected_hashes
assert importlib.metadata.version('xgrammar') == '0.2.6rc2+phala.feedback.20261017.p6'
emit({'phase': 'identity', 'xgrammar': importlib.metadata.version('xgrammar'),
      'package_sha256': observed_hashes, 'matches_p6_wheel': True,
      'synthetic_vocab_size': len(VOCAB), 'real_tokenizer': False,
      'python_workers': 2, 'native_threads': 2, 'native_cache_limit_bytes': 256 << 20})
for case_id, grammar_a, grammar_b, prefix in CASES:
    for order, source, target in [('A_then_B', grammar_a, grammar_b),
                                  ('B_then_A', grammar_b, grammar_a)]:
        for window in ('serial', 'before_compile_finally', 'after_compile_before_from_context'):
            run(case_id, source, target, prefix, order, window)
emit({'phase': 'finished', 'controls': 18})
