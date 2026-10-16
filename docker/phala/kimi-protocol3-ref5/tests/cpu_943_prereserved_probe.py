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


def run(case_id, source, target, prefix, order):
    backend = XGrammarGrammarBackend(SyntheticTokenizer(), len(VOCAB))
    candidate = hasattr(backend, '_compile_condition')
    assert backend.use_cache_budget and backend.executor._max_workers == 2
    assert backend.grammar_compiler.cache_limit_bytes == 256 << 20
    fresh = xgr.GrammarCompiler(INFO, max_threads=2, cache_enabled=False,
                               cache_limit_bytes=256 << 20).compile_grammar(target)
    expected = fingerprint(fresh, prefix)
    source_native_done, source_release = threading.Event(), threading.Event()
    target_reserved, target_release = threading.Event(), threading.Event()
    events, event_lock = [], threading.Lock()
    began = time.monotonic()
    original_compile = backend._compile

    def record(event):
        with event_lock:
            events.append({'event': event, 'seconds': time.monotonic() - began})

    def observed(compile_fn, *args, **kwargs):
        is_source = args[0] == source
        if not is_source:
            record('target-dispatch-before-native')
            target_reserved.set()
            assert target_release.wait(15), 'target reservation pause timed out'

        def native(*a, **k):
            record('source-native-entry' if is_source else 'target-native-entry')
            result = compile_fn(*a, **k)
            record('source-native-return' if is_source else 'target-native-return')
            if is_source:
                source_native_done.set()
                assert source_release.wait(15), 'source scheduling pause timed out'
            return result

        return original_compile(native, *args, **kwargs)

    backend._compile = observed
    first = second = None
    try:
        first, hit = backend.get_cached_or_future_value(('ebnf', source), False)
        assert not hit and source_native_done.wait(10)
        second, hit = backend.get_cached_or_future_value(('ebnf', target), False)
        assert not hit and target_reserved.wait(10)
        active_before_source_release = getattr(backend, '_active_native_compiles', None)
        source_release.set()
        assert not isinstance(first.result(15), InvalidGrammarObject)
        cache_before_target_native = backend.grammar_compiler.get_cache_size_bytes()
        pending_before_target_native = getattr(backend, '_native_clear_pending', None)
        active_before_target_native = getattr(backend, '_active_native_compiles', None)
        assert not second.done()
        target_release.set()
        value = second.result(15)
        assert not isinstance(value, InvalidGrammarObject)
        actual = fingerprint(value.ctx, prefix)
        diff = differences(actual, expected)
        assert not diff, (candidate, case_id, order, diff)
        backend.release_future(first)
        backend.release_future(second)
        backend.executor.shutdown(wait=True)
        assert backend.grammar_compiler.get_cache_size_bytes() == 0
        assert actual == fingerprint(value.ctx, prefix)
        emit({'phase': 'prereserved_control', 'candidate': candidate, 'case': case_id,
              'order': order, 'prefix_token_ids': prefix,
              'active_before_source_release': active_before_source_release,
              'active_before_target_native': active_before_target_native,
              'pending_before_target_native': pending_before_target_native,
              'cache_before_target_native': cache_before_target_native,
              'mask_differences': diff, 'final_native_cache_bytes': 0,
              'same_mask_after_final_clear': True, 'events': events})
    finally:
        source_release.set()
        target_release.set()
        backend.executor.shutdown(wait=True)
        backend.reset()


assert importlib.metadata.version('xgrammar') == '0.2.6rc2+phala.feedback.20261017.p6'
emit({'phase': 'identity', 'xgrammar': importlib.metadata.version('xgrammar'),
      'synthetic_vocab_size': len(VOCAB), 'real_tokenizer': False,
      'python_workers': 2, 'native_threads': 2, 'native_cache_limit_bytes': 256 << 20})
for case_id, grammar_a, grammar_b, prefix in CASES:
    for order, source, target in [('A_then_B', grammar_a, grammar_b),
                                  ('B_then_A', grammar_b, grammar_a)]:
        run(case_id, source, target, prefix, order)
emit({'phase': 'finished', 'controls': 6})


