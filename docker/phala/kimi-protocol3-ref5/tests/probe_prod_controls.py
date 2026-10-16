"""Production-sized finite native cache, clear timing, and lookahead controls.

Synthetic CPU evidence only. No product overlays or runtime changes.
"""
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time

import torch
import xgrammar as xgr

from upstream943_cases import CASE_IDS, CROSSING_CACHE_DIFFERENT_FSM_CASES


def emit(row):
    print(json.dumps(row, ensure_ascii=True), flush=True)


def memory():
    return {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in Path('/proc/self/smaps_rollup').read_text().splitlines()
            if line.split(':')[0] in ('Rss', 'Anonymous')}


EXPECTED = {
    'compiler.py': '67e2154127ac14c7f9afde0761755124f2a981f7b0c5a11a1fb96092b6fd109c',
    'libxgrammar_bindings.so': '03b6b083f1a488c7154870bc70b4317eef21ce5d036664cf6893a7c3e91c38eb',
}
package = Path(xgr.__file__).parent
observed = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in EXPECTED}
version = importlib.metadata.version('xgrammar')
assert version == '0.2.6rc2+phala.feedback.20261017.p6' and observed == EXPECTED
assert not Path('/dev/nvidia0').exists()
torch.set_num_threads(1)

VOCAB = ['(', ')', 'a', '0', 'x', 'y', '"', '""', 'xa', 'x0', 'q', 'qx\x01', 'qa', 'q0']
INFO = xgr.TokenizerInfo(VOCAB)
CACHE_LIMIT = 256 << 20
NATIVE_THREADS = 2


def compiler(enabled=True):
    return xgr.GrammarCompiler(INFO, max_threads=NATIVE_THREADS,
                               cache_enabled=enabled, cache_limit_bytes=CACHE_LIMIT)


def matcher(compiled):
    return xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)


def replay(compiled, prefix):
    result = matcher(compiled)
    assert all(result.accept_token(token) for token in prefix), prefix
    assert not result.is_terminated()
    return result


def mask_bits(instance):
    mask = xgr.allocate_token_bitmask(1, INFO.vocab_size)
    xgr.reset_token_bitmask(mask)
    apply = bool(instance.fill_next_token_bitmask(mask))
    bits = [not apply or bool((int(mask[0, token // 32]) >> (token % 32)) & 1)
            for token in range(len(VOCAB))]
    return apply, bits, [int(x) for x in mask[0]]


def direct_accepts(compiled, prefix, token):
    return bool(replay(compiled, prefix).accept_token(token))


def completion(compiled, prefix, token, suffix):
    instance = replay(compiled, prefix)
    token_ok = bool(instance.accept_token(token))
    suffix_ok = bool(instance.accept_string(suffix)) if token_ok else None
    return {'token_accepts': token_ok, 'suffix_accepts': suffix_ok,
            'terminated': bool(instance.is_terminated()) if suffix_ok else False}


def compare(compiled, fresh, prefix, target_token, suffix):
    actual_apply, actual, actual_raw = mask_bits(replay(compiled, prefix))
    fresh_apply, expected, fresh_raw = mask_bits(replay(fresh, prefix))
    differences, fresh_parser_differences, parser_differences = [], [], []
    for token in range(len(VOCAB)):
        parser = direct_accepts(fresh, prefix, token)
        cached_parser = direct_accepts(compiled, prefix, token)
        if actual[token] != expected[token]:
            differences.append({'token_id': token, 'token': VOCAB[token],
                'cached_allows': actual[token], 'fresh_allows': expected[token],
                'cached_parser_accepts': cached_parser, 'fresh_parser_accepts': parser})
        if expected[token] != parser:
            fresh_parser_differences.append({'token_id': token, 'token': VOCAB[token],
                'fresh_allows': expected[token], 'fresh_parser_accepts': parser})
        if cached_parser != parser:
            parser_differences.append({'token_id': token, 'cached': cached_parser, 'fresh': parser})
    return {'prefix_token_ids': prefix, 'prefix_tokens': [VOCAB[x] for x in prefix],
            'cached_apply': actual_apply, 'fresh_apply': fresh_apply,
            'cached_raw_mask': actual_raw, 'fresh_raw_mask': fresh_raw,
            'mask_differences': differences, 'fresh_mask_vs_direct_parser': fresh_parser_differences,
            'direct_parser_differences': parser_differences,
            'target_valid_completion': {'token_id': target_token, 'token': VOCAB[target_token],
                'suffix': suffix, 'cached': completion(compiled, prefix, target_token, suffix),
                'fresh': completion(fresh, prefix, target_token, suffix)}}


emit({'phase': 'identity', 'xgrammar': version, 'package_sha256': observed,
      'matches_p6_wheel': observed == EXPECTED,
      'compiler_signature': str(inspect.signature(xgr.GrammarCompiler)),
      'cache_enabled': True, 'cache_limit_bytes': CACHE_LIMIT, 'native_threads': NATIVE_THREADS,
      'enable_dynamic_compilation': False, 'synthetic_vocab': VOCAB,
      'real_tokenizer': False, 'NVIDIA_VISIBLE_DEVICES': os.environ.get('NVIDIA_VISIBLE_DEVICES'),
      'nvidia_device_present': Path('/dev/nvidia0').exists(), **memory()})
started = time.monotonic()
rows = []
cases = ('lookahead-rule-ref', 'partial-hash-start-state', 'partial-hash-self-reference')
for case_id in cases:
    grammar_a, grammar_b, _ = CROSSING_CACHE_DIFFERENT_FSM_CASES[CASE_IDS.index(case_id)]
    prefix = [0, 2] if case_id == 'partial-hash-self-reference' else [0]
    for order in ('A_then_B', 'B_then_A'):
        source, target = (grammar_a, grammar_b) if order == 'A_then_B' else (grammar_b, grammar_a)
        valid_token = (13 if order == 'A_then_B' else 12) if case_id != 'partial-hash-self-reference' else 13
        # For self-reference the prefix already consumes the A/B distinguishing byte.
        # Both target continuations are q0/qA after (a; q0 is valid only for B.
        if case_id == 'partial-hash-self-reference' and order == 'B_then_A':
            valid_token = 12
        suffix = ')' if case_id == 'lookahead-rule-ref' else ('])' if case_id == 'partial-hash-start-state' else 'b)')
        fresh = compiler(False).compile_grammar(target)

        unseeded = compiler()
        clean = unseeded.compile_grammar(target)
        row = {'phase': 'production_cache_control', 'case': case_id, 'order': order,
               'mode': 'unseeded', 'native_cache_bytes': unseeded.get_cache_size_bytes(),
               **compare(clean, fresh, prefix, valid_token, suffix)}
        rows.append(row)
        emit(row)
        unseeded.clear_cache()

        retained = compiler()
        retained.compile_grammar(source)
        seed_bytes = retained.get_cache_size_bytes()
        cached = retained.compile_grammar(target)
        target_bytes = retained.get_cache_size_bytes()
        row = {'phase': 'production_cache_control', 'case': case_id, 'order': order,
               'mode': 'seeded_no_clear', 'seed_native_cache_bytes': seed_bytes,
               'native_cache_bytes': target_bytes, **compare(cached, fresh, prefix, valid_token, suffix)}
        rows.append(row)
        emit(row)
        retained.clear_cache()
        row = {'phase': 'production_cache_control', 'case': case_id, 'order': order,
               'mode': 'same_compiled_after_batch_clear', 'native_cache_bytes': retained.get_cache_size_bytes(),
               **compare(cached, fresh, prefix, valid_token, suffix)}
        rows.append(row)
        emit(row)
        recompiled = retained.compile_grammar(target)
        row = {'phase': 'production_cache_control', 'case': case_id, 'order': order,
               'mode': 'recompiled_after_batch_clear', 'native_cache_bytes': retained.get_cache_size_bytes(),
               **compare(recompiled, fresh, prefix, valid_token, suffix)}
        rows.append(row)
        emit(row)
        retained.clear_cache()

        cleared = compiler()
        cleared.compile_grammar(source)
        seed_bytes = cleared.get_cache_size_bytes()
        cleared.clear_cache()
        after_clear_bytes = cleared.get_cache_size_bytes()
        new_target = cleared.compile_grammar(target)
        row = {'phase': 'production_cache_control', 'case': case_id, 'order': order,
               'mode': 'seed_then_clear', 'seed_native_cache_bytes': seed_bytes,
               'after_clear_cache_bytes': after_clear_bytes,
               'native_cache_bytes': cleared.get_cache_size_bytes(),
               **compare(new_target, fresh, prefix, valid_token, suffix)}
        rows.append(row)
        emit(row)
        cleared.clear_cache()

# Direct parser acceptance omits compile-time lookahead checks. These two rows
# explain the four duplicated exact1 observations; they are not cache defects.
grammar_a, grammar_b, _ = CROSSING_CACHE_DIFFERENT_FSM_CASES[CASE_IDS.index('lookahead')]
lookahead = []
for target_label, grammar, invalid_token in [('A', grammar_a, 8), ('B', grammar_b, 9)]:
    compiled = compiler(False).compile_grammar(grammar)
    apply, bits, raw = mask_bits(replay(compiled, [0]))
    row = {'phase': 'lookahead_direct_parser_control', 'target': target_label,
           'prefix_token_ids': [0], 'token_id': invalid_token, 'token': VOCAB[invalid_token],
           'mask_apply': apply, 'mask_allows': bits[invalid_token], 'raw_mask': raw,
           'direct_parser_completion': completion(compiled, [0], invalid_token, ')'),
           'expected_lookahead_range': '[0-3]' if target_label == 'A' else '[1-t]'}
    lookahead.append(row)
    emit(row)

by_mode = {}
for row in rows:
    summary = by_mode.setdefault(row['mode'], {'rows': 0, 'rows_with_mask_difference': 0,
        'mask_difference_tokens': 0, 'direct_parser_difference_tokens': 0,
        'fresh_mask_vs_direct_parser_tokens': 0})
    summary['rows'] += 1
    summary['rows_with_mask_difference'] += bool(row['mask_differences'])
    summary['mask_difference_tokens'] += len(row['mask_differences'])
    summary['direct_parser_difference_tokens'] += len(row['direct_parser_differences'])
    summary['fresh_mask_vs_direct_parser_tokens'] += len(row['fresh_mask_vs_direct_parser'])
emit({'phase': 'summary', 'production_cache_controls': len(rows), 'by_mode': by_mode,
      'lookahead_direct_parser_controls': len(lookahead), 'seconds': time.monotonic() - started,
      **memory()})
