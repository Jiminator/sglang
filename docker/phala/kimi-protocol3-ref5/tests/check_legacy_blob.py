"""Reject an actually polluted p5 blob and validate a fresh p6 compiled roundtrip."""
import hashlib
import json
from pathlib import Path

import torch
import xgrammar as xgr
from upstream943_cases import CROSSING_CACHE_DIFFERENT_FSM_CASES

vocab = ['(', ')', 'a', '0', 'x', 'y', '"', '""', 'xa', 'x0', 'q', 'qx\x01', 'qa', 'q0']
info = xgr.TokenizerInfo(vocab)
raw = (Path(__file__).parent / 'legacy-p5-compiled.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'c51b60edc94aa4ff36b5502cbdd988bcc369277bcf75bfcfa7bd789eaab9c3a1'
try:
    xgr.CompiledGrammar.deserialize_json(raw.decode(), info)
except xgr.DeserializeVersionError as error:
    assert 'v21-phala-unordered-constraints-v2' in str(error)
    assert 'v21-phala-unordered-constraints-v3' in str(error)
    print(json.dumps({'phase': 'real_p5_blob_rejected', 'exception': type(error).__name__,
                      'sha256': hashlib.sha256(raw).hexdigest(), 'error': str(error)}), flush=True)
else:
    raise AssertionError('p6 accepted an actual p5 compiled grammar with a wrong stored mask')

grammar_a, _, _ = CROSSING_CACHE_DIFFERENT_FSM_CASES[8]
compiled = xgr.GrammarCompiler(info, max_threads=2, cache_limit_bytes=256 << 20).compile_grammar(grammar_a)
serialized = compiled.serialize_json()
assert json.loads(serialized)['__VERSION__'] == 'v21-phala-unordered-constraints-v3'
recovered = xgr.CompiledGrammar.deserialize_json(serialized, info)
masks = []
for candidate in (compiled, recovered):
    instance = xgr.GrammarMatcher(candidate, terminate_without_stop_token=True)
    assert instance.accept_token(0)
    mask = xgr.allocate_token_bitmask(1, len(vocab))
    xgr.reset_token_bitmask(mask)
    assert instance.fill_next_token_bitmask(mask)
    assert not bool((int(mask[0, 0]) >> 13) & 1)
    assert instance.accept_token(12) and instance.accept_string('])') and instance.is_terminated()
    masks.append(mask.clone())
torch.testing.assert_close(masks[0], masks[1], rtol=0, atol=0)
assert serialized == recovered.serialize_json()
print(json.dumps({'phase': 'p6_roundtrip', 'new_blob_version': 'v21-phala-unordered-constraints-v3',
                  'mask_preserved': True, 'invalid_q0_rejected': True,
                  'valid_qa_completed': True}), flush=True)
