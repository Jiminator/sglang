"""Unmodified synthetic cases and regression body from XGrammar PR #943.
Helper uses the final upstream mixed string/token replay. No model or real data.
"""
from typing import List, Union
import torch
import xgrammar as xgr

CROSSING_CACHE_DIFFERENT_FSM_CASES = [
    (
        'root ::= "(" r ")"\nr ::= [0-3] "x" | "y"',
        'root ::= "(" r ")"\nr ::= [1-t] "x" | "y"',
        "(ax)",
    ),
    (
        'root ::= "(" s ")"\ns ::= "\\"" [^"\\\\\\r\\n]{0,64} "\\""',
        'root ::= "(" s ")"\ns ::= "\\"" [^"\\\\\\r\\n]{1,129} "\\""',
        '("a")',
    ),
    (
        'root ::= "(" r [0-9a-z] ")" | "(" r "y" ")"\nr ::= ("x" | "yy") (=[0-3])',
        'root ::= "(" r [0-9a-z] ")" | "(" r "y" ")"\nr ::= ("x" | "yy") (=[1-t])',
        "(xa)",
    ),
    (
        'root ::= "(" r ")"\nr ::= s | "x" s | "x0y"\ns ::= "a" | "yy" | "a" s',
        'root ::= "(" r ")"\nr ::= s | "x0" s | "x0y"\ns ::= "a" | "yy" | "a" s',
        "(x0a)",
    ),
    (
        'root ::= "(" r ")"\nr ::= "a" Token(4) | "0"',
        'root ::= "(" r ")"\nr ::= "a" Token(5) | "0"',
        ["(", "a", 5, ")"],
    ),
    (
        'root ::= "(" r "x" [c-d] ")"\nr ::= "q" (="x" [c-d])',
        'root ::= "(" r "x\\x01\\x00\\x63\\x64" ")"\nr ::= "q" (="x\\x01\\x00\\x63\\x64")',
        ["(", 11, "\x00", "c", "d", ")"],
    ),
    (
        'root ::= "(" r z ")"\nr ::= "q" (=z)\n'
        'v ::= "[" w "]" | "a"\nw ::= v | "{" z "}"\nz ::= w | "b"',
        'root ::= "(" r z ")"\nr ::= "q" (=z)\n'
        'v ::= "[" w "]" | "0"\nw ::= v | "{" z "}"\nz ::= w | "b"',
        "(q0)",
    ),
    ('root ::= "(" r ")"\nr ::= [a-x]* "x"', 'root ::= "(" r ")"\nr[lazy] ::= [a-x]* "x"', "(ax)"),
    (
        'root ::= "(" r ")"\nr ::= z | "qx"\n'
        'v ::= "q" w "]" | "a"\nw ::= v | "{" z "}"\nz ::= w | "b"',
        'root ::= "(" r ")"\nr ::= z | "qx"\n'
        'v ::= "q" w "]" | "0"\nw ::= v | "{" z "}"\nz ::= w | "b"',
        "(q0])",
    ),
    (
        'root ::= "(" r ")"\nr ::= x "b" | "aq" r | "az"\nx ::= r "c" | y\ny ::= x "d" | "a"',
        'root ::= "(" r ")"\nr ::= x "b" | "aq" r | "az"\nx ::= r "c" | y\ny ::= x "d" | "0"',
        "(aq0b)",
    ),
]

CASE_IDS = ['character-range', 'repeat-bounds', 'lookahead', 'rule-edge-source', 'token-edge', 'lookahead-expr-boundary', 'lookahead-rule-ref', 'lazy', 'partial-hash-start-state', 'partial-hash-self-reference']

def _mask_trace(compiled_grammar, input_str):
    matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, compiled_grammar.tokenizer_info.vocab_size)
    trace = []
    for char in input_str:
        xgr.reset_token_bitmask(bitmask)
        need_apply = matcher.fill_next_token_bitmask(bitmask)
        trace.append((need_apply, bitmask.clone()))
        if isinstance(char, int):
            assert matcher.accept_token(char)
        else:
            assert matcher.accept_string(char)
    assert matcher.is_terminated()
    return trace

def test_grammar_compiler_crossing_cache_different_fsm(
    grammar_a: str, grammar_b: str, input_str: Union[str, List[Union[str, int]]]
):
    """A grammar must not reuse the token masks of another grammar whose rule differs only in
    small integers (character ranges, repetition bounds), in the state a rule reference leaves,
    in the tokens of a token edge, or in where one lookahead element ends: their FSM hashes used to
    collide. Nor may it reuse them for a lookahead that references a rule with a partial hash, which
    leaves out a rule referenced at its start state, or for a rule that differs only in being lazy.
    The start state of a rule with a partial hash must not reuse them either: its mask includes the
    tokens of that left-out rule, and so do the masks of every state of such a rule that references
    itself."""
    tokenizer_info = xgr.TokenizerInfo(
        ["(", ")", "a", "0", "x", "y", '"', '""', "xa", "x0", "q", "qx\x01", "qa", "q0"]
    )
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1, cache_enabled=True)
    compiler.compile_grammar(grammar_a)
    cached = compiler.compile_grammar(grammar_b)
    fresh = xgr.GrammarCompiler(tokenizer_info, max_threads=1, cache_enabled=False).compile_grammar(
        grammar_b
    )

    expected_trace = _mask_trace(fresh, input_str)
    actual_trace = _mask_trace(cached, input_str)
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(
        expected_trace, actual_trace
    ):
        assert actual_apply == expected_apply
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)
