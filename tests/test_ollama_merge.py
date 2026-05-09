"""Ollama thinking vs response merge (DeepSeek R1 and similar)."""

from __future__ import annotations

from cot_knob.llm.ollama_client import _merge_generate_output


def test_merge_both_thinking_and_response():
    assert _merge_generate_output({"thinking": "trace", "response": "ANSWER: d3"}) == (
        "trace\n\nANSWER: d3"
    )


def test_merge_response_only():
    assert _merge_generate_output({"response": "hello"}) == "hello"


def test_merge_thinking_only():
    assert _merge_generate_output({"thinking": "only think"}) == "only think"


def test_merge_empty():
    assert _merge_generate_output({}) == ""
