"""End-to-end benchmark: Grammar-Aware Speculative Decoding with vLLM.

Compares baseline ngram speculative decoding vs grammar-aware filtering
on JSON-schema-constrained generation.

Models:
  Target: Qwen/Qwen2.5-1.5B (bf16)
  Draft:  ngram (built-in)

Measures:
  1. Output validity (100% valid JSON required)
  2. Time per output token (TPOT)
  3. Throughput (tokens/sec)
  4. Grammar filter rate (from GrammarAwareProposer stats)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json"
)

# ---------------------------------------------------------------------------
# JSON schema for weather reporting (realistic tool-calling shape)
# ---------------------------------------------------------------------------
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "temperature_celsius": {"type": "number"},
        "conditions": {
            "type": "string",
            "enum": ["sunny", "cloudy", "rainy", "snowy", "stormy"],
        },
        "humidity_percent": {"type": "integer"},
        "wind_speed_kmh": {"type": "number"},
    },
    "required": ["city", "temperature_celsius", "conditions"],
}

PROMPTS = [
    "What is the weather in Tokyo?",
    "Tell me the current weather in London.",
    "Give me weather info for New York City.",
    "How is the weather in Mumbai right now?",
    "Weather report for Paris please.",
    "What's it like outside in Sydney?",
    "Current conditions in Berlin?",
    "Weather data for San Francisco.",
]


def run_baseline(target_model: str, num_steps: int = 20) -> dict:
    """Run baseline: ngram spec decode + JSON schema, no grammar-aware filtering."""
    print("=" * 70)
    print("BASELINE: ngram spec decode + JSON schema (no grammar-aware filtering)")
    print("=" * 70)

    from vllm import LLM, SamplingParams
    from vllm.config import SpeculativeConfig, VllmConfig
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=target_model,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 1,
        },
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        structured_outputs=StructuredOutputsParams(json=WEATHER_SCHEMA),
    )

    # Run inference
    start = time.perf_counter()
    outputs = llm.generate(PROMPTS, sampling_params)
    elapsed = time.perf_counter() - start

    # Measure validity
    valid_count = 0
    invalid_outputs = []
    total_tokens = 0
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        total_tokens += len(out.outputs[0].token_ids)
        try:
            parsed = json.loads(text)
            # Check required fields
            if all(k in parsed for k in WEATHER_SCHEMA["required"]):
                valid_count += 1
            else:
                invalid_outputs.append((i, "missing required fields", text[:100]))
        except json.JSONDecodeError as e:
            invalid_outputs.append((i, str(e), text[:100]))

    result = {
        "name": "baseline_ngram",
        "valid": valid_count,
        "total": len(outputs),
        "validity_rate": valid_count / len(outputs),
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "tpot_ms": (elapsed / total_tokens * 1000) if total_tokens > 0 else 0,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
        "invalid_outputs": invalid_outputs,
    }

    print(f"  Valid outputs: {valid_count}/{len(outputs)} ({result['validity_rate']:.1%})")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  TPOT: {result['tpot_ms']:.1f} ms")
    print(f"  Throughput: {result['tokens_per_sec']:.1f} tok/s")
    if invalid_outputs:
        print(f"  Invalid outputs: {len(invalid_outputs)}")
        for idx, reason, snippet in invalid_outputs[:3]:
            print(f"    [{idx}] {reason}: {snippet}")
    print()

    return result


def run_grammar_aware(target_model: str, num_steps: int = 20) -> dict:
    """Run grammar-aware: ngram spec decode with GrammarAwareProposer wrapper."""
    print("=" * 70)
    print("GRAMMAR-AWARE: ngram + GrammarAwareProposer + JSON schema")
    print("=" * 70)

    import json as _json
    import os
    import tempfile

    import xgrammar as xgr
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer

    from grammar_aware_proposer import GrammarAwareProposer

    # Write schema + tokenizer name to temp files for the subprocess
    schema_path = tempfile.mktemp(suffix=".json")
    with open(schema_path, "w") as f:
        _json.dump(WEATHER_SCHEMA, f)

    # Stats file for cross-process communication
    stats_path = tempfile.mktemp(suffix=".json")

    # Patch NgramProposer at class level so it survives spawn re-import
    original_init = NgramProposer.__init__
    original_propose = NgramProposer.propose

    def patched_init(self, vllm_config):
        original_init(self, vllm_config)
        wrapped = GrammarAwareProposer(
            self,
            num_speculative_tokens=vllm_config.speculative_config.num_speculative_tokens,
            vocab_size=vllm_config.model_config.get_vocab_size(),
        )
        # Pass the ORIGINAL unbound propose so wrapped._original_propose
        # doesn't resolve through the patched class.
        wrapped._original_propose = lambda *a, **kw: original_propose(self, *a, **kw)
        wrapped._schema_path = schema_path
        wrapped._tokenizer_name = target_model
        wrapped._stats_path = stats_path
        wrapped._compiled_grammar_cache = None
        self._grammar_aware = wrapped
        print(f"  [ENGINE] GrammarAwareProposer created, stats_path={stats_path}", file=sys.stderr)
        sys.stderr.flush()

    def _get_compiled_grammar(wrapped):
        """Compile grammar inside the subprocess (spawn-safe)."""
        if wrapped._compiled_grammar_cache is None:
            tok = AutoTokenizer.from_pretrained(wrapped._tokenizer_name)
            tok_info = xgr.TokenizerInfo.from_huggingface(tok)
            compiler = xgr.GrammarCompiler(tok_info)
            with open(wrapped._schema_path) as f:
                schema = _json.load(f)
            grammar = xgr.Grammar.from_json_schema(schema)
            wrapped._compiled_grammar_cache = compiler.compile_grammar(grammar)
            wrapped._tokenizer_info = tok_info
            print(f"  [ENGINE] Grammar compiled in subprocess, vocab={tok_info.vocab_size}", file=sys.stderr)
            sys.stderr.flush()
        return wrapped._compiled_grammar_cache

    def patched_propose(
        self,
        sampled_token_ids,
        req_ids,
        num_tokens_no_spec,
        token_ids_cpu,
        spec_decode_unsupported_reqs,
    ):
        wrapped = getattr(self, "_grammar_aware", None)
        if wrapped is None:
            return original_propose(
                self,
                sampled_token_ids,
                req_ids,
                num_tokens_no_spec,
                token_ids_cpu,
                spec_decode_unsupported_reqs,
            )
        # Compile grammar if not yet done
        compiled = _get_compiled_grammar(wrapped)
        # Attach grammar to any new requests (pass tokenizer_info for correct bitmask size)
        tok_info = getattr(wrapped, "_tokenizer_info", None)
        for req_id in req_ids:
            if req_id not in wrapped._matchers:
                wrapped.set_grammar(req_id, compiled, tokenizer_info=tok_info)
        return wrapped.propose(
            sampled_token_ids,
            req_ids,
            num_tokens_no_spec,
            token_ids_cpu,
            spec_decode_unsupported_reqs,
        )

    NgramProposer.__init__ = patched_init
    NgramProposer.propose = patched_propose

    try:
        llm = LLM(
            model=target_model,
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 5,
                "prompt_lookup_max": 3,
                "prompt_lookup_min": 1,
            },
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=False,
            disable_log_stats=False,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            structured_outputs=StructuredOutputsParams(json=WEATHER_SCHEMA),
        )

        start = time.perf_counter()
        outputs = llm.generate(PROMPTS, sampling_params)
        elapsed = time.perf_counter() - start

        # Restore original methods
        NgramProposer.__init__ = original_init
        NgramProposer.propose = original_propose

        # Measure validity
        valid_count = 0
        invalid_outputs = []
        total_tokens = 0
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            total_tokens += len(out.outputs[0].token_ids)
            try:
                parsed = json.loads(text)
                if all(k in parsed for k in WEATHER_SCHEMA["required"]):
                    valid_count += 1
                else:
                    invalid_outputs.append((i, "missing required fields", text[:100]))
            except json.JSONDecodeError as e:
                invalid_outputs.append((i, str(e), text[:100]))

        # Try to read stats from the engine subprocess
        filter_stats = {}
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                filter_stats = json.load(f)
            print(f"  [STATS] Loaded from {stats_path}: {filter_stats}", file=sys.stderr)
        else:
            print(f"  [STATS] No stats file at {stats_path}", file=sys.stderr)
        sys.stderr.flush()

        result = {
            "name": "grammar_aware",
            "valid": valid_count,
            "total": len(outputs),
            "validity_rate": valid_count / len(outputs),
            "total_tokens": total_tokens,
            "elapsed_s": elapsed,
            "tpot_ms": (elapsed / total_tokens * 1000) if total_tokens > 0 else 0,
            "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
            "invalid_outputs": invalid_outputs,
            "filter_stats": filter_stats,
        }

        print(f"  Valid outputs: {valid_count}/{len(outputs)} ({result['validity_rate']:.1%})")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  TPOT: {result['tpot_ms']:.1f} ms")
        print(f"  Throughput: {result['tokens_per_sec']:.1f} tok/s")
        if filter_stats:
            print(f"  Grammar filter rate: {filter_stats.get('filter_rate', 0):.1%}")
            print(f"  Drafts proposed: {filter_stats.get('total_drafts_proposed', 0)}")
            print(f"  Drafts filtered: {filter_stats.get('total_drafts_filtered', 0)}")
        print()

        return result

    finally:
        # Cleanup temp files
        for p in [schema_path, stats_path]:
            if os.path.exists(p):
                os.unlink(p)


def run_no_spec(target_model: str) -> dict:
    """Run without speculative decoding (pure baseline)."""
    print("=" * 70)
    print("NO SPEC DECODE: pure autoregressive + JSON schema")
    print("=" * 70)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=target_model,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        structured_outputs=StructuredOutputsParams(json=WEATHER_SCHEMA),
    )

    start = time.perf_counter()
    outputs = llm.generate(PROMPTS, sampling_params)
    elapsed = time.perf_counter() - start

    valid_count = 0
    total_tokens = 0
    for out in outputs:
        text = out.outputs[0].text
        total_tokens += len(out.outputs[0].token_ids)
        try:
            parsed = json.loads(text)
            if all(k in parsed for k in WEATHER_SCHEMA["required"]):
                valid_count += 1
        except json.JSONDecodeError:
            pass

    result = {
        "name": "no_spec",
        "valid": valid_count,
        "total": len(outputs),
        "validity_rate": valid_count / len(outputs),
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "tpot_ms": (elapsed / total_tokens * 1000) if total_tokens > 0 else 0,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
    }

    print(f"  Valid outputs: {valid_count}/{len(outputs)} ({result['validity_rate']:.1%})")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  TPOT: {result['tpot_ms']:.1f} ms")
    print(f"  Throughput: {result['tokens_per_sec']:.1f} tok/s")
    print()

    return result


if __name__ == "__main__":
    target = "Qwen/Qwen2.5-1.5B"

    print()
    print("Grammar-Aware Speculative Decoding — End-to-End Benchmark")
    print(f"Target model: {target}")
    print(f"Draft method: ngram (k=5)")
    print(f"Schema: weather report (5 fields, 3 required)")
    print(f"Prompts: {len(PROMPTS)}")
    print()

    results = []

    # Run all three configurations
    results.append(run_no_spec(target))
    results.append(run_baseline(target))
    results.append(run_grammar_aware(target))

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25} {'Valid':<10} {'TPOT (ms)':<12} {'Tok/s':<10} {'Speedup':<10}")
    print("-" * 70)

    baseline = next(r for r in results if r["name"] == "baseline_ngram")
    for r in results:
        speedup = baseline["tpot_ms"] / r["tpot_ms"] if r["tpot_ms"] > 0 else 0
        print(
            f"{r['name']:<25} {r['valid']}/{r['total']:<6} "
            f"{r['tpot_ms']:<12.1f} {r['tokens_per_sec']:<10.1f} {speedup:<10.2f}x"
        )

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print(f"Results saved to {RESULTS_PATH}")
