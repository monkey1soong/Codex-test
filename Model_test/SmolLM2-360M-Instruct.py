"""Evaluate local SmolLM2 instruction following and token generation in Colab.

    %pip install -r requirements.txt
    !python SmolLM2-360M-Instruct.py
    !python SmolLM2-360M-Instruct.py --compare --show-tokens

Downloads Hugging Face weights on the first run; generation runs on this machine.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's instructions carefully."
CASES = [
    {
        "name": "instruction",
        "prompt": (
            "Classify the sentiment of this review: 'I loved the movie.' "
            "Reply with exactly one lowercase word: positive, negative, or neutral."
        ),
        "check": "positive",
    },
    {
        "name": "arithmetic",
        "prompt": (
            "There are 12 boxes with 7 pencils in each box. "
            "How many pencils are there in total? Reply with only the integer."
        ),
        "check": "84",
    },
    {
        "name": "json",
        "prompt": (
            "Extract the name and age from: 'Alice is 24 years old.' "
            "Return only a JSON object with exactly two keys: name (string) and age (integer). "
            "Do not include markdown or an explanation."
        ),
        "check": "json",
    },
    {
        "name": "summary",
        "prompt": (
            "Summarize the following passage in one sentence of at most 25 words. "
            "Do not add facts. Passage: A school opened a library in March. "
            "Students can borrow two books at a time. The library is closed on Sundays."
        ),
        "check": None,
    },
    {
        "name": "code",
        "prompt": (
            "Write a Python function named count_vowels(text) that counts the English "
            "vowels a, e, i, o, u, ignoring case. Return only the Python code."
        ),
        "check": None,
    },
    {
        "name": "korean",
        "prompt": "인공지능의 토큰이 무엇인지 초보자가 이해하도록 한국어로 두 문장으로 설명해줘.",
        "check": None,
    },
]


def check_response(case, response):
    """Strict checks for three simple tasks; open-ended tasks require human review."""
    expected = case.get("check")
    if expected is None:
        return {"status": "manual_review", "expected": None}
    if expected == "json":
        try:
            obj = json.loads(response)
        except (ValueError, TypeError):
            passed = False
        else:
            passed = (
                isinstance(obj, dict)
                and set(obj) == {"name", "age"}
                and obj["name"] == "Alice"
                and type(obj["age"]) is int
                and obj["age"] == 24
            )
        expected = {"name": "Alice", "age": 24}
    else:
        passed = response.strip() == expected
    return {"status": "pass" if passed else "fail", "expected": expected}


def token_metrics(ids):
    counts = Counter(ids)
    trigrams = [tuple(ids[i:i + 3]) for i in range(max(0, len(ids) - 2))]
    return {
        "unique_token_ratio": len(counts) / len(ids) if ids else None,
        "repeated_trigram_ratio": (
            1 - len(set(trigrams)) / len(trigrams) if trigrams else None
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default="main", help="Model branch or commit SHA")
    parser.add_argument("--prompt", help="Run your own instruction instead of built-in cases")
    parser.add_argument("--case", choices=[c["name"] for c in CASES])
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    parser.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--compare", action="store_true", help="Run both greedy and sampling")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--show-tokens", action="store_true", help="Show first 32 output token pieces")
    parser.add_argument("--output", type=Path, default=Path("results/smollm2_results.json"))
    args = parser.parse_args()
    if args.prompt is not None and args.case is not None:
        parser.error("--prompt and --case cannot be used together")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    return args


def main():
    args = parse_args()
    # Defer heavyweight imports so --help and the scoring helpers work without a model.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Select a Colab GPU runtime or use --device cpu.")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    set_seed(args.seed)
    print(f"Loading {args.model} ({args.revision}) on {device}, dtype={dtype}", flush=True)
    load_started = time.perf_counter()
    options = {"revision": args.revision, "local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, **options
    ).to(device).eval()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    load_seconds = time.perf_counter() - load_started
    if not tokenizer.chat_template:
        raise SystemExit("This test requires an instruction model with a chat template.")
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = next(iter(sorted(eos_ids)), None)
    if pad is None:
        raise SystemExit("The model needs a pad or EOS token for generation.")

    cases = ([{"name": "custom", "prompt": args.prompt, "check": None}]
             if args.prompt is not None else
             [case for case in CASES if args.case is None or case["name"] == args.case])
    modes = ["greedy", "sample"] if args.compare else [args.mode]
    report = {
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "dtype": str(dtype),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "load_seconds": load_seconds,
        "system_prompt": args.system,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": 0},
        "timing_definition": "Synchronized model.generate wall time, including prefill; excludes load, tokenization, decode and warmup.",
        "runs": [],
    }
    warmed_up = False
    for case in cases:
        messages = [{"role": "system", "content": args.system},
                    {"role": "user", "content": case["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(
            text, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt"
        ).to(device)
        input_count = inputs["input_ids"].shape[1]
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit and input_count + args.max_new_tokens > context_limit:
            raise SystemExit(
                f"{case['name']}: input ({input_count}) + output budget ({args.max_new_tokens}) "
                f"exceeds model context ({context_limit}). Shorten the prompt or output budget."
            )
        for mode in modes:
            generation_options = {
                "do_sample": mode == "sample",
                "max_new_tokens": args.max_new_tokens,
                "use_cache": True,
                "pad_token_id": pad,
                "eos_token_id": eos,
                "num_beams": 1,
                "repetition_penalty": 1.0,
                "temperature": args.temperature if mode == "sample" else None,
                "top_p": args.top_p if mode == "sample" else None,
                "top_k": 0 if mode == "sample" else None,
            }
            # Warm up kernels once; keep this work out of generation timing.
            if not warmed_up:
                with torch.inference_mode():
                    model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                   temperature=None, top_p=None, top_k=None,
                                   pad_token_id=pad, eos_token_id=eos)
                synchronize()
                warmed_up = True
            set_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, **generation_options)
            synchronize()
            seconds = time.perf_counter() - started
            # Exclude prompt tokens. EOS is still counted in generated token IDs.
            ids = output[0, input_count:].tolist()
            response = tokenizer.decode(ids, skip_special_tokens=True).strip()
            ended_with_eos = bool(ids and ids[-1] in eos_ids)
            run = {
                "case": case["name"], "mode": mode, "prompt": case["prompt"],
                "response": response, "input_tokens": input_count,
                "generated_tokens": len(ids), "generation_seconds": seconds,
                "tokens_per_second": len(ids) / seconds,
                "ended_with_eos": ended_with_eos,
                "hit_token_limit": len(ids) == args.max_new_tokens and not ended_with_eos,
                "output_token_ids": ids,
                "token_pieces": tokenizer.convert_ids_to_tokens(ids),
                "quality_check": check_response(case, response),
                "peak_cuda_allocated_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == "cuda" else None
                ),
                **token_metrics(ids),
            }
            report["runs"].append(run)
            print(f"\n[{case['name']} / {mode}]\nInstruction: {case['prompt']}\nResponse:\n{response}")
            print(f"tokens={len(ids)} | {seconds:.3f}s | {run['tokens_per_second']:.2f} tokens/s "
                  f"| check={run['quality_check']['status']} | hit_limit={run['hit_token_limit']}", flush=True)
            if args.show_tokens:
                print("First 32 token pieces:", run["token_pieces"][:32])
            # Save after each successful case, retaining earlier results if a later run fails.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nCase             Mode     Tokens  Seconds  Tokens/s  Check")
    for run in report["runs"]:
        print(f"{run['case']:<16} {run['mode']:<8} {run['generated_tokens']:>6} "
              f"{run['generation_seconds']:>8.3f} {run['tokens_per_second']:>9.2f} "
              f"{run['quality_check']['status']}")
    print(f"\nSaved: {args.output}")
    print("These prompts are a small diagnostic set, not a standardized benchmark.")


if __name__ == "__main__":
    main()
