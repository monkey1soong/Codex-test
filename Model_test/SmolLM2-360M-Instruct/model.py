"""Colab entry point: !python model.py

Use colab.ipynb once to forward HF_TOKEN and GITHUB_TOKEN from Colab Secrets.
This script installs missing packages, logs in, clones/reuses Codex-test, and
runs SmolLM2 instruction/token tests. No companion Python modules are required.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
GITHUB_REPO = "monkey1soong/Codex-test"
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


class SetupError(RuntimeError):
    """A setup error that can be displayed without exposing credential details."""


def read_secret(name, *aliases):
    for key in (name, *aliases):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    # userdata requires the live notebook kernel. In !python, use inherited env.
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from google.colab import userdata
            value = userdata.get(name)
            if value and value.strip():
                return value.strip()
    except Exception:
        pass
    raise SetupError(
        f"Missing {name}. In colab.ipynb, enable this Colab Secret and run the "
        "preparation cell before !python model.py. Do not paste tokens into commands."
    )


def ensure_dependencies():
    from importlib import metadata
    requirements = {
        "torch": ">=2.2",
        "transformers": ">=4.46,<5",
        "huggingface-hub": ">=0.34,<1",
        "safetensors": ">=0.4",
    }
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "packaging>=23"], check=True)
        from packaging.specifiers import SpecifierSet
    missing = []
    for package, specifier in requirements.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package + specifier)
        else:
            if installed not in SpecifierSet(specifier):
                missing.append(package + specifier)
    if missing:
        print("Installing required packages:", ", ".join(missing), flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)


def github_json(path, token):
    request = urllib.request.Request(
        "https://api.github.com" + path,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "SmolLM2-Colab-Test"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SetupError(
            f"GitHub API HTTP {exc.code}. Check token access to the repository "
            "(Contents: read) and api.github.com network access."
        ) from None
    except (urllib.error.URLError, TimeoutError):
        raise SetupError("Cannot connect to api.github.com. Check runtime network access.") from None


def checkout_repository(repo, branch, destination, token):
    if not shutil.which("git"):
        raise SetupError("Git is unavailable; use a standard Colab runtime.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise SetupError("--github-repo must be owner/repository, not a URL or token.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch):
        raise SetupError("Invalid --github-branch.")
    url = f"https://github.com/{repo}.git"
    # Git reads the token from process memory, never a remote URL or command arg.
    with tempfile.TemporaryDirectory(prefix="smollm-git-") as directory:
        askpass = Path(directory) / "askpass.py"
        askpass.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "print('x-access-token' if 'username' in sys.argv[1].lower() "
            "else os.environ['GITHUB_TOKEN'])\n",
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        git_env = {**os.environ, "GITHUB_TOKEN": token, "GIT_ASKPASS": str(askpass),
                   "GIT_TERMINAL_PROMPT": "0"}

        def git(*arguments):
            result = subprocess.run(
                ["git", "-c", "credential.helper=", *arguments], env=git_env,
                capture_output=True, text=True,
            )
            if result.returncode:
                raise SetupError("Git access failed. Check the token's Contents: read permission and branch name.")
            return result.stdout.strip()

        if destination.exists():
            if not (destination / ".git").exists():
                raise SetupError(f"{destination} exists but is not a Git checkout. Use another --repo-dir.")
            origin = git("-C", str(destination), "remote", "get-url", "origin")
            if origin.rstrip("/").removesuffix(".git") not in (
                url.removesuffix(".git"), f"git@github.com:{repo}"
            ):
                raise SetupError("The existing checkout has a different origin. Use another --repo-dir.")
            # Check remote readability without pulling over an existing checkout.
            git("ls-remote", "--exit-code", url, f"refs/heads/{branch}")
            print(f"Reusing checkout: {destination} (no automatic pull/reset)", flush=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            git("clone", "--depth", "1", "--branch", branch, url, str(destination))
            print(f"Cloned repository: {destination}", flush=True)
        return git("-C", str(destination), "rev-parse", "HEAD")


def prepare_runtime(args):
    hf_token = read_secret("HF_TOKEN")
    github_token = read_secret("GITHUB_TOKEN", "GH_TOKEN")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.github_repo):
        raise SetupError("--github-repo must be owner/repository.")
    if not args.no_install:
        ensure_dependencies()
    from huggingface_hub import login
    from huggingface_hub.utils import logging as hub_logging
    hub_logging.set_verbosity_error()
    try:
        login(token=hf_token, add_to_git_credential=False)
    except Exception:
        raise SetupError("Hugging Face login failed. Check HF_TOKEN read access and huggingface.co connectivity.") from None
    print("Hugging Face login complete.", flush=True)
    user = github_json("/user", github_token)
    repo = github_json(f"/repos/{args.github_repo}", github_token)
    print(f"GitHub login: {user['login']} | repository: {repo['full_name']}", flush=True)
    destination = args.repo_dir
    if destination is None:
        # Reuse a containing checkout; a standalone /content/model.py clones instead.
        destination = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None)
        if destination is None:
            base = Path("/content") if Path("/content").is_dir() else Path.cwd()
            destination = base / args.github_repo.split("/")[1]
    destination = destination.expanduser().resolve()
    commit = checkout_repository(args.github_repo, args.github_branch, destination, github_token)
    return {"user": user["login"], "repository": args.github_repo,
            "branch_checked": args.github_branch, "local_commit": commit,
            "checkout": str(destination)}


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
    parser.add_argument("--github-repo", default=GITHUB_REPO)
    parser.add_argument("--github-branch", default="main")
    parser.add_argument("--repo-dir", type=Path, help="Clone destination or existing checkout")
    parser.add_argument("--no-install", action="store_true", help="Use already installed packages")
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
    parser.add_argument("--local-files-only", action="store_true", help="Use cached weights; login still requires network")
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


def run_model_tests(args, github_info=None):
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
        "github": github_info,
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


def main():
    args = parse_args()
    try:
        github_info = prepare_runtime(args)
    except SetupError as exc:
        raise SystemExit(f"Setup failed: {exc}") from None
    run_model_tests(args, github_info)


if __name__ == "__main__":
    main()
