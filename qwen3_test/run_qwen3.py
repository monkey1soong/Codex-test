"""Run Qwen3 locally, including in a Google Colab GPU runtime.

Colab:
    %pip install -U "transformers>=4.51.0,<5" "torch>=2.2" safetensors
    !python run_qwen3.py --prompt "Transformer의 동작 방식을 설명해줘."

The first run downloads weights from Hugging Face; inference runs locally.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", help="Generate one response instead of starting a chat.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 reasoning mode.")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached weights offline.")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    return args


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=args.local_files_only
    ).to(device).eval()

    def generate(messages):
        # Chat template adds role markers and the assistant generation prefix.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.thinking,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        # The Transformer predicts the next token repeatedly using prior tokens.
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.6 if args.thinking else 0.7,
                top_p=0.95 if args.thinking else 0.8,
                top_k=20,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if args.prompt is not None:
        print(generate([{"role": "user", "content": args.prompt}]))
        return

    messages = []
    print("Type /exit to quit or /clear to reset conversation history.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt == "/exit":
            break
        if prompt == "/clear":
            messages.clear()
            continue
        if not prompt:
            continue
        messages.append({"role": "user", "content": prompt})
        answer = generate(messages)
        print(f"Assistant: {answer}\n")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
