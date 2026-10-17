"""Colab: %pip install torch>=2.2; !python train.py --steps 500
Character-level decoder-only Transformer trained from random weights.
"""
import argparse
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

DEMO = ('안녕하세요. 우리는 작은 Transformer를 처음부터 학습합니다.\n'
        '모델은 앞의 문자를 보고 다음 문자를 예측합니다.\n'
        '어텐션은 문장 안에서 필요한 정보를 찾아 연결합니다.\n'
        '좋은 데이터와 충분한 학습은 더 나은 예측을 만듭니다.\n') * 100


class Attention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        b, t, c = x.shape
        # Queries compare with keys; the resulting weights aggregate values.
        q, k, v = [a.view(b, t, self.heads, c // self.heads).transpose(1, 2)
                   for a in self.qkv(x).chunk(3, dim=-1)]
        # Future tokens must stay hidden during next-token prediction.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = Attention(width, heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.ff(self.norm2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, block_size, width, heads, layers, dropout):
        super().__init__()
        self.block_size = block_size
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(block_size, width)
        self.blocks = nn.Sequential(*[Block(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, tokens, targets=None):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(pos)
        logits = self.head(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(self, tokens, count, temperature):
        self.eval()
        for _ in range(count):
            logits, _ = self(tokens[:, -self.block_size:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            tokens = torch.cat((tokens, torch.multinomial(probs, 1)), dim=1)
        return tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, help='UTF-8 corpus; default: small built-in demo')
    for name, default in [('steps', 500), ('batch-size', 32), ('block-size', 64),
                          ('width', 128), ('heads', 4), ('layers', 3),
                          ('eval-every', 100), ('eval-batches', 10), ('generate-tokens', 200)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt')
    p.add_argument('--output', type=Path, default=Path('transformer.pt'))
    p.add_argument('--load', type=Path, help='Generation only; not training resume')
    a = p.parse_args()
    for name in ('steps', 'batch_size', 'block_size', 'width', 'heads', 'layers',
                 'eval_every', 'eval_batches', 'generate_tokens'):
        if getattr(a, name) < 1:
            p.error(name + ' must be positive')
    if a.width % a.heads or not 0 <= a.dropout < 1:
        p.error('width must be divisible by heads; dropout must be in [0,1)')
    if any(not math.isfinite(v) or v <= 0 for v in (a.lr, a.temperature)):
        p.error('lr and temperature must be finite and positive')
    torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f'Device: {device}')
    if a.load:
        saved = torch.load(a.load, map_location='cpu', weights_only=True)
        chars, config = saved['chars'], saved['config']
    else:
        text = a.data.read_text(encoding='utf-8') if a.data else DEMO
        chars = sorted(set(text))
        if not chars:
            p.error('empty training corpus')
        config = dict(vocab_size=len(chars), block_size=a.block_size,
                      width=a.width, heads=a.heads, layers=a.layers, dropout=a.dropout)
    stoi = {c: i for i, c in enumerate(chars)}
    prompt = a.prompt if a.prompt is not None else ('안녕' if not a.data and not a.load else chars[0])
    unknown = sorted(set(prompt) - set(chars))
    if not prompt or unknown:
        p.error(f'prompt must use corpus characters and be nonempty; unknown={unknown}')
    model = TransformerLM(**config).to(device)
    print(f'Parameters: {sum(t.numel() for t in model.parameters()):,}; vocabulary: {len(chars)}')
    if a.load:
        model.load_state_dict(saved['model'])
    else:
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        split = int(len(data) * 0.9)
        datasets = {'train': data[:split], 'val': data[split:]}
        if min(len(d) for d in datasets.values()) <= a.block_size:
            p.error('each 90/10 corpus split needs at least block-size + 1 characters')

        def batch(part):
            d = datasets[part]
            starts = torch.randint(len(d) - a.block_size, (a.batch_size,))
            x = torch.stack([d[i:i + a.block_size] for i in starts])
            # Each target is exactly one character ahead of its input.
            y = torch.stack([d[i + 1:i + a.block_size + 1] for i in starts])
            return x.to(device), y.to(device)

        @torch.inference_mode()
        def evaluate(step):
            model.eval()
            losses = {part: sum(model(*batch(part))[1].item()
                               for _ in range(a.eval_batches)) / a.eval_batches
                      for part in datasets}
            print(f"step {step}: train={losses['train']:.4f} val={losses['val']:.4f} "
                  f"val_ppl={math.exp(min(losses['val'], 80)):.2f}", flush=True)
            model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
        evaluate(0)
        for step in range(1, a.steps + 1):
            _, loss = model(*batch('train'))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; inspect data and learning rate')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % a.eval_every == 0 or step == a.steps:
                evaluate(step)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': model.state_dict(), 'config': config, 'chars': chars}, a.output)
        print(f'Saved: {a.output}')
    tokens = torch.tensor([[stoi[c] for c in prompt]], device=device)
    output = model.generate(tokens, a.generate_tokens, a.temperature)[0].tolist()
    print('\nGenerated:\n' + ''.join(chars[i] for i in output))


if __name__ == '__main__':
    main()
