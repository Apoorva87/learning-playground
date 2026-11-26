"""Train the toy looped transformer used by Article A, and export weights and test vectors.

Task: pointer chasing. The input lists 8 (src, dst) pairs (each pair is embedded as one position) of a random map f on 8 nodes (no node
maps to itself), then a query node q. After pass t the model must read out f^t(q), so each pass
advances the pointer one hop. Loss is summed over passes 1..4; passes 5 and 6 are never trained
and are reported as extrapolation.

Two variants share the layers and the task:
  carry    s_t = R(e, s_{t-1}): pass t starts from pass t-1's final state, plus the prelude
           output e (input injection), as in Huginn-style recurrent depth.
  inkling  every pass restarts from the prelude output e. For each core layer, a message is built
           from the previous pass's layer outputs by attention with a learned, zero-initialised
           query, and enters through the Inkling gate: (1 - f) * h + g * m, with
           f = tanh(W_f z), g = tanh(W_m z), z = W_down [rms(h); rms(m)].
Both are simplified on purpose: plain residuals (no attention residuals), a dense ReLU MLP (no
experts), no short convolutions, 1 prelude layer and 2 core layers, and no coda.

Everything is float64, so the JavaScript port in toy/toy.js can match it to better than 1e-5.

Usage (from the playground root):
  .local/inkling2/venv/bin/python topics/looped-transformers/assets/toy/train_toy.py
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
N = 8                      # nodes
T = 2 * N + 1              # input tokens: 8 (src, dst) pairs, then the query
P = N + 1                  # positions after merging each pair into one: 8 pairs + the query
D, H, FF, RANK, EPS = 16, 2, 32, 4, 1e-6
HD = D // H
TRAIN_PASSES, EVAL_PASSES = 4, 6
OUT = Path(__file__).parent


def rms(x, g=None):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y if g is None else y * g


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(D, D, bias=False) for _ in range(4))
        self.w1, self.w2 = nn.Linear(D, FF, bias=False), nn.Linear(FF, D, bias=False)
        self.g1, self.g2 = nn.Parameter(torch.ones(D)), nn.Parameter(torch.ones(D))

    def forward(self, h):
        B = h.shape[0]
        x = rms(h, self.g1)
        q, k, v = (w(x).view(B, P, H, HD).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HD), dim=-1)          # [B,H,T,T]
        h = h + self.wo((att @ v).transpose(1, 2).reshape(B, P, D))
        h = h + self.w2(torch.relu(self.w1(rms(h, self.g2))))
        return h, att


class Gate(nn.Module):
    """Message gate for one core layer (Inkling variant)."""

    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(D))                       # zero-initialised, as in Inkling
        self.down = nn.Linear(2 * D, RANK, bias=False)
        self.wf, self.wm = nn.Linear(RANK, D, bias=False), nn.Linear(RANK, D, bias=False)
        nn.init.zeros_(self.wf.weight)
        nn.init.zeros_(self.wm.weight)                                  # starts closed

    def forward(self, h, sources):
        S = torch.stack([rms(s) for s in sources], dim=-2)              # [B,T,nSources,D]
        w = torch.softmax((S * self.query).sum(-1) / math.sqrt(D), dim=-1)
        m = (w.unsqueeze(-1) * S).sum(-2)
        z = self.down(torch.cat([rms(h), rms(m)], dim=-1))
        f, g = torch.tanh(self.wf(z)), torch.tanh(self.wm(z))
        return (1 - f) * h + g * m, w, f, g


class Toy(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.tok = nn.Embedding(N, D)                                   # one table for every node
        self.wsrc, self.wdst, self.wq = (nn.Linear(D, D, bias=False) for _ in range(3))   # role of the node: source, destination, query
        self.prelude = Layer()
        self.core = nn.ModuleList([Layer(), Layer()])
        self.gates = nn.ModuleList([Gate(), Gate()]) if variant == 'inkling' else None
        self.gout, self.out = nn.Parameter(torch.ones(D)), nn.Linear(D, N, bias=False)

    def forward(self, tokens, passes, trace=False):
        B = tokens.shape[0]
        x = self.tok(tokens)                                            # [B, T, D]
        pairs = self.wsrc(x[:, 0:2 * N:2]) + self.wdst(x[:, 1:2 * N:2])  # one position per (src, dst) pair
        e, _ = self.prelude(torch.cat([pairs, self.wq(x[:, -1:])], dim=1))
        logits, tr, prev = [], [], None
        for t in range(1, passes + 1):
            h = e if (self.variant == 'inkling' or t == 1) else prev[-1] + e
            outs, att_rows, fs, gs, ws = [], [], [], [], []
            for j, layer in enumerate(self.core):
                if self.variant == 'inkling' and t > 1:
                    h, w, f, g = self.gates[j](h, prev)
                    fs.append(f); gs.append(g); ws.append(w)
                h, att = layer(h)
                outs.append(h); att_rows.append(att)
            prev = outs
            logits.append(self.out(rms(h[:, -1], self.gout)))
            if trace:
                tr.append(dict(
                    hidden_last=h[:, -1],
                    attn=torch.stack([a[:, hh, -1] for a in att_rows for hh in range(H)], dim=1),      # [B, layers*H, T]
                    f=torch.stack([x[:, -1] for x in fs], 1) if fs else None,                          # [B, layers, D]
                    g=torch.stack([x[:, -1] for x in gs], 1) if gs else None,
                    msg_w=torch.stack([x[:, -1] for x in ws], 1) if ws else None))                     # [B, layers, nSources]
        return (logits, tr) if trace else logits


def make_batch(B, gen):
    shift = torch.randint(1, N, (B, N), generator=gen)
    f = (torch.arange(N) + shift) % N                                   # f[i] != i
    order = torch.argsort(torch.rand(B, N, generator=gen), dim=1)
    src, dst = order, f.gather(1, order)
    q = torch.randint(0, N, (B,), generator=gen)
    tokens = torch.cat([torch.stack([src, dst], 2).reshape(B, 2 * N), q[:, None]], 1)
    targets, cur = [], q
    for _ in range(EVAL_PASSES):
        cur = f.gather(1, cur[:, None])[:, 0]
        targets.append(cur)
    return tokens, torch.stack(targets, 1)                              # targets[:, t-1] = f^t(q)


def accuracy(model, n=2000, seed=777):
    tokens, targets = make_batch(n, torch.Generator().manual_seed(seed))
    with torch.no_grad():
        logits = model(tokens, EVAL_PASSES)
    return [float((logits[t].argmax(-1) == targets[:, t]).double().mean()) for t in range(EVAL_PASSES)]


def hop_matrix(model, n=2000, seed=777):
    """matrix[p-1][k-1]: share of held-out sequences where the readout after p passes is the k-th hop f^k(q)."""
    tokens, targets = make_batch(n, torch.Generator().manual_seed(seed))
    with torch.no_grad():
        logits = model(tokens, EVAL_PASSES)
    return [[float((logits[p].argmax(-1) == targets[:, k]).double().mean()) for k in range(EVAL_PASSES)] for p in range(EVAL_PASSES)]


def eval_subset(model, n=200, seed=777):
    """The first n held-out sequences, saved so the JavaScript port can reproduce the accuracy exactly."""
    tokens, targets = make_batch(2000, torch.Generator().manual_seed(seed))
    tokens, targets = tokens[:n], targets[:n]
    with torch.no_grad():
        logits = model(tokens, EVAL_PASSES)
    acc = [float((logits[t].argmax(-1) == targets[:, t]).double().mean()) for t in range(EVAL_PASSES)]
    return tokens.tolist(), targets.tolist(), acc


def train(variant, steps, seed):
    torch.manual_seed(seed)
    model = Toy(variant)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    for step in range(1, steps + 1):
        tokens, targets = make_batch(256, gen)
        logits = model(tokens, TRAIN_PASSES)
        loss = sum(F.cross_entropy(logits[t], targets[:, t]) for t in range(TRAIN_PASSES))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == steps:
            acc = accuracy(model, 500)
            print(f'{variant:8s} step {step:5d} loss {loss.item():.4f}  acc by pass {[round(a, 2) for a in acc]}  {time.time() - t0:.0f}s', flush=True)
    return model


def tolist(t):
    return t.detach().tolist()


def export(model):
    layer = lambda m: {k: tolist(getattr(m, k).weight) for k in ('wq', 'wk', 'wv', 'wo', 'w1', 'w2')} | {'g1': tolist(m.g1), 'g2': tolist(m.g2)}
    w = {'tok': tolist(model.tok.weight), 'wsrc': tolist(model.wsrc.weight), 'wdst': tolist(model.wdst.weight), 'wq': tolist(model.wq.weight), 'prelude': layer(model.prelude),
         'core': [layer(m) for m in model.core], 'gout': tolist(model.gout), 'out': tolist(model.out.weight)}
    if model.gates is not None:
        w['gates'] = [{'query': tolist(g.query), 'down': tolist(g.down.weight), 'wf': tolist(g.wf.weight), 'wm': tolist(g.wm.weight)} for g in model.gates]
    return w


def vectors(model):
    tokens, targets = make_batch(5, torch.Generator().manual_seed(123))
    with torch.no_grad():
        logits, tr = model(tokens, EVAL_PASSES, trace=True)
    out = []
    for b in range(5):
        row = {'tokens': tokens[b].tolist(), 'targets': targets[b].tolist(), 'passes': []}
        for t in range(EVAL_PASSES):
            p = {'logits': tolist(logits[t][b]), 'hidden_last': tolist(tr[t]['hidden_last'][b]), 'attn': tolist(tr[t]['attn'][b])}
            for k in ('f', 'g', 'msg_w'):
                p[k] = tolist(tr[t][k][b]) if tr[t][k] is not None else None
            row['passes'].append(p)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variants', default='carry,inkling')
    args = ap.parse_args()
    weights, vecs, acc, ev = {}, {}, {}, {}
    for variant in args.variants.split(','):
        model = train(variant, args.steps, args.seed)
        a = accuracy(model)
        weights[variant], vecs[variant] = export(model), vectors(model)
        tk, tg, sub = eval_subset(model)
        ev['tokens'], ev['targets'] = tk, tg
        ev[variant] = sub
        acc[variant] = {'passes': list(range(1, EVAL_PASSES + 1)), 'accuracy': a, 'trained_passes': TRAIN_PASSES,
                        'extrapolation_passes': list(range(TRAIN_PASSES + 1, EVAL_PASSES + 1)), 'held_out': 2000, 'seed': args.seed, 'steps': args.steps, 'hop_matrix': hop_matrix(model)}
    meta = {'N': N, 'T': T, 'P': P, 'D': D, 'H': H, 'FF': FF, 'RANK': RANK, 'EPS': EPS, 'passes_max': EVAL_PASSES}
    for name, new in (('toy-weights.json', weights), ('toy-vectors.json', vecs), ('toy-accuracy.json', acc), ('toy-eval.json', ev)):
        path = OUT / name
        old = json.loads(path.read_text()) if path.exists() else {}      # keep the other variant when run alone
        old.update(new)
        if name != 'toy-accuracy.json':
            old['meta'] = meta
        path.write_text(json.dumps(old, indent=1) if name == 'toy-accuracy.json' else json.dumps(old))
    print(json.dumps({k: [round(x, 3) for x in v['accuracy']] for k, v in acc.items()}))


if __name__ == '__main__':
    main()
