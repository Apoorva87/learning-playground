"""Run Inkling 2 Mini in BF16 on the CPU and record what the loop does.

Usage (from the playground root, with the environment in requirements.txt):
    python topics/looped-transformers/assets/capture_inkling.py .local/inkling2/checkpoint .local/inkling2/capture

Transformers dequantizes the checkpoint's NVFP4 and MXFP8 experts while loading
(Inkling2LoopDequantizeExperts), so the model runs with BF16 weights. Hooks recompute
the small quantities the loop depends on from each module's real inputs:

- per visit: which layer, which pass, hidden-state RMS, wall time
- depth-wise attention residual weights over the stored block states
- message attention weights over the previous pass, and the forget/message gates
- routed experts per visit, so passes 1-3 of the same layer can be compared
- KV and convolution cache sizes per cache layer

It also reruns the prompt with 1 and 2 passes. Inkling 2 Mini was trained with
three, so those two runs are outside its training setting.
"""
import hashlib
import inspect
import json
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

PROMPT = ("Mira keeps her passport in the blue box. The blue box sits inside the oak chest. "
          "The oak chest is stored in the attic. So, Mira's passport is ultimately in the")
EXPECTED_SHA256 = "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"
REVISION = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
TRANSFORMERS_COMMIT = "b62646bbda836d467060b7e3d64028202079b77d"


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def r(x, n=5):
    return round(float(x), n)


class Recorder:
    def __init__(self, text_model):
        self.tm = text_model
        cfg = text_model.config
        self.cfg = cfg
        self.H = cfg.hidden_size
        self.reset()
        self.handles = []
        self._install()

    def reset(self):
        self.visit = -1
        self.layer_calls = {}
        self.visits = []
        self.residual = []   # attention residual weights
        self.messages = []   # message attention weights
        self.gates = []
        self.routes = []
        self.t0 = None
        self.cur = None

    # ---- helpers -------------------------------------------------------
    def _stage(self, layer):
        if layer < self.cfg.loop_layer_index_start:
            return "prelude"
        if layer < self.cfg.loop_layer_index_end:
            return "core"
        return "coda"

    def _depth_weights(self, module, block_states, partial, query):
        stacked = torch.cat([block_states, partial.unsqueeze(-2)], dim=-2).float()
        stacked = module.norm(stacked)
        q = query.float().view(-1, self.H)
        scores = torch.einsum("btnh,qh->btqn", stacked, q) * self.H ** -0.5
        return scores.softmax(-1)  # [B, T, Q, N]

    # ---- hooks ---------------------------------------------------------
    def _install(self):
        for idx, layer in enumerate(self.tm.layers):
            self.handles.append(layer.register_forward_pre_hook(self._layer_pre(idx), with_kwargs=True))
            self.handles.append(layer.register_forward_hook(self._layer_post(idx), with_kwargs=True))
            for role in ("attention_input", "mlp_input"):
                mod = getattr(layer, role, None)
                if mod is not None:
                    self.handles.append(mod.register_forward_hook(self._residual_hook(role)))
            for role in ("q", "k", "v", "mlp"):
                mod = getattr(layer, f"{role}_message_gate", None)
                if mod is not None:
                    self.handles.append(mod.register_forward_hook(self._gate_hook(role)))
            router = getattr(getattr(layer, "mlp", None), "gate", None)
            if router is not None:
                self.handles.append(router.register_forward_hook(self._route_hook()))
        self.handles.append(self.tm.final_input.register_forward_hook(self._residual_hook("final_input")))
        self.handles.append(self.tm.message_input.register_forward_hook(self._message_hook()))

    def _layer_pre(self, idx):
        def hook(module, args, kwargs):
            self.visit += 1
            n = self.layer_calls.get(idx, 0)
            self.layer_calls[idx] = n + 1
            self.cur = {"visit": self.visit, "layer": idx, "pass": n + 1 if self._stage(idx) == "core" else None,
                        "stage": self._stage(idx), "kind": module.attention_type.replace("_attention", ""),
                        "messages": (kwargs.get("messages") if "messages" in kwargs else (args[2] if len(args) > 2 else None)) is not None}
            self.t0 = time.perf_counter()
        return hook

    def _layer_post(self, idx):
        def hook(module, args, kwargs, output):
            out = output[0] if isinstance(output, tuple) else output
            x = out.float()
            rms = x.pow(2).mean(-1).sqrt()  # [B, T]
            v = dict(self.cur)
            v["ms"] = r((time.perf_counter() - self.t0) * 1e3, 2)
            v["rms_last"] = r(rms[0, -1])
            v["rms_mean"] = r(rms.mean())
            self.visits.append(v)
        return hook

    def _residual_hook(self, role):
        def hook(module, args, output):
            block_states, partial = args[0], args[1]
            w = self._depth_weights(module, block_states, partial, module.query)
            self.residual.append({
                "visit": self.visit if role != "final_input" else None, "role": role,
                "n_sources": int(w.shape[-1]),
                "w_last": [r(x, 4) for x in w[0, -1, 0].tolist()],
                "w_mean": [r(x, 4) for x in w[0, :, 0].mean(0).tolist()],
            })
        return hook

    def _message_hook(self):
        def hook(module, args, output):
            block_states, partial, loop_layer_idx = args[0], args[1], args[2]
            w = self._depth_weights(module, block_states, partial, module.query[loop_layer_idx])
            # sources: previous pass's block states (after the first), then its final hidden state
            self.messages.append({
                "before_visit": self.visit + 1, "loop_layer": int(loop_layer_idx),
                "layer": self.cfg.loop_layer_index_start + int(loop_layer_idx),
                "n_sources": int(w.shape[-1]),
                "w_last": {role: [r(x, 4) for x in w[0, -1, qi].tolist()] for qi, role in enumerate("q k v mlp".split())},
            })
        return hook

    def _gate_hook(self, role):
        def hook(module, args, output):
            h, m = args[0].float(), args[1].float()
            z = torch.cat([module.norm(h), module.norm(m)], dim=-1)
            z = module.down_proj(z.to(module.down_proj.weight.dtype)).float()
            f = torch.tanh(module.forget_gate_proj(z))
            g = torch.tanh(module.message_gate_proj(z))
            keep, add = ((1 - f) * h), (g * m)
            last = slice(None), -1
            self.gates.append({
                "visit": self.visit, "layer": self.cur["layer"], "pass": self.cur["pass"], "role": role,
                "forget_mean": r(f.mean()), "msg_mean": r(g.mean()), "msg_abs_mean": r(g.abs().mean()),
                "h_norm_last": r(h[last].norm()), "m_norm_last": r(m[last].norm()),
                "keep_norm_last": r(keep[last].norm()), "add_norm_last": r(add[last].norm()),
                "out_norm_last": r((keep + add)[last].norm()),
                "forget_hist": torch.histc(f, bins=20, min=-1, max=1).int().tolist(),
                "msg_hist": torch.histc(g, bins=20, min=-1, max=1).int().tolist(),
                "forget_last_first64": [r(x, 3) for x in f[0, -1, :64].tolist()],
                "msg_last_first64": [r(x, 3) for x in g[0, -1, :64].tolist()],
            })
        return hook

    def _route_hook(self):
        def hook(module, args, output):
            logits, weights, indices = output
            self.routes.append({
                "visit": self.visit, "layer": self.cur["layer"], "pass": self.cur["pass"],
                "experts": indices.reshape(-1, indices.shape[-1]).tolist(),
                "weights_last": [r(x, 4) for x in weights.reshape(-1, weights.shape[-1])[-1].tolist()],
            })
        return hook

    def take(self):
        out = {"visits": self.visits, "residual": self.residual, "messages": self.messages,
               "gates": self.gates, "routes": self.routes}
        self.reset()
        return out


def cache_summary(cache):
    rows = []
    for i, layer in enumerate(getattr(cache, "layers", [])):
        parts = {}

        def add(name, val):
            if torch.is_tensor(val) and val.numel() > 1:
                parts[name] = {"shape": list(val.shape), "dtype": str(val.dtype).replace("torch.", ""),
                               "bytes": val.numel() * val.element_size()}

        for name, val in vars(layer).items():
            if isinstance(val, dict):
                for k, x in val.items():
                    add(f"{name}.{k}", x)
            else:
                add(name, val)
        rows.append({"cache_layer": i, "type": type(layer).__name__,
                     "sliding_window": getattr(layer, "sliding_window", None), "parts": parts})
    return rows


def top_tokens(tok, logits, k=10):
    p = logits.float().softmax(-1)
    vals, ids = p.topk(k)
    return [{"id": int(i), "text": tok.decode([int(i)]), "p": r(v, 5)} for v, i in zip(vals.tolist(), ids.tolist())]


def main(ckpt, out_dir, verify=True):
    ckpt, out_dir = Path(ckpt), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt / "model-00001-of-00001.safetensors"
    meta = {"model": "thinkingmachines/Inkling-2-Mini", "revision": REVISION,
            "transformers": transformers.__version__, "transformers_commit": TRANSFORMERS_COMMIT,
            "torch": torch.__version__, "python": sys.version.split()[0], "machine": platform.platform(),
            "dtype": "bfloat16", "device": "cpu", "threads": torch.get_num_threads(), "prompt": PROMPT}
    if verify:
        digest = sha256(weights)
        assert digest == EXPECTED_SHA256, f"checkpoint hash {digest} does not match {EXPECTED_SHA256}"
        meta["sha256"] = digest

    from transformers import Inkling2LoopForConditionalGeneration as Model
    tok = AutoTokenizer.from_pretrained(ckpt)
    t = time.perf_counter()
    model = Model.from_pretrained(ckpt, dtype=torch.bfloat16).eval()  # CPU is the default device
    meta["load_s"] = r(time.perf_counter() - t, 1)
    tm = model.model.language_model
    cfg = tm.config
    meta["loop"] = {"start": cfg.loop_layer_index_start, "end": cfg.loop_layer_index_end,
                    "passes": cfg.loop_num_passes, "gate_rank": cfg.loop_gate_rank}
    expert_param = next(p for n, p in model.named_parameters() if "experts" in n)
    meta["expert_param_dtype"] = str(expert_param.dtype)
    print("loaded in", meta["load_s"], "s; experts", meta["expert_param_dtype"], flush=True)

    ids = tok(PROMPT, return_tensors="pt").input_ids
    meta["tokens"] = [tok.decode([i]) for i in ids[0].tolist()]
    meta["token_ids"] = ids[0].tolist()
    rec = Recorder(tm)
    cache_kw = "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"
    runs = {}

    def prefill(passes, keep_cache=False):
        cfg.loop_num_passes = passes
        rec.reset()
        t = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=keep_cache, logits_to_keep=1)
        run = {"passes": passes, "phase": "prefill", "T": ids.shape[1], "wall_s": r(time.perf_counter() - t, 2),
               "top": top_tokens(tok, out.logits[0, -1])}
        run.update(rec.take())
        print(f"prefill passes={passes}: {run['wall_s']} s, top {run['top'][0]}", flush=True)
        cache = getattr(out, "past_key_values", None) or getattr(out, "cache", None)
        return run, out, cache

    # Warm the process once, then record. The main record uses the trained setting: three passes.
    prefill(3)
    for k in (1, 2):
        runs[f"prefill_{k}pass"], _, _ = prefill(k)
    main_run, out, cache = prefill(3, keep_cache=True)
    main_run["cache"] = cache_summary(cache)
    runs["prefill_3pass"] = main_run

    nxt = out.logits[0, -1].argmax().view(1, 1)
    rec.reset()
    t = time.perf_counter()
    with torch.inference_mode():
        dec = model(input_ids=nxt, use_cache=True, logits_to_keep=1, **{cache_kw: cache})
    d = {"passes": 3, "phase": "decode", "T": 1, "input": tok.decode([int(nxt)]),
         "wall_s": r(time.perf_counter() - t, 2), "top": top_tokens(tok, dec.logits[0, -1])}
    d.update(rec.take())
    dcache = getattr(dec, "past_key_values", None) or getattr(dec, "cache", None)
    d["cache"] = cache_summary(dcache)
    runs["decode_3pass"] = d
    print(f"decode: {d['wall_s']} s, input {d['input']!r}, top {d['top'][0]}", flush=True)

    path = out_dir / "inkling-capture.json"
    path.write_text(json.dumps({"meta": meta, "runs": runs}, ensure_ascii=False))
    print("wrote", path, f"{path.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], verify="--no-verify" not in sys.argv)
