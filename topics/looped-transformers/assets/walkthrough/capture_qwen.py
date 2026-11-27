"""Record a real forward pass of Qwen3-0.6B-Base on "The capital of France is", then one decode step.

Layer 0 is recomputed by hand in float64 numpy from the model's own weights, and every stage is
checked against the Hugging Face model before anything is written. The recorded numbers are the
hand computation, so the page can recompute the same arithmetic in JavaScript and match them.

Run from the playground root:
  hf download Qwen/Qwen3-0.6B-Base --local-dir .local/walkthrough/qwen3
  .local/inkling2/venv/bin/python topics/looped-transformers/assets/walkthrough/capture_qwen.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / ".local/walkthrough/qwen3"
OUT = Path(__file__).parent / "walkthrough-data.json"
SENTENCE = "The capital of France is"
ALLOWED_NEXT = [" Paris", " the", " a", " located", " one", " not", " also"]
TRANSFORMERS_COMMIT = "b62646bbda836d467060b7e3d64028202079b77d"


def r(x, digits=6):
    """Round to `digits` significant digits so the JSON stays small."""
    a = np.asarray(x, dtype=np.float64)
    return np.vectorize(lambda v: float(f"{v:.{digits}g}"))(a).tolist() if a.ndim else float(f"{float(a):.{digits}g}")


def rms_norm(x, gain, eps):
    ms = (x ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    return x * inv * gain, ms[..., 0], inv[..., 0]


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    ROPE_THETA = float(json.load(open(MODEL_DIR / "config.json"))["rope_theta"])      # Transformers 5 moves this into rope_parameters
    H, KV, HD, D, FF, EPS = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size, cfg.intermediate_size, cfg.rms_norm_eps
    ids = tok(SENTENCE, return_tensors="pt").input_ids
    T = ids.shape[1]
    token_texts = [tok.decode([i]) for i in ids[0].tolist()]
    print("tokens:", list(zip(ids[0].tolist(), token_texts)))
    assert token_texts == ["The", " capital", " of", " France", " is"], token_texts

    outputs = {}
    hooks = [layer.register_forward_hook(lambda m, i, o, n=n: outputs.__setitem__(n, (o[0] if isinstance(o, tuple) else o).detach()))
             for n, layer in enumerate(model.model.layers)]
    with torch.no_grad():
        out = model(ids, use_cache=True, output_attentions=True)
    for h in hooks:
        h.remove()
    cache = out.past_key_values
    hf_probs0 = out.attentions[0][0].numpy()                                       # [16, T, T]

    # ---------- layer 0 by hand, float64 ----------
    P = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    L = "model.layers.0."
    E = P["model.embed_tokens.weight"]
    h0 = E[ids[0].numpy()]                                                         # [T, 1024]
    n1, ms1, inv1 = rms_norm(h0, P[L + "input_layernorm.weight"], EPS)
    q = (n1 @ P[L + "self_attn.q_proj.weight"].T).reshape(T, H, HD)
    k = (n1 @ P[L + "self_attn.k_proj.weight"].T).reshape(T, KV, HD)
    v = (n1 @ P[L + "self_attn.v_proj.weight"].T).reshape(T, KV, HD)
    qn, _, qinv = rms_norm(q, P[L + "self_attn.q_norm.weight"], EPS)
    kn, _, kinv = rms_norm(k, P[L + "self_attn.k_norm.weight"], EPS)
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HD, 2) / HD))                # 64 frequencies
    pos = np.arange(T)
    ang = np.outer(pos, inv_freq)                                                  # [T, 64]
    cos, sin = np.cos(np.concatenate([ang, ang], -1))[:, None, :], np.sin(np.concatenate([ang, ang], -1))[:, None, :]
    qr, kr = qn * cos + rotate_half(qn) * sin, kn * cos + rotate_half(kn) * sin
    raw = np.zeros((H, T, T)); probs = np.zeros((H, T, T)); mixed = np.zeros((T, H, HD))
    for hh in range(H):
        kv = hh // (H // KV)
        s = qr[:, hh] @ kr[:, kv].T / np.sqrt(HD)
        masked = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
        raw[hh] = s
        probs[hh] = softmax(masked)
        mixed[:, hh] = probs[hh] @ v[:, kv]
    concat = mixed.reshape(T, H * HD)
    o = concat @ P[L + "self_attn.o_proj.weight"].T
    resid1 = h0 + o
    n2, ms2, inv2 = rms_norm(resid1, P[L + "post_attention_layernorm.weight"], EPS)
    gate, up = n2 @ P[L + "mlp.gate_proj.weight"].T, n2 @ P[L + "mlp.up_proj.weight"].T
    prod = silu(gate) * up
    down = prod @ P[L + "mlp.down_proj.weight"].T
    h1 = resid1 + down

    hf_h1 = outputs[0][0].double().numpy()
    assert np.allclose(h1, hf_h1, rtol=1e-4, atol=1e-4), np.abs(h1 - hf_h1).max()
    assert np.allclose(probs, hf_probs0, atol=1e-4), np.abs(probs - hf_probs0).max()
    k_cached = cache.layers[0].keys[0].double().numpy()                            # [8, T, 128], after QK-norm and RoPE
    assert np.allclose(k_cached.transpose(1, 0, 2), kr, atol=1e-3)
    print("layer 0 by hand matches the model; max |h1 diff| =", float(np.abs(h1 - hf_h1).max()))

    # ---------- stack, logit lens, head ----------
    def lens(h_last, apply_norm=True):
        x = torch.tensor(h_last).float()
        x = model.model.norm(x) if apply_norm else x
        pr = torch.softmax(model.lm_head(x), -1)
        p, i = pr.max(-1)
        return {"id": int(i), "text": tok.decode([int(i)]), "prob": float(p)}
    last = [outputs[n][0, -1].detach().numpy() for n in range(cfg.num_hidden_layers)]
    stack = {"layer_norms_last": r([float(np.linalg.norm(x)) for x in last]), "logit_lens": [lens(x) for x in last],
             "hidden_norm_embedding_last": float(np.linalg.norm(h0[-1]))}
    final_in = torch.tensor(last[-1])
    with torch.no_grad():
        final_out = model.model.norm(final_in)
        logits = model.lm_head(final_out).double().numpy()
    assert np.allclose(logits, out.logits[0, -1].double().numpy(), atol=1e-3)
    pr = softmax(logits)
    order = np.argsort(-logits)
    top = [{"id": int(i), "text": tok.decode([int(i)]), "logit": float(logits[i]), "prob": float(pr[i])} for i in order[:20]]
    assert top[0]["text"] in ALLOWED_NEXT, [(t["text"], round(t["prob"], 3)) for t in top[:10]]
    temps = []
    for t in (0.5, 1.0, 2.0):
        p = softmax(logits / t); o2 = np.argsort(-p)[:5]
        temps.append({"t": t, "top": [{"text": tok.decode([int(i)]), "prob": float(p[i])} for i in o2]})
    head = {"final_norm_out_last": r(final_out.double().numpy()), "final_norm_gain_first16": r(P["model.norm.weight"][:16]),
            "logits": {"vocab": int(logits.size), "min": float(logits.min()), "max": float(logits.max()), "mean": float(logits.mean()), "std": float(logits.std()),
                       "top20": top, "first16": r(logits[:16])},
            "entropy_nats": float(-(pr * np.log(pr + 1e-30)).sum()), "temperatures": temps,
            "chosen": {"id": top[0]["id"], "text": top[0]["text"], "prob": top[0]["prob"]},
            "tied_embeddings": bool(cfg.tie_word_embeddings), "lm_head_shape": [int(logits.size), D]}
    print("next token:", head["chosen"])

    # ---------- decode: one more token, reading the cache ----------
    new_id = torch.tensor([[top[0]["id"]]])
    with torch.no_grad():
        dec = model(new_id, past_key_values=cache, use_cache=True, output_attentions=True)
    dprobs_hf = dec.attentions[0][0, :, 0].double().numpy()                       # [16, 6]
    en = E[top[0]["id"]][None]
    d1, _, _ = rms_norm(en, P[L + "input_layernorm.weight"], EPS)
    dq = (d1 @ P[L + "self_attn.q_proj.weight"].T).reshape(1, H, HD)
    dk = (d1 @ P[L + "self_attn.k_proj.weight"].T).reshape(1, KV, HD)
    dv = (d1 @ P[L + "self_attn.v_proj.weight"].T).reshape(1, KV, HD)
    dqn, _, _ = rms_norm(dq, P[L + "self_attn.q_norm.weight"], EPS)
    dkn, _, _ = rms_norm(dk, P[L + "self_attn.k_norm.weight"], EPS)
    a5 = (5 * inv_freq)[None]; c5, s5 = np.cos(np.concatenate([a5, a5], -1))[:, None], np.sin(np.concatenate([a5, a5], -1))[:, None]
    dqr, dkr = dqn * c5 + rotate_half(dqn) * s5, dkn * c5 + rotate_half(dkn) * s5
    K6 = np.concatenate([kr, dkr], 0); V6 = np.concatenate([v, dv], 0)           # [6, 8, 128]
    dprobs = np.zeros((H, 6)); dmixed = np.zeros((H, HD))
    for hh in range(H):
        kv = hh // (H // KV)
        dprobs[hh] = softmax(dqr[0, hh] @ K6[:, kv].T / np.sqrt(HD)); dmixed[hh] = dprobs[hh] @ V6[:, kv]
    assert np.allclose(dprobs, dprobs_hf, atol=1e-4), np.abs(dprobs - dprobs_hf).max()
    dlog = dec.logits[0, -1].double().numpy(); dpr = softmax(dlog); do = np.argsort(-dlog)[:10]
    decode = {"token": {"id": top[0]["id"], "text": top[0]["text"]}, "position": T, "embed_first16": r(en[0, :16]),
              "norm1_out_first16": r(d1[0, :16]), "q_heads_first8": r(dqr[0, :, :8]), "k_new_first8": r(dkr[0, :, :8]), "v_new_first8": r(dv[0, :, :8]),
              "probs": r(dprobs), "mixed_head0_first8": r(dmixed[0, :8]),
              "cache_keys_head0_first4": r(K6[:, 0, :4]), "cache_len_after": 6,
              "q_head0": r(dqr[0, 0]), "k6_head0": r(K6[:, 0]), "v6_head0": r(V6[:, 0]),
              "logits_top10": [{"id": int(i), "text": tok.decode([int(i)]), "logit": float(dlog[i]), "prob": float(dpr[i])} for i in do]}
    print("decode next:", decode["logits_top10"][:3])

    data = {
        "meta": {"model": "Qwen/Qwen3-0.6B-Base", "revision": HfApi().model_info("Qwen/Qwen3-0.6B-Base").sha, "transformers_commit": TRANSFORMERS_COMMIT,
                 "transformers_version": transformers.__version__, "torch": torch.__version__, "dtype": "float32", "sentence": SENTENCE,
                 "tokens": [{"id": int(i), "text": t} for i, t in zip(ids[0].tolist(), token_texts)], "positions": list(range(T)),
                 "note": "Layer 0 is recomputed in float64 numpy from the model weights and checked against the model (tolerance 1e-4). Numbers are rounded to 6 significant digits."},
        "config": {"hidden": D, "layers": cfg.num_hidden_layers, "heads": H, "kv_heads": KV, "head_dim": HD, "ffn": FF, "vocab": cfg.vocab_size,
                   "rope_theta": ROPE_THETA, "eps": EPS, "tied": bool(cfg.tie_word_embeddings), "q_dim": H * HD, "kv_dim": KV * HD},
        "embed": {"h0": r(h0), "row_norms": r(np.linalg.norm(h0, axis=1))},
        "layer0": {
            "norm1": {"mean_square": r(ms1), "inv_rms": r(inv1), "gain": r(P[L + "input_layernorm.weight"]), "out": r(n1)},
            "qkv": {"q0": r(q[:, 0]), "k0": r(k[:, 0]), "v0": r(v[:, 0]), "q_last_heads": r(q[-1, :, :8]), "k_last_heads": r(k[-1, :, :8]), "v_last_heads": r(v[-1, :, :8]),
                    "q_shape": [T, H, HD], "k_shape": [T, KV, HD], "v_shape": [T, KV, HD]},
            "qknorm": {"q_gain": r(P[L + "self_attn.q_norm.weight"]), "k_gain": r(P[L + "self_attn.k_norm.weight"]), "q0_out": r(qn[:, 0]), "k0_out": r(kn[:, 0]),
                       "q0_inv_rms": r(qinv[:, 0]), "k0_inv_rms": r(kinv[:, 0])},
            "rope": {"inv_freq": r(inv_freq), "angles": r(ang), "q0_out": r(qr[:, 0]), "k0_out": r(kr[:, 0])},
            "scores": {"heads": [{"raw": r(np.where(np.tril(np.ones((T, T), bool)), raw[hh], 0.0)), "probs": r(probs[hh])} for hh in range(H)]},
            "attn_out": {"head0": r(mixed[:, 0]), "concat_last": r(concat[-1]), "o_proj_out": r(o)},
            "resid1": {"out": r(resid1)},
            "norm2": {"mean_square": r(ms2), "inv_rms": r(inv2), "gain": r(P[L + "post_attention_layernorm.weight"]), "out": r(n2)},
            "ffn": {"gate_last": r(gate[-1]), "up_last": r(up[-1]), "silu_last": r(silu(gate[-1])), "prod_last": r(prod[-1]), "down_last": r(down[-1])},
            "out": {"h1": r(h1)},
        },
        "stack": stack, "head": head,
        "cache": {"k_shape": [KV, T, HD], "v_shape": [KV, T, HD], "layers": cfg.num_hidden_layers, "dtype_bytes_stored": 2,
                  "first_layer_k_head0_first4": r(k_cached[0, :, :4])},
        "decode": decode,
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print("wrote", OUT, OUT.stat().st_size // 1000, "kB")


if __name__ == "__main__":
    sys.exit(main())
