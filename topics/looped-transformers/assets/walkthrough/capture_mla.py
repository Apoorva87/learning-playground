"""Record a real Multi-head Latent Attention layer: layer 0 of DeepSeek-V2-Lite on "The capital of France is".

Only the embedding rows of the sentence's tokens and the layer-0 attention tensors are fetched, by HTTP
range requests into the model's safetensors shard (about 100 MB instead of 31 GB). The model's own
DeepseekV2Attention module runs on them. From the tensors it records, the explicit scores (rebuild every
head's keys from the latent) and the absorbed scores (fold the key up-projection into the query and
attend in the latent space) are recomputed by hand and checked against each other and the module.

  .local/inkling2/venv/bin/python topics/looped-transformers/assets/walkthrough/capture_mla.py
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from transformers import AutoTokenizer
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention, DeepseekV2RMSNorm, DeepseekV2RotaryEmbedding

REPO = "deepseek-ai/DeepSeek-V2-Lite"
SENTENCE = "The capital of France is"
OUT = Path(__file__).parent / "walkthrough-mla.json"
TRANSFORMERS_COMMIT = "b62646bbda836d467060b7e3d64028202079b77d"
LAYER = "model.layers.0."


def r(x, d=6):
    a = np.asarray(x, dtype=np.float64)
    return np.vectorize(lambda v: float(f"{v:.{d}g}"))(a).tolist() if a.ndim else float(f"{float(a):.{d}g}")


class RemoteShard:
    """Read individual tensors out of a remote safetensors file with Range requests."""

    def __init__(self, repo, filename):
        self.url = hf_hub_url(repo, filename)
        head = self.get(0, 7)
        (n,) = struct.unpack("<Q", head)
        self.base = 8 + n
        self.header = json.loads(self.get(8, 8 + n - 1))

    def get(self, a, b):
        resp = requests.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
        resp.raise_for_status()
        return resp.content

    def tensor(self, name, rows=None):
        info = self.header[name]
        s, e = info["data_offsets"]
        shape = info["shape"]
        assert info["dtype"] == "BF16", info["dtype"]
        if rows is None:
            raw = self.get(self.base + s, self.base + e - 1)
            return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape).float()
        width = shape[1] * 2
        parts = [self.get(self.base + s + i * width, self.base + s + (i + 1) * width - 1) for i in rows]
        return torch.frombuffer(bytearray(b"".join(parts)), dtype=torch.bfloat16).reshape(len(rows), shape[1]).float()


def main():
    api = HfApi()
    sha = api.model_info(REPO).sha
    idx = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
    shard_name = idx["model.embed_tokens.weight"]
    assert all(idx[LAYER + n] == shard_name for n in ["self_attn.q_proj.weight", "input_layernorm.weight"]), "layer 0 is in another shard"
    shard = RemoteShard(REPO, shard_name)
    cfg = DeepseekV2Config.from_pretrained(REPO)
    cfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(REPO)
    ids = tok(SENTENCE, return_tensors="pt").input_ids
    ids_list = ids[0].tolist()
    # DeepSeek-V2-Lite's tokenizer_config sets add_bos_token, but Transformers 5 loads the tokenizer without it,
    # so add the start-of-sequence token by hand: the model is always run with it in front.
    bos = tok.bos_token_id
    assert bos is not None and bos == 100000, bos
    if ids_list[0] != bos:
        ids_list = [bos] + ids_list
    texts = [tok.decode([i]) for i in ids_list]
    T = len(ids_list)
    print("tokens:", list(zip(ids_list, texts)))

    emb = shard.tensor("model.embed_tokens.weight", rows=ids_list)                    # [T, 2048]
    norm = DeepseekV2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = shard.tensor(LAYER + "input_layernorm.weight")
    attn = DeepseekV2Attention(cfg, 0).eval()
    sd = {k: shard.tensor(LAYER + "self_attn." + k + ".weight") for k in ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]}
    attn.q_proj.weight.data, attn.kv_a_proj_with_mqa.weight.data, attn.kv_b_proj.weight.data, attn.o_proj.weight.data = (sd[k] for k in ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"])
    attn.kv_a_layernorm.weight.data = shard.tensor(LAYER + "self_attn.kv_a_layernorm.weight")

    rot = DeepseekV2RotaryEmbedding(cfg)
    pos = torch.arange(T)[None]
    with torch.no_grad():
        x = norm(emb[None])                                                           # [1, T, 2048]
        freqs_cis = rot(x, pos)
        grabbed = {}
        for name in ["q_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj"]:
            getattr(attn, name).register_forward_hook(lambda m, i, o, n=name: grabbed.__setitem__(n, o.detach()))
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)[None, None]
        _, w = attn(x, attention_mask=mask, position_embeddings=freqs_cis)            # w: [1, 16, T, T]
    H, NOPE, ROPE, VD, KR = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    scaling = float(attn.scaling)

    q = grabbed["q_proj"][0].double().numpy().reshape(T, H, NOPE + ROPE)
    q_nope = q[..., :NOPE]
    c_kv = grabbed["kv_a_layernorm"][0].double().numpy()                              # [T, 512], the latent that is cached
    pre = grabbed["kv_a_proj_with_mqa"][0].double().numpy()
    k_pe_raw = pre[:, KR:]
    kvb = grabbed["kv_b_proj"][0].double().numpy().reshape(T, H, NOPE + VD)
    k_nope, v = kvb[..., :NOPE], kvb[..., NOPE:]

    def rope(xr):                                                                      # interleaved pairs times the model's complex frequencies
        z = xr.reshape(*xr.shape[:-1], -1, 2)
        zc = z[..., 0] + 1j * z[..., 1]
        f = freqs_cis[0].numpy().astype(np.complex128)
        f = f.reshape(T, *([1] * (zc.ndim - 2)), -1)
        out = zc * f
        return np.stack([out.real, out.imag], -1).reshape(xr.shape)
    q_pe = rope(q[..., NOPE:])                                                         # [T, H, 64]
    k_pe = rope(k_pe_raw)                                                              # [T, 64], shared by all heads

    hd = 0
    # explicit: every head's key is the up-projection of the latent, plus the shared rope key
    explicit = (q_nope[:, hd] @ k_nope[:, hd].T + q_pe[:, hd] @ k_pe.T) * scaling
    # absorbed: fold W_UK into the query, then score against the latent itself
    Wb = sd["kv_b_proj"].double().numpy().reshape(H, NOPE + VD, KR)
    W_UK = Wb[hd, :NOPE]                                                               # [128, 512]
    q_abs = q_nope[:, hd] @ W_UK                                                       # [T, 512]
    absorbed = (q_abs @ c_kv.T + q_pe[:, hd] @ k_pe.T) * scaling
    tril = np.tril(np.ones((T, T), bool))
    diff = float(np.abs(np.where(tril, explicit - absorbed, 0)).max())
    assert diff < 1e-3, diff
    masked = np.where(tril, explicit, -np.inf)
    e = np.exp(masked - masked.max(-1, keepdims=True)); probs = e / e.sum(-1, keepdims=True)
    hf_probs = w[0, hd].double().numpy()
    pdiff = float(np.abs(probs - hf_probs).max())
    assert pdiff < 1e-4, pdiff
    print(f"explicit vs absorbed max diff {diff:.2e}; vs model probabilities {pdiff:.2e}")

    mla_bytes = (KR + ROPE) * 2
    mha_bytes = H * (NOPE + ROPE + VD) * 2
    data = {
        "meta": {"model": REPO, "revision": sha, "transformers_commit": TRANSFORMERS_COMMIT, "layer": 0, "sentence": SENTENCE, "dtype": "bfloat16 weights, float32 compute",
                 "note": "Only the embedding rows of these tokens and the layer-0 attention weights were read from the checkpoint, by range requests. The model's start-of-sequence token is placed in front of the same five word pieces, so there are six tokens.",
                 "head_shown": hd},
        "tokens": [{"id": i, "text": t} for i, t in zip(ids_list, texts)],
        "config": {"hidden": cfg.hidden_size, "heads": H, "kv_lora_rank": KR, "q_lora_rank": cfg.q_lora_rank, "qk_nope": NOPE, "qk_rope": ROPE, "v_head": VD,
                   "rope_type": cfg.rope_parameters.get("rope_type"), "rope_theta": cfg.rope_parameters.get("rope_theta"), "softmax_scale": scaling,
                   "q_head_dim": NOPE + ROPE, "kv_a_out": KR + ROPE, "kv_b_out": H * (NOPE + VD)},
        "cache_bytes": {"mla_per_token_per_layer": mla_bytes, "mha_per_token_per_layer": mha_bytes, "ratio": mha_bytes / mla_bytes, "dtype_bytes": 2},
        "latent": {"c_kv": r(c_kv), "k_rope": r(k_pe), "c_kv_norms": r(np.linalg.norm(c_kv, axis=1)), "k_pe_before_rope_first8": r(k_pe_raw[:, :8])},
        "query": {"q_nope_head0": r(q_nope[:, hd]), "q_pe_head0": r(q_pe[:, hd]), "q_absorbed_head0": r(q_abs)},
        "explicit": {"k_nope_head0": r(k_nope[:, hd]), "v_head0": r(v[:, hd]), "scores_head0": r(np.where(tril, explicit, 0.0)), "probs_head0": r(probs),
                     "k_nope_shape": [T, H, NOPE], "v_shape": [T, H, VD]},
        "absorbed": {"scores_head0": r(np.where(tril, absorbed, 0.0)), "max_abs_diff": diff, "W_UK_head0_shape": [NOPE, KR]},
        "hf": {"probs_head0": r(hf_probs), "max_abs_prob_diff": pdiff},
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print("wrote", OUT, OUT.stat().st_size // 1000, "kB")


if __name__ == "__main__":
    sys.exit(main())
