"""Collect small activation samples from the pinned Qwen checkpoint on CPU.

Run from the repository root after downloading the checkpoint using assets/get_checkpoint.py:
    HF_HUB_OFFLINE=1 .local/tools/huggingarch/source/.venv/bin/python \
        topics/qwen-forward-pass/walkthrough/capture_steps.py .local/qwen38/checkpoint

This verifies the 18 shards, runs two full forward calls, checks the existing
fixture and writes observed.json next to this script. No full tensor is saved.
"""
import argparse
import hashlib
import json
import math
import platform
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import transformers
from tokenizers import Tokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
SPEC = json.loads((ASSETS / "workload.json").read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample(tensor, *index):
    """Expose the real index, including the token position in this particular tensor."""
    index = tuple(i if i >= 0 else tensor.shape[axis] + i for axis, i in enumerate(index))
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype).removeprefix("torch."),
                index=list(index), value=tensor[index].item())


def delta_example(query, key, value, g, beta, initial, native_output, native_state):
    """Replay one value head in FP32, comparing every read and final state with the native kernel."""
    # Value heads 0, 1 and 2 share key/query head 0. Show value head 0.
    qs = qwen.l2norm(query[0, :, 0].float()) / math.sqrt(query.shape[-1])
    ks = qwen.l2norm(key[0, :, 0].float())
    vs = value[0, :, 0].float()
    state = torch.zeros_like(native_state[0, 0]) if initial is None else initial[0, 0].clone()
    reads = []
    for t, (q, k, v) in enumerate(zip(qs, ks, vs)):
        decay = g[0, t, 0].float().exp()
        strength = beta[0, t, 0].float()
        old = state
        decayed = state * decay
        predicted = (decayed * k[:, None]).sum(0)
        error = v - predicted
        correction = error * strength
        state = decayed + k[:, None] * correction[None, :]
        read = (state * q[:, None]).sum(0)
        reads.append(read)
    expected_reads = torch.stack(reads)
    native_reads = native_output[0, :, 0]
    state_error = (native_state[0, 0] - state).abs().max().item()
    read_error = (native_reads - expected_reads.to(native_reads.dtype)).abs().max().item()
    torch.testing.assert_close(state, native_state[0, 0], atol=5e-4, rtol=2e-4)
    torch.testing.assert_close(expected_reads.to(native_reads.dtype), native_reads, atol=5e-4, rtol=0.01)
    return dict(token_offset=t, value_head=0, key_feature=0, value_feature=0,
                old=old[0, 0].item(), log_decay=g[0, t, 0].item(), decay=decay.item(),
                decayed=decayed[0, 0].item(), normalized_key=k[0].item(),
                target_value=v[0].item(), predicted_value=predicted[0].item(),
                error=error[0].item(), beta=strength.item(), correction=correction[0].item(),
                write=(k[0] * correction[0]).item(), updated=state[0, 0].item(),
                native_updated=native_state[0, 0, 0, 0].item(), scaled_query=q[0].item(),
                read_fp32=read[0].item(), native_read=native_reads[-1, 0].item(),
                final_head_state_max_abs_error=state_error,
                all_head_reads_max_abs_error_after_bf16=read_error)


class Capture:
    """Hooks retain selected activations for one call. Kernel wrappers return the original tensors."""
    def __init__(self):
        self.t = {}
        self.r = {}
        self.active = None

    def hook(self, stack, module, name, capture_input=False):
        def observe(_module, args, output):
            self.t[name] = output
            if capture_input:
                self.t[name + "_input"] = args[0]
        stack.callback(module.register_forward_hook(observe).remove)

    def scope(self, stack, module, name):
        def enter(_module, _args):
            self.active = name
        def leave(_module, _args, _output):
            self.active = None
        stack.callback(module.register_forward_pre_hook(enter).remove)
        stack.callback(module.register_forward_hook(leave).remove)

    def watch(self, stack, name, scope, observer):
        native = getattr(qwen, name)
        def call(*args, **kwargs):
            output = native(*args, **kwargs)
            if self.active == scope:
                observer(args, kwargs, output)
            return output
        stack.enter_context(patch.object(qwen, name, call))

    def attach(self, stack, model):
        lm = model.model.language_model
        self.hook(stack, lm.embed_tokens, "embedding")
        self.hook(stack, lm.norm, "final_norm", True)
        self.hook(stack, model.lm_head, "logits", True)
        for i in (0, 3):
            layer = lm.layers[i]
            prefix = f"b{i}_"
            self.hook(stack, layer, prefix + "result")
            for name in ("input_layernorm", "post_attention_layernorm"):
                self.hook(stack, getattr(layer, name), prefix + name, True)
            for name in ("gate_proj", "up_proj", "down_proj"):
                self.hook(stack, getattr(layer.mlp, name), prefix + name, name == "down_proj")
            self.hook(stack, layer.mlp.act_fn, prefix + "silu")
        linear = lm.layers[0].linear_attn
        self.scope(stack, linear, "linear")
        for name in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"):
            self.hook(stack, getattr(linear, name), name, name == "out_proj")
        def observe_read_norm(module, args, output):
            read, gate = args
            promoted = read.float()
            inv_rms = (promoted.square().mean(-1, keepdim=True) + module.variance_epsilon).rsqrt()
            normalized = (promoted * inv_rms).to(read.dtype) * module.weight
            factor = torch.nn.functional.silu(gate.float())
            assert torch.equal((normalized * factor).to(output.dtype), output)
            shape = self.t["in_proj_z"].shape[:2] + (linear.num_v_heads, linear.head_v_dim)
            self.r["normalized_readout"] = sample(normalized.view(shape), 0, -1, 0, 0)
        stack.callback(linear.norm.register_forward_hook(observe_read_norm).remove)
        full = lm.layers[3].self_attn
        self.scope(stack, full, "full")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            self.hook(stack, getattr(full, name), name, name in ("o_proj", "q_norm", "k_norm"))
        for name in ("causal_conv1d_fn", "causal_conv1d_update"):
            self.watch(stack, name, "linear", self.convolution)
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            self.watch(stack, name, "linear", self.delta)
        self.watch(stack, "apply_rotary_pos_emb", "full", self.rope)
        self.watch(stack, "eager_attention_forward", "full", self.attention)

    def convolution(self, args, kwargs, output):
        self.r["convolution_activated"] = sample(output, 0, 0, -1)

    def delta(self, args, kwargs, output):
        query, key, value = args[:3]
        g, beta, initial = (kwargs[name] for name in ("g", "beta", "initial_state"))
        read, state = output
        for name, tensor in (("q", query), ("k", key), ("v", value), ("readout", read)):
            self.r[name] = sample(tensor, 0, -1, 0, 0)
        self.r["log_decay"] = sample(g, 0, -1, 0)
        self.r["beta"] = sample(beta, 0, -1, 0)
        self.r["state_in"] = None if initial is None else sample(initial, 0, 0, 0, 0)
        self.r["state_out"] = sample(state, 0, 0, 0, 0)
        self.r["delta_example"] = delta_example(query, key, value, g, beta, initial, read, state)

    def rope(self, args, kwargs, output):
        q, k, cos, sin = args[:4]
        self.t.update(q_rotated=output[0], k_rotated=output[1])
        rotary_width = cos.shape[-1]
        assert torch.equal(q[..., rotary_width:], output[0][..., rotary_width:])
        assert torch.equal(k[..., rotary_width:], output[1][..., rotary_width:])
        self.r["rotation"] = dict(width=rotary_width, unchanged_tail=q.shape[-1] - rotary_width,
                                  cosine=sample(cos, 0, -1, 0), sine=sample(sin, 0, -1, 0),
                                  paired_q=sample(q, 0, 0, -1, rotary_width // 2))

    def attention(self, args, kwargs, output):
        module, q, k, v, mask = args[:5]
        mixed, probability = output
        self.t.update(attention_q=q, cache_k=k, cache_v=v, attention_mix=mixed)
        scores = (q @ qwen.repeat_kv(k, module.num_key_value_groups).transpose(-1, -2)) * kwargs["scaling"]
        masked = scores if mask is None else scores + mask
        assert torch.equal(masked.softmax(-1, dtype=torch.float32).to(q.dtype), probability)
        recomputed = probability @ qwen.repeat_kv(v, module.num_key_value_groups)
        assert torch.equal(recomputed.transpose(1, 2), mixed)
        self.r["attention_row"] = dict(
            head=0, kv_head=0, query_position=k.shape[-2] - 1, scale=kwargs["scaling"],
            scores=sample(scores, 0, 0, -1, -1), weights=sample(probability, 0, 0, -1, -1),
            scores_by_position=scores[0, 0, -1].tolist(),
            weights_by_position=probability[0, 0, -1].tolist(),
            scores_by_head=scores[0, :, -1].tolist(),
            weights_by_head=probability[0, :, -1].tolist(),
            value_feature0_by_position=v[0, 0, :, 0].tolist(),
            mixed=sample(mixed, 0, -1, 0, 0),
            first_query_nonzero_positions=probability[0, 0, 0].nonzero().flatten().tolist())

    def common(self, layer):
        t = self.t
        prefix = f"b{layer}_"
        entry = t[prefix + "input_layernorm_input"]
        mixed = t["out_proj" if layer == 0 else "o_proj"]
        residual = t[prefix + "post_attention_layernorm_input"]
        assert torch.equal(entry + mixed, residual)
        assert torch.equal(residual + t[prefix + "down_proj"], t[prefix + "result"])
        assert torch.equal(t[prefix + "silu"] * t[prefix + "up_proj"], t[prefix + "down_proj_input"])
        names = {"residual_in": "input_layernorm_input", "normalized": "input_layernorm",
                 "residual_after_mixer": "post_attention_layernorm_input", "mlp_input": "post_attention_layernorm",
                 "mlp_gate": "gate_proj", "mlp_silu": "silu", "mlp_up": "up_proj",
                 "mlp_product": "down_proj_input", "mlp_down": "down_proj", "residual_out": "result"}
        result = {label: sample(t[prefix + key], 0, -1, 0) for label, key in names.items()}
        result["row_rms_before_input_norm"] = entry[0, -1].float().square().mean().sqrt().item()
        result["mixer_output"] = sample(mixed, 0, -1, 0)
        return result

    def finish(self, output, ids, cache_before, tokenizer, reference_top):
        t, r = self.t, self.r
        block0, block3 = self.common(0), self.common(3)
        for name in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z"):
            block0[name] = sample(t[name], 0, -1, 0)
        block0["gated_readout"] = sample(t["out_proj_input"], 0, -1, 0)
        block0["silu_z"] = sample(torch.nn.functional.silu(t["in_proj_z"].float()), 0, -1, 0)
        block0["conv_state_in"] = cache_before
        conv = output.cache.get_conv_state(0)
        block0["conv_state_out"] = sample(conv, 0, 0, -1)
        block0["conv_channel0_newest_last"] = conv[0, 0].tolist()
        for name in ("convolution_activated", "q", "k", "v", "readout", "log_decay", "beta",
                     "state_in", "state_out", "delta_example", "normalized_readout"):
            block0[name] = r[name]
        assert sample(output.cache.get_recurrent_state(0), 0, 0, 0, 0) == block0["state_out"]

        for name in ("q_proj", "k_proj", "v_proj", "o_proj_input"):
            block3[name] = sample(t[name], 0, -1, 0)
        for name in ("q_norm_input", "q_norm", "k_norm_input", "k_norm"):
            block3[name] = sample(t[name], 0, -1, 0, 0)
        for name in ("q_rotated", "k_rotated", "cache_k", "cache_v"):
            block3[name] = sample(t[name], 0, 0, -1, 0)
        gate = t["q_proj"].unflatten(-1, (24, 512)).chunk(2, dim=-1)[1]
        factor = gate.sigmoid()
        assert torch.equal((t["attention_mix"] * factor).flatten(-2), t["o_proj_input"])
        block3.update(gate=sample(gate, 0, -1, 0, 0), sigmoid_gate=sample(factor, 0, -1, 0, 0),
                      rotation=r["rotation"], attention_row=r["attention_row"])
        scores = output.logits[0, -1]
        winner = scores.argmax().item()
        assert winner == reference_top
        assert torch.equal(t["logits_input"], t["final_norm"][:, -1:])
        assert torch.equal(t["logits"], output.logits)
        return dict(input_ids=ids, embedding=sample(t["embedding"], 0, -1, 0), block0=block0, block3=block3,
                    final_norm_input=sample(t["final_norm_input"], 0, -1, 0),
                    final_norm=sample(t["final_norm"], 0, -1, 0),
                    head_input=sample(t["logits_input"], 0, 0, 0), logits=sample(output.logits, 0, 0, winner),
                    prediction=dict(id=winner, text=tokenizer.decode([winner]), logit=scores[winner].item()),
                    supplied_decode_token_logit=scores[SPEC["decode_id"]].item())


def run(checkpoint):
    for shard in SPEC["shards"]:
        path = checkpoint / shard["name"]
        if path.stat().st_size != shard["size"] or sha256(path) != shard["sha256"]:
            raise ValueError(f"Shard differs from workload.json: {path.name}")
    print("Verified all checkpoint shards", flush=True)
    previous = json.loads((ASSETS / "whole-pass.json").read_text())
    fixture_path = ASSETS / "layer3-decode.npz"
    assert sha256(fixture_path) == previous["fixture"]["sha256"]
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    assert tokenizer.encode(SPEC["prompt"], add_special_tokens=False).ids == SPEC["prompt_ids"]
    assert tokenizer.decode([SPEC["decode_id"]]) == SPEC["decode_text"]
    torch.set_num_threads(SPEC["torch_threads_full"])
    torch.set_num_interop_threads(1)
    model = qwen.Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=True, attn_implementation="eager", dtype=torch.bfloat16).eval()
    report = dict(model=SPEC["model"], revision=SPEC["revision"], device="CPU", machine=platform.machine(),
                  torch=str(torch.__version__), transformers=transformers.__version__,
                  verified_shards=len(SPEC["shards"]), fixture_sha256=previous["fixture"]["sha256"],
                  method=[
                      "Both calls run all 64 text blocks and request logits only at the last position. Decode consumes the actual prefill cache.",
                      "Tensor records give the native shape, dtype, exact index and element value. Samples use the final input token, head 0 and feature 0 unless indexed otherwise.",
                      "Module hooks and wrappers around the native convolution, delta rule, rotary and attention functions collect the activations. The wrappers return the original outputs.",
                      "The delta example replays one value head and tracks the final token. A missing prefill state starts at zero. The native prefill uses a chunked algorithm and decode uses a recurrent algorithm.",
                      "The attention row reconstructs the native scaled scores before masking and records native probabilities after FP32 softmax and BF16 rounding. The final query sees every supplied position.",
                      "Both residual sums, MLP products, attention probabilities, value mixtures and full-attention output gates are checked over complete tensors in the selected blocks or mixer.",
                      "The block 0 readout normalization and SiLU gate are verified over all heads. The block 3 decode keys and values preserve the full prefill prefix and append the new rotated key and projected value.",
                      "Both top token IDs match whole-pass.json. Block 3 decode Q, cached K/V and native value mixture match every element of layer3-decode.npz."
                  ])
    capture = Capture()
    cache = None
    with ExitStack() as stack, torch.inference_mode():
        capture.attach(stack, model)
        for phase, ids in (("prefill", SPEC["prompt_ids"]), ("decode", [SPEC["decode_id"]])):
            capture.t.clear()
            capture.r.clear()
            prior_conv = None if cache is None else sample(cache.get_conv_state(0), 0, 0, -1)
            prior_kv = None if cache is None else (cache.layers[3].keys, cache.layers[3].values)
            output = model(input_ids=torch.tensor([ids]), cache=cache, use_cache=True, logits_to_keep=1)
            cache = output.cache
            report[phase] = capture.finish(output, ids, prior_conv, tokenizer, previous[phase]["top_id"])
            cached_k, cached_v = capture.t["cache_k"], capture.t["cache_v"]
            assert torch.equal(cached_k[:, :, -len(ids):], capture.t["k_rotated"])
            projected_v = capture.t["v_proj"].unflatten(-1, (4, 256)).transpose(1, 2)
            assert torch.equal(cached_v[:, :, -len(ids):], projected_v)
            if prior_kv is not None:
                assert torch.equal(prior_kv[0], cached_k[:, :, :-len(ids)])
                assert torch.equal(prior_kv[1], cached_v[:, :, :-len(ids)])
            report[phase]["block3"]["prior_kv_positions"] = cached_k.shape[-2] - len(ids)
            report[phase]["block3"]["appended_kv_positions"] = len(ids)
            print(f"Captured {phase}", flush=True)
        with np.load(fixture_path, allow_pickle=False) as fixture:
            for ref, observed in (("q", capture.t["attention_q"]), ("k", capture.t["cache_k"]),
                                  ("v", capture.t["cache_v"]), ("native", capture.t["attention_mix"].transpose(1, 2))):
                assert torch.equal(observed.float(), torch.from_numpy(fixture[ref]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    report = run(args.checkpoint)
    (HERE / "observed.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
