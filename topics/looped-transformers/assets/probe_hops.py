"""Score multi-hop answers in Inkling 2 Mini at 1, 2 and 3 loop passes.

Usage: python probe_hops.py <checkpoint> <out.json>

Each story chains an object through containers. A prompt with k facts asks where the
object ultimately is; the right answer is the k-th container and the earlier containers
are the natural wrong answers. Every candidate is scored by teacher forcing its full
token sequence, so tokenization (for example " attic" -> " ", "attic") does not matter.

Inkling 2 Mini was trained with three passes. Runs with one or two passes are outside
that setting and are reported as a probe, not as the model's intended behaviour.
"""
import json
import sys
import time

import torch
from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration as Model

STORIES = [
    {"who": "Mira", "obj": "passport", "chain": ["blue box", "oak chest", "attic"],
     "facts": ["Mira keeps her passport in the blue box.", "The blue box sits inside the oak chest.",
               "The oak chest is stored in the attic."], "ask": "So, Mira's passport is ultimately in the"},
    {"who": "Theo", "obj": "ring", "chain": ["red pouch", "leather bag", "garage"],
     "facts": ["Theo keeps his ring in the red pouch.", "The red pouch is inside the leather bag.",
               "The leather bag is kept in the garage."], "ask": "So, Theo's ring is ultimately in the"},
]


def prompts():
    for s in STORIES:
        for k in (1, 2, 3):
            for order in (("forward", "reversed") if k > 1 else ("forward",)):
                facts = s["facts"][:k] if order == "forward" else list(reversed(s["facts"][:k]))
                yield {"story": s["obj"], "hops": k, "order": order, "prompt": " ".join(facts) + " " + s["ask"],
                       "answer": s["chain"][k - 1], "candidates": s["chain"][:k]}


@torch.inference_mode()
def score(model, tok, prompt, cand):
    p = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + " " + cand, return_tensors="pt").input_ids
    assert torch.equal(full[0, : p.shape[1]], p[0]), "prompt tokens changed when the answer was appended"
    logits = model(input_ids=full, use_cache=False).logits[0].float().log_softmax(-1)
    tgt = full[0, p.shape[1]:]
    lp = logits[p.shape[1] - 1: -1].gather(-1, tgt[:, None]).squeeze(-1)
    return float(lp.sum()), [tok.decode([int(t)]) for t in tgt]


def main(ckpt, out):
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = Model.from_pretrained(ckpt, dtype=torch.bfloat16).eval()
    cfg = model.model.language_model.config
    rows = []
    for item in prompts():
        for passes in (1, 2, 3):
            cfg.loop_num_passes = passes
            t = time.perf_counter()
            scores = {}
            for c in item["candidates"]:
                lp, toks = score(model, tok, item["prompt"], c)
                scores[c] = {"logp": round(lp, 4), "tokens": toks}
            # probability of each candidate renormalized over the candidate set
            z = torch.tensor([scores[c]["logp"] for c in item["candidates"]]).softmax(0).tolist()
            for c, pz in zip(item["candidates"], z):
                scores[c]["p_among"] = round(pz, 4)
                scores[c]["p"] = round(float(torch.tensor(scores[c]["logp"]).exp()), 5)
            best = max(item["candidates"], key=lambda c: scores[c]["logp"])
            row = {**{k: item[k] for k in ("story", "hops", "order", "answer")}, "passes": passes,
                   "correct": best == item["answer"], "scores": scores, "s": round(time.perf_counter() - t, 1)}
            rows.append(row)
            print(f"{item['story']:8s} hops={item['hops']} {item['order']:8s} passes={passes}: "
                  f"P(answer)={scores[item['answer']]['p']:.4f}  among={scores[item['answer']]['p_among']:.3f}  "
                  f"best={best!r} {'✓' if row['correct'] else '✗'}", flush=True)
    cfg.loop_num_passes = 3
    json.dump({"prompts": list(prompts()), "rows": rows}, open(out, "w"), ensure_ascii=False, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
