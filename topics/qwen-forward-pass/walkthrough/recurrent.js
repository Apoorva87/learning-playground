window.QWEN_RECURRENT = [
  {
    id: "rec-project",
    title: "Project the query, key, value and gates",
    action: "Four learned projections read the same normalized token vector. One packs the query, key and value together. The other three produce the output gate z and two numbers per value head that control writing and forgetting.",
    why: "The query chooses what to read, the key chooses where to write, and the value supplies the content. A sigmoid turns b into the write fraction β. The negative log decay g comes from a and two learned head parameters, so exp(g) gives the fraction of old memory to retain.",
    shape: "Each B × T × 5,120 input produces 10,240 packed features, 6,144 gate features, and two sets of 48 control values per token.",
    code: `qkv = self.in_proj_qkv(x)
z = self.in_proj_z(x).reshape(B, T, 48, 128)
b, a = self.in_proj_b(x), self.in_proj_a(x)
beta = b.sigmoid()
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`,
    source: [565, 616],
    nodes: [
      {id: "x", label: "Normalized token", shape: [1, "T", 5120], role: "input", col: 0, row: 1, detail: "All four projections see this same input from the decoder block’s first RMSNorm."},
      {id: "qkv", label: "Packed Q, K, V", shape: [1, "T", 10240], role: "output", col: 1, row: 0, detail: "The first 2,048 channels hold queries, the next 2,048 hold keys, and the final 6,144 hold values."},
      {id: "z", label: "Output gate z", shape: [1, "T", 48, 128], role: "output", col: 1, row: 1, detail: "One gate value per readout feature waits until the state update and normalization are complete."},
      {id: "ba", label: "Controls b and a", shape: [1, "T", 48], role: "work", col: 1, row: 2, detail: "Two separate projections produce this shape each. Each of the 48 value heads gets its own b and a."},
      {id: "beta", label: "Write fraction β", shape: [1, "T", 48], role: "output", col: 2, row: 1, detail: "The sigmoid of b scales how much of the value prediction error gets written."},
      {id: "g", label: "Log decay g", shape: [1, "T", 48], role: "output", col: 2, row: 2, detail: "The recurrence exponentiates g. Values of g close to zero keep more of the old state."}
    ],
    edges: [["x", "qkv"], ["x", "z"], ["x", "ba"], ["ba", "beta"], ["ba", "g"]]
  },
  {
    id: "rec-conv",
    title: "Mix four neighboring positions",
    action: "Each of the 10,240 channels applies its own four learned weights to the current position and the three before it, then applies SiLU. The filtered channels split back into query, key and value heads.",
    why: "At the start of a fresh prompt, missing earlier positions contribute zeros. For the 17th token (index 16), cached decode combines the new projected input with indices 13, 14 and 15, giving the same causal window as processing that position in a longer prompt.",
    shape: "The filter preserves B × T × 10,240. Q and K each reshape to B × T × 16 × 128. V reshapes to B × T × 48 × 128.",
    code: `x = qkv.transpose(1, 2)  # channels before time
if T == 1 and conv_state is not None:
    y = causal_conv1d_update(x, conv_state, kernel, None, "silu")
else:
    x = cache.update_conv_state(x, layer_idx, conv_kernel_size=4)
    y = causal_conv1d_fn(x, kernel, None, "silu")[..., -T:]
q, k, v = y.transpose(1, 2).split([2048, 2048, 6144], dim=-1)`,
    source: [574, 612],
    nodes: [
      {id: "x", label: "Packed channels", shape: [1, 10240, "T"], role: "input", col: 0, row: 0, detail: "Transposing the input places positions on the last axis for the convolution."},
      {id: "cache", label: "Convolution cache", shape: [1, 10240, 4], role: "state", col: 0, row: 1, detail: "Prefill saves the last four input samples per channel. Decode slides the window forward and uses the newest four, including the incoming sample."},
      {id: "weight", label: "Filter weights", shape: [10240, 1, 4], role: "weight", col: 0, row: 2, detail: "Each output channel sees one input channel across four positions. The kernel helper squeezes the middle axis."},
      {id: "y", label: "Filtered channels", shape: [1, 10240, "T"], role: "work", col: 1, row: 1, detail: "SiLU transforms each channel’s weighted sum before the split."},
      {id: "qk", label: "Queries and keys", shape: [1, "T", 16, 128], role: "output", col: 2, row: 0, detail: "Q and K are separate tensors of this shape. Each has 16 heads with 128 features."},
      {id: "v", label: "Values", shape: [1, "T", 48, 128], role: "output", col: 2, row: 2, detail: "Splitting and reshaping the last 6,144 filtered channels gives 48 independent value vectors per position."}
    ],
    edges: [["x", "y"], ["cache", "y"], ["weight", "y"], ["y", "qk"], ["y", "v"]]
  },
  {
    id: "rec-update",
    title: "Correct the memory, then read it",
    action: "Each of the 16 query/key heads serves three consecutive value heads. The rule L2-normalizes Q and K, decays the old state, then writes a β-scaled correction where the key’s prediction differs from the incoming value. Prefill evaluates the rule in chunks, while cached decode applies one update to the saved state.",
    why: "Imagine one memory entry holding 8, with key 1 and retention 0.5. After decay the prediction is 4, so an incoming value of 10 with β = 0.5 adds half the gap, or 3. The entry now holds 7, and the query reads from this updated state.",
    shape: "Each block retains B × 48 × 128 × 128 state values across calls. For every new position, each of the 48 heads reads a 128-feature vector.",
    code: `# One token, one value head, and its shared query and key
q, k = l2norm(q.float()) / (128 ** 0.5), l2norm(k.float())
faded = g.exp() * state
predicted = k @ faded
delta = beta * (v.float() - predicted)
state = faded + torch.outer(k, delta)
read = q @ state`,
    source: [448, 487],
    nodes: [
      {id: "old", label: "Previous state", shape: [1, 48, 128, 128], role: "state", col: 0, row: 0, detail: "A fresh prompt begins with zeros. Each head’s FP32 matrix maps a key direction to a predicted value."},
      {id: "kv", label: "Key and value", shape: [1, 48, 128], role: "input", col: 0, row: 1, detail: "For this single position, each tensor has the displayed shape after sharing each normalized key across three value heads."},
      {id: "controls", label: "β and g", shape: [1, 48], role: "input", col: 0, row: 2, detail: "Each scalar tensor supplies one number per head at this position. exp(g) scales the previous matrix and β scales the value correction."},
      {id: "new", label: "Updated state", shape: [1, 48, 128, 128], role: "state", col: 1, row: 1, detail: "The outer product writes a vector correction along the current key. This state becomes the starting point for the next position."},
      {id: "q", label: "Scaled query", shape: [1, 48, 128], role: "input", col: 1, row: 2, detail: "Each shared query is L2-normalized, then divided by the square root of 128 before reading the state."},
      {id: "read", label: "Current read", shape: [1, 48, 128], role: "output", col: 2, row: 1, detail: "Each query reads its head’s updated matrix, producing one vector for this position."}
    ],
    edges: [["old", "new"], ["kv", "new"], ["controls", "new"], ["new", "read"], ["q", "read"]]
  },
  {
    id: "rec-readout",
    title: "Gate what reaches the residual stream",
    action: "RMSNorm rescales each 128-feature head read and applies its learned feature weights. The mixer multiplies the result by SiLU(z), joins the 48 heads and projects the joined vector to the model width.",
    why: "The same token can update memory for a later position while passing little of the current read into the block. The output gate makes that choice feature by feature, after the recurrent update has already occurred.",
    shape: "B × T × 48 × 128 becomes B × T × 6,144. The output projection maps 6,144 features to 5,120 so the decoder can add the mixer result to the residual stream.",
    code: `rows = read.reshape(-1, 128)
inv_rms = (rows.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
normed = self.norm.weight * (rows.float() * inv_rms).to(rows.dtype)
gated = normed * F.silu(z.reshape(-1, 128).float())
merged = gated.to(rows.dtype).reshape(B, T, 6144)
output = self.out_proj(merged)`,
    source: [649, 656],
    nodes: [
      {id: "read", label: "Head reads", shape: [1, "T", 48, 128], role: "input", col: 0, row: 0, detail: "The recurrent kernel returns one 128-feature read for each head and input position."},
      {id: "normed", label: "RMS-normalized", shape: [1, "T", 48, 128], role: "work", col: 1, row: 0, detail: "The mean square is computed in FP32 across the 128 features of one head. All heads share the same 128 learned normalization weights."},
      {id: "z", label: "Output gate z", shape: [1, "T", 48, 128], role: "input", col: 1, row: 2, detail: "The z projection from the start of this mixer supplies a SiLU multiplier for each normalized feature."},
      {id: "merged", label: "Gated heads", shape: [1, "T", 6144], role: "work", col: 2, row: 1, detail: "The normalized reads are multiplied by SiLU(z). Flattening the head and feature axes makes one row per token."},
      {id: "weight", label: "Projection weight", shape: [5120, 6144], role: "weight", col: 2, row: 2, detail: "Each of the 5,120 output features combines all 6,144 input features through this learned matrix."},
      {id: "output", label: "Mixer output", shape: [1, "T", 5120], role: "output", col: 3, row: 1, detail: "The mixer returns a residual-width update for every input position."}
    ],
    edges: [["read", "normed"], ["normed", "merged"], ["z", "merged"], ["merged", "output"], ["weight", "output"]]
  }
];
