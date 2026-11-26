/* Shapes are logical tensor axes. W is shown in PyTorch's [out, in] storage order. */
(() => {
  const node = (id, label, shape, role, col, row, detail) => ({id, label, shape, role, col, row, detail});
  const hidden = [1, "T", 5120];
  const q = [1, 24, "T", 256], kv = [1, 4, "L", 256];
  const scores = [1, 24, "T", "L"];
  const h = (id, label, col, row, detail, role="input") => node(id,label,hidden,role,col,row,detail);
  function entry(recurrent) {
    return {
      id:recurrent?"rec-open":"full-open", title:recurrent?"Normalize the block input":"Enter the first full-attention block",
      action:recurrent?"Block 0 receives one 5,120-number vector per current token. It saves that vector for a skip connection and normalizes a copy for the token mixer. RMSNorm divides by the root mean square of the features and applies a learned scale.":"Blocks 1 and 2 repeat the recurrent block’s two residual updates with their own weights. Block 3 receives the result. It saves the incoming vectors and applies the same input RMSNorm. This time the configuration sends the normalized vectors into grouped-query attention.",
      why:"The skip path carries the existing representation intact to the addition. Normalization controls the magnitude seen by the new calculation. Every block preserves the outer width, so its update can be added element by element.",
      shape:"Input, normalized input and skip path all have shape [1, T, 5120]. Statistics are calculated separately for each token over its 5,120 features, with epsilon 0.000001.",
      code:"skip = x\nrms = sqrt(mean(x.float() ** 2, dim=-1, keepdim=True) + eps)\nu = (x.float() / rms * (1 + norm_weight)).to(x.dtype)",
      source:[835,903],
      nodes:[h("x","Block input",0,0,"The residual stream entering this particular block."),
        h("skip","Skip path",1,0,"The original tensor for the first residual addition."),
        node("w","Norm scale",[5120],"weight",0,1,"One learned adjustment for each hidden feature."),
        h("u","Normalized input",2,1,"The token mixer receives this representation.", "output")],
      edges:[["x","skip"],["x","u"],["w","u"]]
    };
  }
  function join(recurrent) {
    return {
      id:recurrent?"rec-skip":"full-skip",title:"Add the mixer update and normalize again",
      action:"The mixer has returned a 5,120-feature update for each current token. Add it to the block’s original input. Save the sum for the second skip connection, then send a separately normalized copy into the feed-forward network.",
      why:"The first addition combines earlier features with information gathered across tokens. The second RMSNorm has its own learned weights because its input already includes the mixer’s contribution.",
      shape:"Both operands and their sum are [1, T, 5120]. The normalization changes the values and keeps the shape.",
      code:"after_mixer = skip + mixer_update\nmlp_input = block.post_attention_layernorm(after_mixer)",
      source:[898,908],
      nodes:[h("skip","Original input",0,0,"From this block’s input skip path."),
        h("update","Mixer update",0,1,"Output of this block’s token mixer."),
        h("sum","Residual sum",1,0,"The intermediate residual stream.", "work"),
        h("u","MLP input",2,1,"The normalized sum consumed by both MLP projections.","output")],
      edges:[["skip","sum"],["update","sum"],["sum","u"]]
    };
  }
  function mlp(recurrent) {
    return {
      id:recurrent?"rec-mlp":"full-mlp",title:"Build new features for each token",
      action:"Two independent projections of the same normalized input each produce 17,408 features. SiLU smooths the gate, and multiplication lets it select which features from the up projection pass through. A third matrix returns the result to 5,120 features for the second residual addition.",
      why:"The mixer let one token use information from the past. This gated MLP transforms each position’s own features. All positions share its weights. The two projections can be computed independently, but their product needs both results.",
      shape:"Two [1, T, 17408] branches multiply element by element. The down projection produces [1, T, 5120], matching the residual. Each block has its own three MLP matrices.",
      code:"gate = silu(mlp.gate_proj(mlp_input))\nup = mlp.up_proj(mlp_input)\nupdate = mlp.down_proj(gate * up)\nx = after_mixer + update",
      source:[818,831],nodes:[h("x","MLP input",0,0,"The post-mixer normalized stream."),
        node("gate","SiLU gate",[1,"T",17408],"work",1,0,"SiLU(a) = a × sigmoid(a)."),
        node("up","Up projection",[1,"T",17408],"work",1,1,"Candidate features from a second independent matrix."),
        node("mul","Gated product",[1,"T",17408],"work",2,0,"The gate acts separately on each intermediate feature."),
        h("skip","Residual sum",2,1,"The stream saved just before the MLP.", "input"),
        h("out","Block output",3,0,"The down-projected update plus the residual stream.","output")],
      edges:[["x","gate"],["x","up"],["gate","mul"],["up","mul"],["mul","out"],["skip","out"]]
    };
  }
  const token = {
    id:"tokens",title:"Look up one vector per token",
    action:"The tokenizer turns the prompt into 16 integer IDs. The embedding table supplies one learned row for each ID. During the cached decode the caller supplies just ID 561, the token “ The”, at position 16 counting from zero.",
    why:"An ID identifies a vocabulary entry. Its learned vector gives the network numerical features to transform. Position IDs are prepared separately. Full-attention blocks use them when rotating queries and keys.",
    shape:"Read a tensor shape from left to right. [1, 16, 5120] means one sequence, 16 token positions and 5,120 features at each position. The table is [248320, 5120]. Prefill reads 16 rows into [1, 16, 5120]. The next call reads one row into [1, 1, 5120].",
    code:"x = text.embed_tokens(input_ids)\npositions = past_length + arange(input_ids.shape[1])\nrotary = text.rotary_emb(x, position_ids)",
    source:[1236,1278],nodes:[node("ids","Token IDs",[1,"T"],"input",0,0,"Indices from the pinned tokenizer. Each element is an integer."),
      node("table","Embedding table",[248320,5120],"weight",0,1,"The model reads the selected rows of this learned matrix."),
      h("x","Token vectors",2,0,"One row of 5,120 working features per current token.","output")],
    edges:[["ids","x"],["table","x"]]
  };
  const fullProject = {
    id:"full-project",title:"Project queries, keys, values and a gate",
    action:"The query projection emits 12,288 numbers per token. Reshape to 24 groups of 512 and split each group into a 256-feature query and a 256-feature output gate. Two smaller projections each emit 1,024 numbers, forming four key heads and four value heads.",
    why:"A query asks which earlier positions matter. A key describes what a position can match. Its value carries the information to retrieve. Six query heads share each KV head, letting them ask different questions while retaining one copy of that group’s history.",
    shape:"After reshaping, Q is [1, 24, T, 256]. New K and V are each [1, 4, T, 256]. The ungated readout will have 24 × 256 = 6,144 features. Head widths here are explicit model settings.",
    code:"paired = attn.q_proj(u).reshape(1, T, 24, 512)\nquery, gate = paired.chunk(2, dim=-1)\nkey = attn.k_proj(u).reshape(1, T, 4, 256)\nvalue = attn.v_proj(u).reshape(1, T, 4, 256)",
    source:[742,789],nodes:[h("u","Normalized input",0,1,"Each of the learned projections reads the same input."),
      node("paired","Q + gate",[1,"T",24,512],"work",1,0,"The split is inside each head."),
      node("q","Queries",q,"output",2,0,"Shown in the head-first axis order used for attention."),
      node("gate","Output gate",[1,"T",6144],"work",2,1,"Saved for after the value mixture."),
      node("k","New keys",[1,4,"T",256],"output",1,1,"Four groups of match features."),
      node("v","New values",[1,4,"T",256],"output",1,2,"Four groups of retrievable content.")],
    edges:[["u","paired"],["paired","q"],["paired","gate"],["u","k"],["u","v"]]
  };
  const fullCache = {
    id:"full-cache",title:"Encode position and append to the KV cache",
    action:"Normalize each query and key across its 256 features. Rotary position embeddings rotate 64 of those features in pairs, leaving 192 untouched. Append each new rotated key and its corresponding value to this block’s own cache.",
    why:"A shared rotation rule makes a query–key dot product sensitive to relative position. The old keys already contain their positions, so the next step can reuse them. Queries and output gates are needed only for the current calculation.",
    shape:"For decode, each of K and V grows from [1, 4, 16, 256] to [1, 4, 17, 256]. For prefill the cache starts empty. The stored keys and values are BF16, two bytes per element.",
    code:"q = attn.q_norm(query).transpose(1, 2)\nk = attn.k_norm(key).transpose(1, 2)\nq, k = apply_rotary_pos_emb(q, k, cos, sin)\nK, V = cache.update(k, value.transpose(1, 2), block_id)",
    source:[780,798],nodes:[node("q","New queries",q,"input",0,0,"Current Q after per-head normalization."),
      node("k","New keys",[1,4,"T",256],"input",0,1,"New keys after per-head normalization."),
      node("v","New values",[1,4,"T",256],"input",0,2,"Values are stored without rotary encoding."),
      node("qr","Rotated Q",q,"output",2,0,"64 of 256 coordinates carry rotary encoding."),
      node("K","Key cache",kv,"state",2,1,"Rotated new keys appended to the retained history."),
      node("V","Value cache",kv,"state",2,2,"New values appended in the same token order.")],
    edges:[["q","qr"],["k","K"],["v","V"]]
  };
  const fullScore = {
    id:"full-score",title:"Score earlier positions and mix their values",
    action:"For each query, take its dot product with every visible key in its group. Divide by √256 = 16 and apply the causal mask. Softmax turns each row into weights summing to one. Multiply those weights by the values and sum over positions.",
    why:"A full-attention head can assign an individual earlier token a large weight. During prefill, each query sees its own position and earlier ones. During decode, the single new query can use all 17 positions. The cache saves recomputing old keys and values, while the new query still has to score the visible keys.",
    shape:"Score grid [1, 24, T, L]. Prefill gives 16 × 16 scores per head before masking. Decode gives 1 × 17 per head. The resulting value mixture is [1, 24, T, 256].",
    code:"group = query_head // 6\nscores = (q[:, query_head] @ K[:, group].transpose(-1, -2)) / 16\np = softmax((scores + causal_mask).float(), dim=-1).to(q.dtype)\nmixed = p @ V[:, group]",
    source:[717,739],nodes:[node("q","Query",q,"input",0,0,"Each query is a 256-feature vector."),
      node("k","Key cache",kv,"state",0,1,"Select the KV head shared by six query heads."),
      node("s","Dot products",scores,"work",1,0,"One number per query and cached position."),
      node("p","Softmax weights",scores,"work",2,0,"A distribution over visible positions for each query."),
      node("v","Value cache",kv,"state",2,1,"Each position supplies a 256-feature value vector."),
      node("mix","Value mixture",q,"output",3,0,"The weighted sum over L gives one vector per query.")],
    edges:[["q","s"],["k","s"],["s","p"],["p","mix"],["v","mix"]]
  };
  const fullOut = {
    id:"full-output",title:"Gate the attention result and rejoin the stream",
    action:"Move the head axis next to the feature axis and concatenate the 24 results. Apply sigmoid to the output gate saved from the query projection, multiply element by element, then project the 6,144 features back to 5,120.",
    why:"The attention weights choose which positions to read. This separate gate lets the current token control how much of each retrieved feature enters its update. The output projection mixes information across heads before the residual addition.",
    shape:"[1, 24, T, 256] becomes [1, T, 6144]. The output matrix is [5120, 6144], giving a block-width update [1, T, 5120].",
    code:"joined = mixture.transpose(1, 2).reshape(1, T, 6144)\ngated = joined * sigmoid(gate.reshape(1, T, 6144))\nmixer_update = attn.o_proj(gated)",
    source:[802,815],
    nodes:[node("heads","Head outputs",q,"input",0,0,"All 24 heads have completed their value mixture."),
      node("gate","Saved gate",[1,"T",6144],"input",0,1,"Produced earlier by the paired query projection."),
      node("gated","Gated features",[1,"T",6144],"work",1,0,"Each sigmoid gate scales one output feature."),
      node("w","Output weights",[5120,6144],"weight",1,1,"Learned mapping from concatenated heads to the residual width."),
      h("out","Mixer update",2,0,"Ready for the first residual addition.","output")],
    edges:[["heads","gated"],["gate","gated"],["gated","out"],["w","out"]]
  };
  const logits = {
    id:"logits",title:"Turn the last position into vocabulary scores",
    action:"Blocks 4 through 63 repeat the same three-recurrent, one-full pattern with distinct weights. The final RMSNorm rescales the last block’s output. This run requests logits for only the final current position, then a separate vocabulary matrix scores all 248,320 possible next-token IDs.",
    why:"The backbone works at width 5,120 regardless of vocabulary size. The output head maps those features back to token choices. The experiment fixes the next input so every tool sees the same cache. An ordinary generator could instead pick or sample a token from the scores.",
    shape:"[1, T, 5120] → [1, 1, 5120] → [1, 1, 248320]. The untied output matrix has shape [248320, 5120]. A logit is an unnormalized score.",
    code:"final = text.norm(x)\nlast = final[:, -1:, :]\nlogits = model.lm_head(last)\ntop_id = logits[0, -1].argmax()",
    source:[1848,1853],nodes:[h("x","Block 63 output",0,0,"The final decoder block has already added its MLP update."),
      node("last","Last position",[1,1,5120],"work",1,0,"Final RMSNorm followed by the requested position slice."),
      node("w","Vocabulary head",[248320,5120],"weight",1,1,"Separate learned weights for next-token scoring."),
      node("scores","Logits",[1,1,248320],"output",2,0,"One score for each possible next token.")],
    edges:[["x","last"],["last","scores"],["w","scores"]]
  };
  const original={
    tokens:[1236,"self.embed_tokens(input_ids)"],
    "rec-open":[880,"self.input_layernorm(hidden_states)"],
    "rec-project":[565,"self.in_proj_qkv(hidden_states)"],
    "rec-conv":[579,"self.conv1d.weight.squeeze(1)"],
    "rec-update":[484,"(v_t - kv_mem) * beta_t"],
    "rec-readout":[652,"self.norm(core_attn_out, z)"],
    "rec-skip":[901,"residual + hidden_states"],
    "rec-mlp":[830,"self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))"],
    "full-project":[786,"self.q_norm(query_states.view(hidden_shape))"],
    "full-cache":[794,"cache.update(key_states, value_states, self.layer_idx, kwargs=kwargs)"],
    "full-score":[736,"torch.matmul(attn_weights, value_states)"],
    "full-output":[812,"torch.sigmoid(gate)"],
    logits:[1851,"self.lm_head(hidden_states)"]
  };
  const stages=[token,entry(true),...window.QWEN_RECURRENT,join(true),mlp(true),
    entry(false),fullProject,fullCache,fullScore,fullOut,join(false),mlp(false),logits];
  for(const s of stages) {
    const ref=original[s.id]||original[s.id.replace("full-","rec-")];
    [s.actualLine,s.actual]=ref;
  }
  window.QWEN_CHAPTERS=stages;
})();
