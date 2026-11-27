/* Parameter, FLOP and byte counts quoted in Article C, recomputed from the model's config. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const facts = JSON.parse(fs.readFileSync(path.join(__dirname, '../../assets/facts.json'), 'utf8'));
const v = id => facts[id].value;
const D = v('q3.hidden'), L = v('q3.layers'), H = v('q3.heads'), KV = v('q3.kv_heads'), HD = v('q3.head_dim'), FF = v('q3.ffn'), V = v('q3.vocab');
const BF16 = 2, T = 5;

test('parameters: attention 6,291,456 and feed-forward 9,437,184 per layer; 596,049,920 in all, 155,582,464 of them the tied embedding', () => {
  const attn = D * H * HD + 2 * D * KV * HD + H * HD * D, ffn = 3 * D * FF, norms = 2 * D + 2 * HD, perLayer = attn + ffn + norms;
  assert.equal(attn, 6291456); assert.equal(ffn, 9437184); assert.equal(perLayer, 15730944);
  const embed = V * D, total = L * perLayer + embed + D;
  assert.equal(embed, 155582464); assert.equal(L * perLayer, 440466432); assert.equal(total, 596049920);
});
test('the parameter count equals the sum of tensor sizes in the downloaded checkpoint, when it is present', t => {
  const f = path.join(__dirname, '../../../../.local/walkthrough/qwen3/model.safetensors');
  if (!fs.existsSync(f)) return t.skip('checkpoint not present');
  const fd = fs.openSync(f, 'r'), len = Buffer.alloc(8); fs.readSync(fd, len, 0, 8, 0);
  const n = Number(len.readBigUInt64LE()), hb = Buffer.alloc(n); fs.readSync(fd, hb, 0, n, 8); fs.closeSync(fd);
  const header = JSON.parse(hb.toString());
  const total = Object.entries(header).filter(([k]) => k !== '__metadata__').reduce((a, [, e]) => a + e.shape.reduce((x, y) => x * y, 1), 0);
  assert.equal(total, 596049920, 'tied embeddings are stored once');
});
test('KV cache bytes per token per layer: MHA 8,192, GQA 4,096, MQA 512; GQA over 28 layers is 114,688 per token and 573,440 for 5 tokens', () => {
  const per = heads => 2 * heads * HD * BF16;
  assert.equal(per(H), 8192); assert.equal(per(KV), 4096); assert.equal(per(1), 512);
  assert.equal(L * per(KV), 114688); assert.equal(T * L * per(KV), 573440); assert.equal(L * per(H), 229376);
});
test('prefill FLOPs for 5 tokens: 62,914,560 attention projections and 94,371,840 feed-forward per layer, 4.72 GFLOPs with the head on the last token', () => {
  const attnProj = 2 * T * (D * H * HD + 2 * D * KV * HD + H * HD * D), ffn = 2 * T * 3 * D * FF, head = 2 * D * V;
  assert.equal(attnProj, 62914560); assert.equal(ffn, 94371840); assert.equal(head, 311164928);
  const scoreMix = 2 * 2 * H * T * T * HD;                       // q.k and weights.v, full square counted
  assert.equal(scoreMix, 204800);
  const total = L * (attnProj + ffn + scoreMix) + head;
  assert.equal(Math.round(total / 1e7) / 100, 4.72);
});
test('one decode step reads every weight once: 1.19 GB, about 1 FLOP per byte; the cache is tiny at 6 tokens and 470 MB at 4,096', () => {
  const params = 596049920, weightBytes = params * BF16, flops = 2 * (L * (D * H * HD + 2 * D * KV * HD + H * HD * D + 3 * D * FF)) + 2 * D * V + L * 2 * 2 * H * 6 * HD;
  assert.equal(weightBytes, 1192099840);
  assert.ok(Math.abs(flops / weightBytes - 1) < 0.01, 'FLOPs per weight byte: ' + flops / weightBytes);
  assert.equal(6 * L * 2 * KV * HD * BF16, 688128); assert.equal(4096 * L * 2 * KV * HD * BF16, 469762048);
});
test('MLA: (512 + 64) x 2 = 1,152 bytes against 16 x (128 + 64 + 128) x 2 = 10,240, a ratio of 8.89', () => {
  const kvr = v('ds.kv_rank'), rope = v('ds.rope'), nope = v('ds.nope'), vd = v('ds.vdim'), heads = v('ds.heads');
  assert.equal((kvr + rope) * BF16, 1152); assert.equal(heads * (nope + rope + vd) * BF16, 10240);
  assert.equal(Math.round(10240 / 1152 * 100) / 100, 8.89);
  assert.equal(v('mla.bytes_mla'), 1152); assert.equal(v('mla.bytes_mha'), 10240);
  assert.equal(27 * 1152, 31104);
});
test('GQA versus MLA at 32,768 tokens on one layer: 134 MB against 37.7 MB for the MLA latent', () => {
  assert.equal(32768 * 4096, 134217728); assert.equal(32768 * 1152, 37748736);
});
