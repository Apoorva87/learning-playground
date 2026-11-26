/* The articles quote derived numbers (48 visits, 196,608 bytes per token, ...). Recompute them
   from the config values in facts.json and from the layout's own layer mapping. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const L = require('../../engine/layout.js');
const facts = JSON.parse(fs.readFileSync(path.join(__dirname, '../../assets/facts.json'), 'utf8'));
const v = id => facts[id].value;
const BYTES = 2;                                              // BF16 keys and values

test('48 layer visits from 24 layers, core 6-17, 3 passes', () => {
  const visits = v('cfg.core_start') + (v('cfg.core_end') - v('cfg.core_start')) * v('cfg.passes') + (v('cfg.layers') - v('cfg.core_end'));
  assert.equal(visits, 48); assert.equal(L.N_VISITS, 48);
});
test('keys and values grow by 196,608 bytes per token: 48 visits x 2 x 8 KV heads x 128 x 2 B', () => {
  const perToken = L.N_VISITS * 2 * v('cfg.kv_heads') * v('cfg.head_dim') * BYTES;
  assert.equal(perToken, 196608); assert.equal(perToken, v('cap.kv_bytes'));
});
test('beyond the 1,024-token window only the 8 full-attention visits keep growing: 32,768 bytes per token', () => {
  const full = Array.from({ length: L.N_VISITS }, (_, i) => i).filter(i => L.FULL_LAYERS.includes(L.physicalOf(i))).length;
  assert.equal(full, 8);
  assert.equal(full * 2 * v('cfg.kv_heads') * v('cfg.head_dim') * BYTES, 32768);
  assert.equal(L.N_VISITS - full, 40);                         // sliding-window visits, capped at cfg.window positions
});
test('full-attention layers are the complement of local_layer_ids in the config', t => {
  const file = path.join(__dirname, '../../../../.local/inkling2/checkpoint/config.json');
  if (!fs.existsSync(file)) return t.skip('checkpoint config not present');
  const local = new Set(JSON.parse(fs.readFileSync(file, 'utf8')).text_config.local_layer_ids);
  assert.deepEqual(Array.from({ length: 24 }, (_, i) => i).filter(i => !local.has(i)), L.FULL_LAYERS);
});
test('the measured pass times scale like layer visits: 1, 2, 3 passes cost about 24, 36, 48 visits', () => {
  const w = [v('cap.wall1'), v('cap.wall2'), v('cap.wall3')];
  assert.ok(Math.abs(w[2] / w[0] - 48 / 24) < 0.3, `3-pass / 1-pass = ${w[2] / w[0]}`);
  assert.ok(Math.abs(w[1] / w[0] - 36 / 24) < 0.3);
});

/* Article B's byte and parameter counts, from the config shapes. */
const D = () => v('cfg.hidden'), FF = () => v('cfg.ff');
test('one expert has 3 x 2304 x 1152 = 7,962,624 parameters; 128 of them in 24 layers are 24.46B of the 26.4B', () => {
  const expert = 3 * D() * FF();
  assert.equal(expert, 7962624);
  assert.equal(v('cfg.experts') * expert, 1019215872);
  assert.equal(v('cfg.layers') * v('cfg.experts') * expert, 24461180928);
});
test('the loop machinery adds 28,311,552 gate parameters and 110,592 message-query parameters', () => {
  const gate = 2 * D() * v('cfg.gate_rank') + 2 * v('cfg.gate_rank') * D();       // down_proj [2*2304 -> 64], forget [64 -> 2304], message [64 -> 2304]
  assert.equal(gate, 589824);
  assert.equal(gate * 4 * (v('cfg.core_end') - v('cfg.core_start')), 28311552);
  assert.equal((v('cfg.core_end') - v('cfg.core_start')) * 4 * D(), 110592);
});
test('an extra pass costs about 1.08B active parameters by our count, against the 1.0B Thinking Machines reports', () => {
  const qDim = v('cfg.heads') * v('cfg.head_dim'), kvDim = v('cfg.kv_heads') * v('cfg.head_dim');
  const attn = D() * qDim * 3 + D() * kvDim * 2;                 // q, output gate and o have q's size; k and v are kv-sized
  const experts = v('cfg.topk') * 3 * D() * FF(), router = v('cfg.experts') * D();
  const perLayer = attn + experts + router;
  assert.equal(attn, 25952256);
  const extra = (v('cfg.core_end') - v('cfg.core_start')) * perLayer;
  assert.ok(extra > 1.0e9 && extra < 1.2e9, String(extra));
});
test('at the full 1,048,576-token context, keys and values come to 34.5 GB', () => {
  const perVisitPerToken = 2 * v('cfg.kv_heads') * v('cfg.head_dim') * 2;
  assert.equal(perVisitPerToken, 4096);
  const full = 8 * perVisitPerToken * v('cfg.max_len'), sliding = 40 * perVisitPerToken * v('cfg.window');
  assert.equal(full, 34359738368); assert.equal(sliding, 167772160);
  assert.equal(Math.round((full + sliding) / 1e8) / 10, 34.5);
});
test('the drafter taps visits 5, 23 and 41, which are full-attention visits (layers 5, 11, 17), and the final state 48', () => {
  const taps = v('cfg.mtp_targets');
  assert.deepEqual(taps, [5, 23, 41, 48]);
  assert.deepEqual(taps.slice(0, 3).map(L.physicalOf), [5, 11, 17]);
  assert.ok(taps.slice(0, 3).every(t => L.FULL_LAYERS.includes(L.physicalOf(t))));
  assert.equal(taps[3], L.N_VISITS);
});
