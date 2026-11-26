/* Run with Node from any folder after updating the chapter scripts or observed.json. */
const fs = require('node:fs');
const path = require('node:path');
const dir = __dirname;
global.window = {};
require('./recurrent.js');
require('./chapters.js');
const chapters = window.QWEN_CHAPTERS;
const read = name => JSON.parse(fs.readFileSync(path.join(dir,name),'utf8'));
if (process.argv[2]) {
  const source = fs.readFileSync(process.argv[2]);
  const hash = require('node:crypto').createHash('sha1').update('blob '+source.length+'\0').update(source).digest('hex');
  if (hash !== '81c4cd1ecaf982f6001a060068958be1cf6a111f') throw Error('Unexpected Transformers source revision');
  const lines=source.toString().split('\n');
  for (const step of chapters) {
    if (!lines[step.actualLine-1]?.includes(step.actual) || step.actualLine<step.source[0] || step.actualLine>step.source[1])
      throw Error('Source mismatch at '+step.id);
  }
}
const run = read('observed.json');
const manifest = read('../assets/workload.json');
const model = 'https://github.com/huggingface/transformers/blob/eeee11099135592f1d03c2871fbd3d8132e5a4d7/src/transformers/models/qwen3_5/modeling_qwen3_5.py';
const escaped = text => String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = x => Number(x.toPrecision(5)).toString();
function observations(phase) {
  const d=run[phase], r=d.block0, a=d.block3, delta=r.delta_example;
  const row=a.attention_row, pred=d.prediction;
  const v = (b,k) => fmt(b[k].value);
  const answers={
    tokens:`At the final input position, feature 0 of the embedding is ${fmt(d.embedding.value)}. The input ID is ${phase==='prefill'?manifest.prompt_ids.at(-1):manifest.decode_id} (${phase==='prefill'?'the period':'“ The”'}).`,
    'rec-project':`For value head 0, the write fraction β is ${fmt(delta.beta)}. The retention factor exp(g) is ${fmt(delta.decay)}. These are calculated separately for each token and value head.`,
    'rec-conv':`For channel 0, the four saved convolution inputs, oldest to newest, are [${r.conv_channel0_newest_last.map(fmt).join(', ')}]. The current activated output on this channel is ${v(r,'convolution_activated')}.`,
    'rec-update':`Head 0, state cell [0, 0]. The old value ${fmt(delta.old)} decays to ${fmt(delta.decayed)}. The key predicts ${fmt(delta.predicted_value)} for value feature 0, while the observed target is ${fmt(delta.target_value)}. The correction writes ${fmt(delta.write)} to this cell, bringing it to ${fmt(delta.updated)}.`,
    'rec-readout':`Head 0 reads ${v(r,'readout')} at feature 0. After the gated RMSNorm this feature is ${v(r,'gated_readout')}. The output projection combines all head features and produces ${v(r,'mixer_output')} at residual feature 0.`,
    'full-project':`At feature 0 of head 0 the query projection yields ${v(a,'q_proj')}. The associated gate logit is ${v(a,'gate')}. KV head 0 supplies a new key feature of ${v(a,'k_proj')} and a value feature of ${v(a,'v_proj')}.`,
    'full-cache':`Head 0, feature 0. The normalized query is ${v(a,'q_norm')} and rotates to ${v(a,'q_rotated')}. The normalized key ${v(a,'k_norm')} rotates to ${v(a,'k_rotated')}, the value stored at the newest key position. The value cache stores ${v(a,'cache_v')} in that feature.`,
    'full-score':`Inspect the last query position of the actual BF16 forward pass. Select a position on the front of the key cache, value cache or softmax tensor, or use the controls below. The bands show recorded softmax weights for the selected head. They can sum to slightly above or below one after BF16 rounding.`,
    'full-output':`Head 0, feature 0. Attention retrieved ${fmt(row.mixed.value)}. The saved gate has sigmoid ${v(a,'sigmoid_gate')}. Their product rounds to ${v(a,'o_proj_input')} before the output matrix, which combines all 6,144 gated features.`,
    logits:`The highest logit is ${fmt(pred.logit)} for ID ${pred.id}, token “${pred.text}”. ${phase==='prefill'?'The experiment instead supplies “ The”, whose logit was '+fmt(d.supplied_decode_token_logit)+', to make every tool analyze the same decode.':'This is the score distribution after the supplied token “ The”.'}`
  };
  for (const [prefix,b] of [['rec',r],['full',a]]) {
    answers[prefix+'-open']=`At the final current position, the 5,120 input features have RMS ${fmt(b.row_rms_before_input_norm)}. Feature 0 goes from ${v(b,'residual_in')} to ${v(b,'normalized')} after the learned normalization. The skip path keeps ${v(b,'residual_in')}.`;
    answers[prefix+'-skip']=`At feature 0, skip ${v(b,'residual_in')} plus mixer update ${v(b,'mixer_output')} gives ${v(b,'residual_after_mixer')} after BF16 rounding. The normalized value passed to the MLP is ${v(b,'mlp_input')}.`;
    answers[prefix+'-mlp']=`At intermediate feature 0, the activated gate is ${v(b,'mlp_silu')} and the up projection is ${v(b,'mlp_up')}. Their BF16 product is ${v(b,'mlp_product')}. After the down projection and residual addition, block output feature 0 is ${v(b,'residual_out')}.`;
  }
  if (chapters.some(s=>!answers[s.id])) throw Error('Missing measured chapter example');
  return answers;
}
const data={model, tokens:manifest.token_text, prefill:observations('prefill'),decode:observations('decode'),attention:{}};
for (const phase of ['prefill','decode']) {
  const r=run[phase].block3.attention_row;
  const len=phase==='prefill'?16:17;
  for (const rows of [r.weights_by_head,r.scores_by_head]) {
    if (rows?.length!==24 || rows.some(row=>row.length!==len)) throw Error('Incomplete attention data');
  }
  data.attention[phase]={weights:r.weights_by_head,scores:r.scores_by_head};
}
fs.writeFileSync(path.join(dir,'examples.js'),'window.QWEN_EXAMPLES='+JSON.stringify(data)+';\n');
function family(s){return s.id.startsWith('rec-')?'Block 0 · recurrent':s.id.startsWith('full-')?'Block 3 · full attention':s.id==='tokens'?'Text input':'Model output';}
function source(s){return model+'#L'+s.source[0]+'-L'+s.source[1]}
const articles=chapters.map((s,i)=>`<article id="${s.id}"><p class="step-number">${i+1} / ${chapters.length} · ${family(s)}</p><h2>${escaped(s.title)}</h2><p>${escaped(s.action)}</p><h3>Why this operation?</h3><p>${escaped(s.why)}</p><h3>Tensor dimensions</h3><p>${escaped(s.shape)}</p><h3>Adapted Python</h3><pre><code>${escaped(s.code)}</code></pre><p>Model code, line ${s.actualLine} · <code>${escaped(s.actual)}</code></p><p><a href="${source(s)}">Read the model implementation, lines ${s.source[0]}–${s.source[1]}</a></p><h3>During the captured decode</h3><p>${escaped(s.id==='full-score'?'For the most recent position, query head 0 gives the current token a score of '+run.decode.block3.attention_row.scores.value+' and a weight of '+fmt(run.decode.block3.attention_row.weights.value)+'.':data.decode[s.id])}</p><a href="../index.html#${s.id}">Open this step in 3D</a></article>`).join('\n');
const menu=chapters.map(s=>`<li><a href="#${s.id}">${escaped(s.title)} · ${family(s)}</a></li>`).join('');
fs.writeFileSync(path.join(dir,'reading.html'),`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen walkthrough · Reading edition</title><link rel="stylesheet" href="lesson.css"></head><body><main class="reader"><a href="../index.html">← Interactive architecture walkthrough</a><h1>Qwen3.8-27B, one block at a time</h1><p>Read the 16-token prefill or the following one-token decode as a series of shape changes. B = 1 sequence. T is the number of current tokens (16 or 1). L is the number of cached positions after the update (16 or 17). Blocks are numbered from zero. Shortened Python uses the operation order of the pinned Transformers implementation.</p><nav aria-label="Steps"><ol>${menu}</ol></nav>${articles}<footer><a href="../index.html#why-hybrid">Why the model uses both mixers</a> · <a href="../index.html#memory">Compare cache sizes</a> · <a href="../profile.html#timeline">Explore measured execution</a> · <a href="observed.json">Recorded evidence</a></footer></main></body></html>\n`);
