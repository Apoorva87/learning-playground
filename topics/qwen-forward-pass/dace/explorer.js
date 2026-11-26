"use strict";
const run = window.DACE_REPLAY;
const byId = id => document.getElementById(id);
const integer = new Intl.NumberFormat("en-US");
const num = value => value === 0 ? "0" : Number(value.toPrecision(6)).toString();
const states = [
  {
    title:"Scaled dot", description:"Each query head compares its 256 features with one key at each of 17 positions. Six neighboring query heads share a KV head.",
    formula:"scores[h,t] = dot(q[h,:], k[⌊h/6⌋,t,:]) / 16",
    ordering:"Each of the 408 head–position pairs writes a different score. Within a pair, one accumulator combines 256 products.",
    plot:"Scaled scores", values:h => run.head_values.scores[h],
    detail:(h,t) => "score["+h+","+t+"] = "+num(run.head_values.scores[h][t])
  },
  {
    title:"Row maximum", description:"Each head finds its largest score across all 17 cached positions. Subtracting that number makes the largest exponent zero.",
    formula:"peak[h] = max(scores[h,0:17])",
    ordering:"The 24 heads can reduce independently. Each reduction compares all 17 scores before the maximum is available.",
    plot:"Candidate scores for the row maximum", values:h => run.head_values.scores[h],
    detail:(h,t) => "score = "+num(run.head_values.scores[h][t])+" · peak = "+num(run.head_values.peak[h])+" at token "+run.head_values.scores[h].indexOf(run.head_values.peak[h])
  },
  {
    title:"Shift and exp", description:"Each position subtracts its head’s maximum and exponentiates the shifted score. Every resulting weight lies between zero and one.",
    formula:"exps[h,t] = exp(scores[h,t] − peak[h])",
    ordering:"All 17 positions of a head read the same maximum. Each pair can write its unnormalized weight independently.",
    plot:"Unnormalized weights", values:h => run.head_values.exps[h],
    detail:(h,t) => "exp("+num(run.head_values.scores[h][t])+" − "+num(run.head_values.peak[h])+") = "+num(run.head_values.exps[h][t])
  },
  {
    title:"Row sum", description:"Each head sums its 17 unnormalized weights. This denominator makes the next state’s probabilities sum to one.",
    formula:"denom[h] = sum(exps[h,0:17])",
    ordering:"One accumulator visits 17 positions per head. Normalization consumes this completed row sum.",
    plot:"Weights entering the sum", values:h => run.head_values.exps[h],
    detail:(h,t) => "weight["+t+"] = "+num(run.head_values.exps[h][t])+" · sum over 17 = "+num(run.head_values.denom[h])
  },
  {
    title:"Normalize", description:"Each query head divides its weights by its own denominator. Token position 16 is the just-appended key, which this decode query can already attend to.",
    formula:"probs[h,t] = exps[h,t] / denom[h]",
    ordering:"Each division reads one weight and the shared denominator for that head. The next state consumes the whole probability row.",
    plot:"Attention probabilities", values:h => run.head_values.probs[h],
    detail:(h,t) => num(run.head_values.exps[h][t])+" / "+num(run.head_values.denom[h])+" = "+num(run.head_values.probs[h][t])
  },
  {
    title:"Value mix", description:"For each output feature, the head weights 17 cached values by its attention probabilities. The inspector shows the contributions for feature zero.",
    formula:"out[h,d] = Σₜ probs[h,t] × v[⌊h/6⌋,t,d]",
    ordering:"The 6,144 head–feature pairs write distinct outputs. Within each pair, 17 tokens reduce to one value. Moving one token in V advances 256 FP32 elements.",
    plot:"Signed contributions to output feature zero",
    values:h => run.head_values.probs[h].map((p,t) => p * run.value_feature0[Math.floor(h/6)][t]),
    detail:(h,t) => "p × v = "+num(run.head_values.probs[h][t])+" × "+num(run.value_feature0[Math.floor(h/6)][t])+" · sum over 17 = "+num(run.output_feature0[h])
  }
];
const selectHead = byId("head");
const selectPosition = byId("position");
const stageButtons = [], plotButtons = [];
let active = 0;
const sumBy = (array, fn) => array.reduce((acc,item) => acc + fn(item),0);
function fillOptions(select, count) {
  for (let i=0; i<count; i++) select.add(new Option(String(i),String(i)));
}
function selectState(i) {
  active = i;
  const stage = run.stages[i], lesson = states[i];
  stageButtons.forEach((b,j) => b.setAttribute("aria-pressed", String(i===j)));
  byId("stage-id").textContent = stage.name;
  byId("stage-title").textContent = lesson.title;
  byId("workers").textContent = integer.format(stage.iterations)+" map iterations";
  byId("explanation").textContent = lesson.description;
  byId("formula").textContent = lesson.formula;
  for (const kind of ["reads","writes"]) {
    const edgeList = byId(kind);
    edgeList.replaceChildren();
    for (const edge of stage[kind]) {
      const para = document.createElement("p"), name = document.createElement("code"), volume = document.createElement("span");
      para.className = "memlet";
      name.textContent = edge.array+"["+edge.subset+"]";
      volume.textContent = integer.format(edge.elements_per_iteration)+" FP32 per iteration · "+integer.format(edge.bytes)+" B across map";
      para.append(name,volume); edgeList.append(para);
    }
  }
  const eventTime=stage.trace_us===0 ? "Both timer samples fell in the same 1 µs interval." : "A separate native timer run recorded "+stage.trace_us+" µs for this state (1 µs resolution).";
  byId("dependency").textContent = lesson.ordering+" "+eventTime;
  byId("code").textContent = stage.code;
  drawValues();
}
function drawValues() {
  const h=Number(selectHead.value), t=Number(selectPosition.value), g=Math.floor(h/6), lesson=states[active];
  const values=lesson.values(h), high=Math.max(0,...values), low=Math.min(0,...values), extent=high-low || 1;
  byId("group").textContent = "Q["+h+"] → KV["+g+"] · query heads "+g*6+"–"+(g*6+5)+" share this cache";
  byId("chart-note").textContent="Choose a query head and a token position to inspect the selected state on the checkpoint fixture.";
  byId("chart-title").textContent=lesson.plot+" · head "+h;
  plotButtons.forEach((button,i) => {
    const v=values[i], label="Token "+i+", "+num(v);
    button.setAttribute("aria-label",label); button.title=label;
    button.setAttribute("aria-pressed",String(i===t));
    button.classList.toggle("negative",v<0);
    button.style.setProperty("--amount", (Math.abs(v)/extent*88)+"%");
    button.style.setProperty("--base", (6+(Math.min(0,v)-low)/extent*88)+"%");
  });
  byId("detail-value").textContent = lesson.detail(h,t);
}
fillOptions(selectHead,24); fillOptions(selectPosition,17);
for (let i=0; i<17; i++) {
  const b=document.createElement("button"); b.type="button";
  b.addEventListener("click",() => { selectPosition.value=String(i); drawValues(); });
  byId("chart").append(b); plotButtons.push(b);
}
let reads=0, writes=0;
states.forEach((lesson,i) => {
  const stage=run.stages[i], b=document.createElement("button"), seq=document.createElement("span");
  b.type="button"; seq.className="stepnum";seq.textContent=String(i+1).padStart(2,"0");
  b.append(seq,document.createTextNode(lesson.title));
  b.addEventListener("click",() => selectState(i)); byId("stages").append(b);stageButtons.push(b);
  const read=sumBy(stage.reads,edge=>edge.bytes), write=sumBy(stage.writes,edge=>edge.bytes);
  reads+=read;writes+=write;
  const row=byId("counts").insertRow();
  for (const value of [lesson.title,...[stage.iterations,read,write].map(v=>integer.format(v))]) row.insertCell().textContent=value;
});
for (const value of ["Total","",integer.format(reads),integer.format(writes)]) {
  const row=byId("total").rows[0] || byId("total").insertRow();
  row.insertCell().textContent=value;
}
byId("reuse").textContent = "For example, K occupies "+integer.format(run.unique_bytes.k)+" B. The score map requests "+integer.format(run.stages[0].reads.find(edge=>edge.array==="k").bytes)+" B of key elements because six query heads reuse each key. These figures describe this explicit, unfused graph.";
byId("timing").textContent =  "On one "+run.cpu_model+" CPU thread, the six-state replay took a median of "+num(run.wall_us.median)+" µs elapsed per call (10th–90th percentile "+num(run.wall_us.p10)+"–"+num(run.wall_us.p90)+" µs). Median process CPU time was "+num(run.cpu_us_median)+" µs. The "+run.samples+" timed Python calls used preallocated buffers following "+run.warmups+" warmups. Compilation and input loading were outside these calls.";
byId("numerics").textContent = "The compiled output matched the shared FP32 reference over 6,144 values. The maximum absolute difference was "+num(run.max_abs_error)+". The SDFG scheduled every map sequentially. Its independent iterations describe opportunities for parallel scheduling. This experiment used one thread.";
selectHead.addEventListener("change",drawValues);
selectPosition.addEventListener("change",drawValues);
selectState(0);
