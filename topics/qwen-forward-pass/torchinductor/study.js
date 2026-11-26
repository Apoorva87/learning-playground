"use strict";

const runs = window.INDUCTOR_CAPTURE;
const byId = id => document.getElementById(id);
const labels = {attention: "Attention · decode", "mlp-decode": "Gated MLP · decode", "mlp-prefill": "Gated MLP · prefill"};
const format = new Intl.NumberFormat("en-US");
const duration = us => us < 1000 ? `${us.toFixed(2)} µs` : `${(us / 1000).toFixed(2)} ms`;
let selectedCase = "attention";
let step = 0;
let diagramEdges = [];

const steps = {
  attention: [
    {short: "Expand K and V", title: "Write six copies per KV head", text: "The four cached KV heads must line up with 24 query heads. The generated C++ function loops over K, then V. For each output head h, it loads from input head floor(h / 6) and stores into a contiguous 24-head buffer."},
    {short: "CPU flash attention", title: "Score, normalize and mix", text: "One ATen CPU flash-attention call receives Q, expanded K and expanded V. Its scale is 1/16. The Python graph’s score and probability tensors each have 408 FP32 values. The generated wrapper delegates both computations to ATen and allocates neither intermediate itself. It returns the context and drops the log-sum-exp result."}
  ],
  mlp: [
    {short: "Gate projection", title: "Project the gate", text: "The first external matrix multiply maps each 5,120-wide hidden row to 17,408 gate values. The transposed weight is a view with swapped strides. The wrapper sends the original weight storage to the matrix library."},
    {short: "Up projection", title: "Project a second branch", text: "The up projection uses the same hidden state and a different matrix. The scheduler lists no dependency on the gate projection. This wrapper calls it after the gate multiply returns."},
    {short: "SiLU × up", title: "Fuse the activation and product", text: "The generated C++ loop loads eight BF16 gate values and eight up values per vector step. It widens the inputs, evaluates gate / (1 + exp(−gate)) and multiplies by up in FP32. A single BF16 store overwrites the gate buffer. The wrapper gives that storage a new view called buf2."},
    {short: "Down projection", title: "Return to the residual width", text: "The final external matrix multiply reads the gated product and projects 17,408 features back to 5,120. Its output is the value captured at the MLP module boundary, before residual addition."}
  ]
};

function currentSteps() { return steps[selectedCase === "attention" ? "attention" : "mlp"]; }

function payloads() {
  const event = runs[selectedCase].wrapper_calls[step];
  const p = event.positional;
  if (selectedCase === "attention") {
    if (step === 0) return [[["K", p[0]], ["V", p[1]]], [["Expanded K · buf0", p[2]], ["Expanded V · buf1", p[3]]]];
    return [[["Q", p[0]], ["Expanded K · buf0", p[1]], ["Expanded V · buf1", p[2]]],
            [["Context · returned", event.returned[0]], ["Log-sum-exp · dropped", event.returned[1]]]];
  }
  if (step === 2) return [[["Gate · buf2 aliases buf0", p[0]], ["Up · buf1", p[1]]], [["Product · buf2", p[0]]]];
  const names = [["Hidden", "Gate weights", "Gate · buf0"], ["Hidden", "Up weights", "Up · buf1"], [], ["Product · buf2", "Down weights", "Output · buf3"]][step];
  return [[[names[0], p[0]], [names[1], p[1]]], [[names[2], event.named.out]]];
}

function selectStep(index) {
  step = index;
  const details = currentSteps()[step], event = runs[selectedCase].wrapper_calls[step];
  byId("ordinal").textContent = `Call ${step + 1} of ${currentSteps().length}`;
  byId("previous").disabled = step === 0;
  byId("next").disabled = step === currentSteps().length - 1;
  byId("step-title").textContent = details.title;
  byId("explain").textContent = details.text;
  byId("call-time").textContent = `Measured interval on caller thread ${event.thread} · ${duration(event.elapsed_us)}`;
  const lists = payloads();
  ["reads", "writes"].forEach((id, i) => {
    const items = lists[i].map(([name, t]) => {
      const li = document.createElement("li"), figure = document.createElement("small");
      li.append(name);
      figure.textContent = `${t.shape.join(" × ")} · ${format.format(t.bytes)} B`;
      li.append(figure);
      return li;
    });
    byId(id).replaceChildren(...items);
  });
  document.querySelectorAll("[data-step]").forEach(el => el.setAttribute("aria-pressed", String(Number(el.dataset.step) === step)));
  renderCode();
}

function drawEdges() {
  const root = byId("diagram"), svg = root.querySelector("svg");
  if (!svg) return;
  const rect = root.getBoundingClientRect();
  const lines = diagramEdges.map(([a, b]) => {
    const from = byId(a).getBoundingClientRect(), to = byId(b).getBoundingClientRect();
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    const coords = {x1: from.left + from.width / 2 - rect.left, y1: from.bottom - rect.top,
                    x2: to.left + to.width / 2 - rect.left, y2: to.top - rect.top - 1};
    for (const [k,v] of Object.entries(coords)) line.setAttribute(k, String(v));
    return line;
  });
  svg.replaceChildren(...lines);
}

function renderDiagram() {
  const root = byId("diagram"), att = selectedCase === "attention";
  const nodes = att ? [
    ["kv", "K and V", 1, "1", null], ["q", "Query", 1, "2", null],
    ["a", null, 2, "1", 0], ["b", null, 3, "1/3", 1], ["out", "Context", 4, "1/3", null]
  ] : [
    ["hidden", "Hidden state", 1, "1/3", null], ["a", null, 2, "1", 0], ["b", null, 2, "2", 1],
    ["c", null, 3, "1/3", 2], ["d", null, 4, "1/3", 3], ["out", "MLP output", 5, "1/3", null]
  ];
  diagramEdges = att ? [["kv","a"],["a","b"],["q","b"],["b","out"]] :
    [["hidden","a"],["hidden","b"],["a","c"],["b","c"],["c","d"],["d","out"]];
  root.style.gridTemplateRows = att ? "1.6rem 4.8rem 4.8rem 1.6rem" : "1.6rem 4.5rem 4.5rem 4.5rem 1.6rem";
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  root.replaceChildren(svg);
  nodes.forEach(([id,label,row,col,index]) => {
    const el = document.createElement(index === null ? "span" : "button");
    el.id = id;
    el.className = index === null ? "boundary" : "node";
    el.style.gridRow = row;
    el.style.gridColumn = col;
    el.textContent = label || currentSteps()[index].short;
    if (index !== null) {
      el.type = "button";
      el.dataset.step = index;
      el.addEventListener("click", () => selectStep(index));
    }
    root.append(el);
  });
  drawEdges();
}

function renderTimeline() {
  const events = runs[selectedCase].wrapper_calls;
  const first = events[0].start_us, last = events[events.length - 1].end_us, span = last - first;
  const track = document.createElement("div");
  track.className = "track";
  events.forEach((event, i) => {
    const b = document.createElement("button"), name = currentSteps()[i].short;
    b.type = "button"; b.className = "interval"; b.dataset.step = String(i);
    b.style.left = `${100 * (event.start_us - first) / span}%`;
    b.style.width = `${100 * event.elapsed_us / span}%`;
    b.setAttribute("aria-label", `${name}, ${duration(event.elapsed_us)}`);
    b.title = `${name} · ${duration(event.elapsed_us)}`;
    b.addEventListener("click", () => selectStep(i));
    track.append(b);
  });
  byId("lanes").replaceChildren(track);
  byId("trace-span").textContent = "One instrumented execution · wall time";
  byId("trace-end").textContent = duration(span);
}

const filenames = {graph:"dynamo_fx.txt",ir:"ir_post_fusion.txt",runner:"output_code.py",cpp:"output_code.py"};
const descriptions = {
  graph: "TorchDynamo supplied this graph to the Inductor backend. Shapes follow each variable name.",
  ir: "Each scheduler node records which buffers it writes and which earlier results must exist. The pre-fusion and post-fusion IR files were identical for this compilation.",
  runner: "This line from Runner.call dispatched the selected computation. Empty buffers, views and output selection are visible in the full file.",
  cpp: "The C++ function below was emitted by Inductor. Library implementations for external calls are outside the generated wrapper."
};
function renderCode() {
  const source = byId("source").value, data = runs[selectedCase];
  byId("excerpt").textContent = ["runner", "ir"].includes(source) ? data[source][step] : data[source];
  byId("code-note").textContent = descriptions[source];
  byId("full-source").href = `evidence/${selectedCase}/${filenames[source]}`;
}

function renderCase() {
  selectedCase = byId("case").value;
  byId("scope").textContent = selectedCase === "attention" ?
    "One new position queries 17 cached positions. Each group of six query heads shares a single KV head. Inputs and arithmetic in this replay use FP32." :
    `${selectedCase === "mlp-decode" ? "One decode token enters" : "Sixteen prompt tokens enter"} the complete BF16 MLP in the fourth block. Each projection uses a real 170 MiB weight matrix.`;
  byId("dep-note").textContent = selectedCase === "attention" ?
    "The two independent KV copy loops ran successively in one generated function. Their outputs feed the same external attention call." :
    "Gate and up can start from the same hidden state. The product waits for both branches. The trace below reveals their actual execution order.";
  renderDiagram();
  renderTimeline();
  const links = [["Dynamo FX","dynamo_fx.txt"],["Input FX after decomposition","fx_graph_readable.py"],["Transformed FX","fx_graph_transformed.py"],["Pre-fusion IR","ir_pre_fusion.txt"],["Post-fusion IR","ir_post_fusion.txt"],["Generated CPU wrapper","output_code.py"],["Measured run JSON","run.json"]];
  byId("evidence-links").replaceChildren(...links.map(([label,file])=>{
    const link = document.createElement("a"); link.textContent = label; link.href=`evidence/${selectedCase}/${file}`; return link;
  }));
  selectStep(0);
}

for (const [caseId, label] of Object.entries(labels)) {
  const r = runs[caseId], row = byId("summary").insertRow();
  const err = r.compiled_vs_reference, count = err.num_values;
  [label,duration(r.timing.median_wall_us),duration(r.timing.median_process_cpu_us),r.first_compiled_call_s.toFixed(2)+" s",
   err.max_abs.toExponential(3), `${format.format(count - r.check.out_of_tolerance)} / ${format.format(count)}`].forEach(value => {row.insertCell().textContent = value;});
}
byId("case").addEventListener("change", renderCase);
byId("previous").addEventListener("click", () => selectStep(step - 1));
byId("next").addEventListener("click", () => selectStep(step + 1));
byId("source").addEventListener("change", renderCode);
renderCase();
new ResizeObserver(drawEdges).observe(byId("diagram"));
