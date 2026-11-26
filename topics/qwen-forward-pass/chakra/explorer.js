"use strict";
(() => {
  const data = window.chakraStudy;
  const byId = id => document.getElementById(id);
  if (!data) {
    byId("diagram").textContent = "The saved trace data could not load. Keep study-data.js beside this page.";
    return;
  }
  const replay = data.replay, rows = replay.rows;
  const number = (value, digits=0) => value.toLocaleString("en-US", {maximumFractionDigits:digits});
  const shape = t => t.length ? t.join(" × ") : "scalar";
  const explain = [
    "Each cached key belongs to six query heads. repeat_interleave expands four KV heads into 24. Open the nested calls to find the clone and copy_ that materialize the expanded representation.",
    "The same grouping expands cached values. This branch first meets the score branch at Mix values. No top-level tensor producer connects Expand keys to Expand values.",
    "Transpose exchanges the position and feature axes so the score multiply can contract 256 features per query. Inspect the result’s stride and storage ID to see the view.",
    "Each query takes a dot product with the key at each of 17 positions. The two inputs are the captured query and the transposed keys. The output holds 408 scores.",
    "Multiplying by 1/√256 scales every score by 0.0625 before normalization. PyTorch represents the scalar operand in this trace.",
    "Softmax normalizes the 17 scaled scores for each head. Each row of the output is one distribution over the cached positions.",
    "The final matmul weights all 17 value vectors by the corresponding probabilities. Its two producers meet here. The result has 256 features for each of the 24 query heads."
  ];
  const point = [[8,24],[8,129],[138,24],[268,24],[398,24],[528,24],[658,24]];
  let step=6, view="lineage", phase="decode", blockIndex=3;
  function el(tag, cls, txt) {
    const n=document.createElement(tag); if (cls) n.className=cls; if (txt !== undefined) n.textContent=txt; return n;
  }
  function press(id, state) { byId(id).setAttribute("aria-pressed",String(state)); }
  function choose(i) {
    step = i;
    byId("child-filter").value="";
    draw();
  }
  function stepButton(row, i, cls) {
    const b=el("button",cls,row.label);
    b.setAttribute("aria-pressed",String(i===step)); b.onclick=()=>choose(i);
    return b;
  }
  function drawGraph(diagram) {
    const graph=el("div","graph");
    const ns="http://www.w3.org/2000/svg", svg=document.createElementNS(ns,"svg");
    svg.setAttribute("viewBox","0 0 780 208"); svg.setAttribute("aria-hidden","true");
    rows.forEach((row,j)=>row.deps.forEach(id=>{
      const i=rows.findIndex(r=>r.id===id), [sx,sy]=point[i], [tx,ty]=point[j];
      const x=sx+115, y=sy+27;
      const path=document.createElementNS(ns,"path");
      let d = "M"+x+" "+y+" H"+tx;
      if(sy!==ty) d = "M"+x+" "+y+" H"+(tx+60)+" V"+(ty+58);
      path.setAttribute("d",d);path.setAttribute("fill","none");
      path.setAttribute("stroke",step===j?"#07526c":"#7798a4"); path.setAttribute("stroke-width",step===j?"3":"1.8");
      svg.append(path);
      const arrow=document.createElementNS(ns,"path");
      arrow.setAttribute("d",sy===ty ? "M"+(tx-6)+" "+(y-4)+" l6 4 -6 4 z" : "M"+(tx+56)+" "+(ty+59)+" l4 -6 4 6 z");
      arrow.setAttribute("fill","#376575");svg.append(arrow);
    }));
    graph.append(svg);
    rows.forEach((row,i)=>{
      const b=stepButton(row,i,"node");
      b.style.left=point[i][0]+"px";b.style.top=point[i][1]+"px";
      b.append(el("span","",row.name.replace("aten::","")));
      graph.append(b);
    });
    diagram.append(graph);
    byId("diagram-caption").textContent="Arrows connect matching output and input tensor IDs between siblings. Select either branch, or start at Mix values and walk backward using Tensor producers.";
  }
  function drawTimeline(diagram) {
    const timeline=el("div","timeline");
    rows.forEach((r,i)=>{
      const b=stepButton(r,i,"");const track=el("span","track"),bar=el("i");
      bar.style.left=(100*r.start/replay.duration)+"%";bar.style.width=(100*r.dur/replay.duration)+"%";
      track.append(bar);b.append(track);b.title=number(r.start,3)+" to "+number(r.start+r.dur,3)+" µs";timeline.append(b);
    });
    diagram.append(timeline);
    byId("diagram-caption").textContent="All rows share one linear axis from 0 to "+number(replay.duration,3)+" µs. The bars show inclusive intervals from the same Kineto CPU track. Blank space includes uninstrumented work and recording overhead.";
  }
  function tensorColumn(label, items) {
    const col=el("div"); col.append(el("h4","",label));
    items.forEach(t=>{
      const n=el("div","tensor");n.append(el("b","",shape(t.shape)+" · "+number(t.bytes)+" B"),
        el("span","","Tensor "+t.id+" · storage "+t.storage+" · offset "+t.offset),
        el("span","",t.dtype+" · strides ["+t.strides.join(", ")+"]"));
      col.append(n);
    });
    return col;
  }
  function relations(id, relatives, empty) {
    const container=byId(id);container.replaceChildren();
    if (!relatives.length) container.append(el("span","caption",empty));
    relatives.forEach(r=>{
      const b=el("button","",r.label);b.onclick=()=>choose(rows.indexOf(r));container.append(b);
    });
  }
  function drawChildren() {
    const q=byId("child-filter").value.toLowerCase();
    const children=rows[step].nested.filter(c=>c.name.toLowerCase().includes(q)||String(c.id)===q);
    byId("child-count").textContent=number(children.length)+" matching calls.";
    const body=byId("child-rows");body.replaceChildren();
    for(const c of children) {
      const tr=el("tr");[c.id,c.parent,c.name,number(c.start,3),number(c.dur,3)].forEach(t=>tr.append(el("td","",t)));body.append(tr);
    }
  }
  function draw() {
    const r=rows[step], diagram=byId("diagram");diagram.replaceChildren();
    if(view==="lineage") drawGraph(diagram); else drawTimeline(diagram);
    press("lineage",view==="lineage");press("intervals",view==="intervals");
    byId("previous").disabled=step===0;byId("next").disabled=step===rows.length-1;
    byId("position").textContent=(step+1)+" of "+rows.length;
    byId("operator").textContent=r.name;
    byId("op-heading").textContent=r.label;
    byId("op-id").textContent="ET "+r.id+" · RecordFunction "+r.rf;
    byId("explanation").textContent=explain[step];
    byId("tensors").replaceChildren(tensorColumn("Inputs",r.inputs),tensorColumn("Outputs",r.outputs));
    relations("producers",rows.filter(p=>r.deps.includes(p.id)),"Supplied to the replay.");
    relations("consumers",rows.filter(c=>c.deps.includes(r.id)),"Returned by the replay.");
    byId("nested-info").textContent="ET "+r.id+" is inside replay region "+r.parent+". Kineto track "+r.tid+" starts this operation at "+number(r.start,3)+" µs. Its inclusive duration is "+number(r.dur,3)+" µs and its estimated self time is "+number(r.self_us,3)+" µs. The rows below include all recorded descendants.";
    drawChildren();
  }
  function drawPhase(){
    const f=data.full[phase], b=f.blocks.find(x=>x.index===blockIndex)||f.blocks[0];
    press("full-prefill",phase==="prefill");press("full-decode",phase==="decode");
    byId("phase-evidence").textContent="Chakra read "+number(f.host_nodes)+" retained nodes from a full host trace of "+number(f.source_host_nodes)+" nodes. RecordFunction IDs and names matched "+number(f.matched)+" to Kineto intervals across "+f.cpu_tracks+" visible CPU operator track"+(f.cpu_tracks===1?"":"s")+". This instrumented phase spanned "+number(f.duration/1000,1)+" ms. A separate hook-timed run used "+number(f.cpu_us/1000,1)+" process CPU ms over "+number(f.wall_us/1000,1)+" wall ms, averaging "+number(f.cpu_us/f.wall_us,2)+" CPU cores across the process. The run configured "+f.threads+" intra-op threads. Worker-level scheduling is outside the operator trace.";
    const selector=byId("block");selector.replaceChildren();
    for(const entry of f.blocks) {const o=el("option","","Block "+entry.index);o.value=entry.index;selector.append(o);}
    selector.value=String(b.index);
    byId("block-title").textContent="Block "+b.index+" · "+(b.index%4===3?"Full attention":"Recurrent attention");
    byId("block-detail").textContent="ET "+b.id+" · parent "+b.parent+" · RecordFunction "+b.rf;
    byId("block-note").textContent="This block starts at "+number(b.start/1000,2)+" ms and spans "+number(b.dur/1000,2)+" ms. Its region contains "+number(b.ops)+" retained descendant calls, with "+number(b.matched)+" matched CPU intervals.";
    const hot=byId("hot");hot.replaceChildren();
    for(const op of b.hot){const tr=el("tr");[op.name,number(op.count),number(op.inclusive_us,2),op.sample,
      op.shapes.map(shape).join(" / ")||"No direct tensor input"].forEach(t=>tr.append(el("td","",t)));hot.append(tr);}
  }
  const first=rows[0],second=rows[1];
  const gap=second.start-(first.start+first.dur);
  byId("concurrency").textContent="Expand keys and Expand values have independent tensor inputs. The replay recorded them on "+(first.tid===second.tid?"the same host track":"different host tracks")+". The key expansion ends at "+number(first.start+first.dur,3)+" µs and the value expansion starts at "+number(second.start,3)+" µs, "+(gap>=0?number(gap,3)+" µs later":number(-gap,3)+" µs before the first finishes")+". An optimization could schedule these branches independently if resources and memory side effects permit.";
  const n=replay.native;
  byId("native-report").textContent=(n.link_exit===0?"Chakra’s trace linker produced the paired input.":"The upstream trace linker exited with status "+n.link_exit+" on this CPU capture. HTA required a kernel queue-length series to construct its graph. The converter read the original PyTorch host trace directly.")+" The native protobuf contains "+number(n.nodes)+" nodes, "+number(n.order_edges)+" replay-order edges and "+number(n.timed_nodes)+" nodes with a positive encoded duration. Our browser view joins exact RecordFunction IDs and names to the original Kineto CPU intervals.";
  byId("check").textContent="The check compared "+number(replay.check.values_compared)+" FP32 output values against the shared fixture. Maximum absolute difference was "+replay.check.max_abs_error+". The capture used PyTorch "+replay.check.torch+" with one intra-op thread. The raw host trace has "+number(replay.host_nodes)+" nodes and "+number(replay.matched)+" matched Kineto intervals.";
  byId("previous").onclick=()=>choose(step-1);byId("next").onclick=()=>choose(step+1);
  for(const mode of ["lineage","intervals"])byId(mode).onclick=()=>{view=mode;draw();};
  for(const ph of ["prefill","decode"])byId("full-"+ph).onclick=()=>{phase=ph;drawPhase();};
  byId("block").onchange=e=>{blockIndex=Number(e.target.value);drawPhase();};
  byId("child-filter").oninput=drawChildren;
  draw();drawPhase();
})();
