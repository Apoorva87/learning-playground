(() => {
  const A = window.QWEN_ATLAS;
  const el = id => document.getElementById(id);
  if (!A) { el("phase-note").textContent = "The timing data could not load. Open the linked full timing records below."; return; }
  let phase = "decode", layer = 3, head = 0, position = 16, feature = 0, timer;
  const ms = us => us >= 1e6 ? (us/1e6).toFixed(2)+" s" : (us/1000).toFixed(us>=1e4?1:2)+" ms";
  const size = bytes => bytes >= 2**30 ? (bytes/2**30).toFixed(2)+" GiB" : bytes >= 2**20 ? (bytes/2**20).toFixed(2)+" MiB" : (bytes/1024).toFixed(1)+" KiB";
  const linearF = n => n >= 1e12 ? (n/1e12).toFixed(2)+" TFLOP" : (n/1e9).toFixed(1)+" GFLOP";
  const core = (cpu,wall) => (cpu/wall).toFixed(2);
  const fixed = n => Number(n).toPrecision(4);
  const add = (parent,tag,text) => {const node=document.createElement(tag); if(text!==undefined) node.textContent=text; parent.append(node);return node;};
  function rect(svg, x,y,w,h,color,title) {
    const node=document.createElementNS("http://www.w3.org/2000/svg","rect");
    for (const [k,v] of Object.entries({x,y,width:w,height:h,fill:color})) node.setAttribute(k,v);
    const t=document.createElementNS(node.namespaceURI,"title");t.textContent=title;node.append(t);svg.append(node);
  }
  const stages={input_layernorm:"Input norm",post_attention_layernorm:"Post-mixer norm",self_attn:"Full attention",linear_attn:"Recurrent mixer",mlp:"Feed-forward"};
  function drawTimeline() {
    const d=A[phase], full=phase==="prefill";
    for(const key of ["prefill","decode"]) el(key).setAttribute("aria-pressed",key===phase);
    el("phase-note").textContent=full?"Start with an empty cache. Read the 16-token prompt and return the vocabulary scores at its final position.":"Start from the 16-token cache. Supply the fixed next token and compute the following-token distribution.";
    const recurrent=d.blocks.flatMap(b=>b.stages).filter(e=>e.name.endsWith(".linear_attn")).reduce((sum,e)=>sum+e.wall_us,0);
    el("finding").textContent="The 48 recurrent mixers together occupied "+ms(recurrent)+" of the "+ms(d.wall_us)+" pass ("+(100*recurrent/d.wall_us).toFixed(1)+"%). In this CPU reference implementation, a grouped convolution dispatches many small operations. The 16 full-attention blocks are where the conventional key/value cache grows. Explore block 3 to see the difference.";
    const stats=el("phase-stats");stats.replaceChildren();
    for (const [value,label] of [[ms(d.wall_us),"Measured elapsed time"],[core(d.cpu_us,d.wall_us),"Average CPU cores active"],[linearF(d.linear_flops),"Linear projection arithmetic"],[size(d.cache_bytes),"Cache tensor payload after pass"]]) {
      const item=add(stats,"div");add(item,"strong",value);add(item,"span",label);
    }
    const axis=el("axis");axis.replaceChildren();for(let i=0;i<=4;i++)add(axis,"span",ms(d.wall_us*i/4));
    const bars=el("timebars"), cp=el("coreplot"), wp=el("weightplot");bars.replaceChildren();cp.replaceChildren();wp.replaceChildren();
    const cpuMax=Math.max(1,...d.blocks.map(b=>b.cpu_us/b.wall_us));
    const rate=r=>r[2]/Math.max(r[1],1)/1000;
    const rateMax=Math.max(...d.linears.map(rate));
    d.blocks.forEach((b,i)=>{
      const btn=add(bars,"button");const att=A.types[i]==="full_attention";
      btn.className=att?"att":"recur";btn.setAttribute("aria-label","Block "+i+", "+(att?"full attention":"recurrent mixer")+", "+ms(b.wall_us));
      btn.title=btn.getAttribute("aria-label");btn.style.left=100*b.start_us/d.wall_us+"%";btn.style.width=100*b.wall_us/d.wall_us+"%";btn.onclick=()=>selectBlock(i);
      const h=64*(b.cpu_us/b.wall_us)/cpuMax;
      rect(cp,960*b.start_us/d.wall_us,65-h,Math.max(.7,960*b.wall_us/d.wall_us),h,att?"#9e692e":"#42827b",btn.title+"; "+core(b.cpu_us,b.wall_us)+" core equivalents");
    });
    for(const r of d.linears) {
      const h=79*rate(r)/rateMax;
      rect(wp,960*r[0]/d.wall_us,80-h,Math.max(.45,960*r[1]/d.wall_us),h,"#725884",r[3]+" · "+rate(r).toFixed(1)+" GB/s equivalent");
    }
    el("plot-note").textContent="Both plots start at zero. Highest block average: "+cpuMax.toFixed(2)+" cores. Highest projection equivalent: "+rateMax.toFixed(0)+" GB/s. Weight payload across linear calls: "+size(d.linear_bytes)+".";
    selectBlock(layer);
  }
  function selectBlock(i, sampleIndex) {
    layer=i;const b=A[phase].blocks[i],att=A.types[i]==="full_attention";
    el("block-range").value=i;el("block-readout").textContent=i+" / 63";
    Array.from(el("timebars").children).forEach((v,k)=>v.setAttribute("aria-pressed",k===i));
    el("block-count").textContent="Decoder block "+i+" · "+ms(b.start_us)+" from start";
    el("block-title").textContent=att?"Full attention with a KV cache":"Gated DeltaNet recurrent mixer";
    el("block-lesson").textContent=att?"Queries score visible cached keys. A softmax weights their values, then a gate and output projection feed the residual stream. The feed-forward network consumes the updated hidden state.":"A short convolution mixes recent inputs. A learned gate controls how the recurrent matrix forgets and incorporates the new key and value. The resulting readout joins the residual stream before the feed-forward network.";
    el("block-cache").textContent="State after this block: "+b.cache.map(r=>r.name.replace("_states","")+" "+size(r.bytes)).join(" · ")+".";
    const list=el("stage-list");list.replaceChildren();
    for(const s of b.stages){
      const row=add(list,"div");row.className="stage";const key=s.name.split(".").at(-1);
      add(row,"span",stages[key]||key);const bar=add(row,"i");bar.style.width=Math.min(100,100*s.wall_us/b.wall_us)+"%";add(row,"b",ms(s.wall_us));
    }
    const table=el("projections");table.replaceChildren();
    for(const e of A[phase].linears.filter(r=>r[3].startsWith("block."+i+"."))){
      const row=add(table,"tr");
      for(const text of [e[3].split(".").slice(2).join("."),ms(e[1]),size(e[2]),linearF(e[4])])add(row,"td",text);
    }
    el("block-stats").textContent=ms(b.wall_us)+" for the block · "+core(b.cpu_us,b.wall_us)+" average CPU cores · "+linearF(b.linear_flops)+" in linear projections · "+size(b.linear_bytes)+" of projection weights.";
    drawOccupancy(sampleIndex);
  }
  function drawOccupancy(chosen) {
    const {windows,blocks}=A[phase].occupancy;
    const end=windows.at(-1)[0], blockStart=blocks[layer][1], blockEnd=blocks[layer+1]?.[1]??end;
    const intervals=windows.map(([to,cpu],i)=>{
      const from=i?windows[i-1][0]:0;
      return {from,to,cpu,mean:cpu/(to-from)};
    });
    const index=chosen??intervals.findIndex(s=>s.to>=(blockStart+blockEnd)/2);
    const peak=Math.max(...intervals.map(s=>s.mean));
    el("peak-window").onclick=()=>chooseSample(intervals.findIndex(s=>s.mean===peak));
    const ymax=4*Math.ceil(peak/4), scale=112/ymax;
    const svg=el("occupancy");svg.replaceChildren();
    rect(svg,960*blockStart/end,0,960*(blockEnd-blockStart)/end,115,"#d1dfce","Span following block "+layer+" entry");
    for(let n=0;n<=ymax;n+=2)rect(svg,0,114-n*scale,960,.65,"#a7b8ae",n+" cores");
    intervals.forEach((s,i)=>{
      const h=Math.max(0,s.mean*scale);
      rect(svg,960*s.from/end,114-h,Math.max(.4,960*(s.to-s.from)/end),h,i===index?"#ad5b25":"#3b7d72",s.mean.toFixed(2)+" average busy cores");
    });
    const s=intervals[index], mid=(s.from+s.to)/2;
    const block=blocks.findLast(b=>b[1]<=mid);
    const at=block?"after entry to block "+block[0]:"during input setup";
    const axis=el("occupancy-axis");axis.replaceChildren();add(axis,"span","0 s");add(axis,"span","Vertical scale 0–"+ymax+" cores");add(axis,"span",ms(end));
    const input=el("sample-choice");input.max=windows.length-1;input.value=index;
    el("sample-readout").textContent=ms(s.from)+"–"+ms(s.to)+", "+at+". Across "+ms(s.to-s.from)+" of wall time, model-process threads accumulated "+ms(s.cpu)+" of CPU time, averaging "+s.mean.toFixed(2)+" busy cores.";
    el("sample-summary").textContent=windows.length+" intervals in this pass. Peak window average "+peak.toFixed(2)+" cores. An average above one demonstrates overlapping CPU work somewhere in the interval. Each sample sums runtime threads and excludes the sampler’s own CPU time. The last block’s band also covers the vocabulary projection. This clock starts with the sampling run.";
  }
  function chooseSample(index) {
    const {windows,blocks}=A[phase].occupancy;
    const midpoint=((index?windows[index-1][0]:0)+windows[index][0])/2;
    const block=blocks.findLast(b=>b[1]<=midpoint);
    stop();selectBlock(block?block[0]:0,index);
  }
  el("sample-choice").oninput=e=>chooseSample(+e.target.value);
  el("occupancy").onclick=e=>{
    const {windows}=A[phase].occupancy, bounds=el("occupancy").getBoundingClientRect();
    const time=windows.at(-1)[0]*(e.clientX-bounds.left)/bounds.width;
    const i=windows.findIndex(([to])=>to>=time);chooseSample(i<0?windows.length-1:i);
  };
  function stop(){clearInterval(timer);timer=null;el("play").textContent="Play blocks";el("play").setAttribute("aria-pressed","false");}
  el("prev").onclick=()=>{stop();selectBlock(Math.max(0,layer-1));};el("next").onclick=()=>{stop();selectBlock(Math.min(63,layer+1));};
  el("block-range").oninput=e=>{stop();selectBlock(+e.target.value);};
  el("play").onclick=()=>{if(timer){stop();return;}if(layer===63)selectBlock(0);el("play").textContent="Pause";el("play").setAttribute("aria-pressed","true");timer=setInterval(()=>{if(layer===63){stop();return;}selectBlock(layer+1);},650);};
  for(const p of ["prefill","decode"])el(p).onclick=()=>{stop();phase=p;drawTimeline();};
  function tensorMap() {
    const svg=el("tensor-map");svg.replaceChildren();
    function shape(parent,tag,attrs,text){
      const n=document.createElementNS(svg.namespaceURI,tag);
      for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);
      if(text!==undefined)n.textContent=text;parent.append(n);return n;
    }
    function tile(x,y,w,h,fill,label,action){
      const g=shape(svg,"g",{class:"tile","aria-label":label});shape(g,"title",{},label);
      shape(g,"path",{d:"M"+x+" "+y+"l5 -5h"+w+"l-5 5Z",fill:"#b8cec5"});
      shape(g,"path",{d:"M"+(x+w)+" "+y+"l5 -5v"+h+"l-5 5Z",fill:"#789b91"});
      shape(g,"rect",{x,y,width:w,height:h,fill});g.onclick=action;
    }
    shape(svg,"text",{x:13,y:21},"Queries · 24 heads");
    shape(svg,"text",{x:268,y:21},"Key / value cache · 17 positions per KV head");
    for(let group=0;group<4;group++){
      const y=40+group*39;
      shape(svg,"path",{d:"M152 "+(y+8)+"H257",stroke:group===Math.floor(head/6)?"#14726f":"#b3c2b9","stroke-width":group===Math.floor(head/6)?3:1});
      shape(svg,"text",{x:177,y:y+3,class:"kv-label"},"KV "+group);
      for(let j=0;j<6;j++){
        const idx=group*6+j;
        tile(13+j*23,y,17,16,idx===head?"#14726f":"#dce8df","Select query head "+idx,()=>{
          head=idx;el("head-choice").value=idx;drawHead();
        });
      }
      for(let j=0;j<17;j++){
        const sel=group===Math.floor(head/6);
        tile(268+j*24,y,17,16,sel&&position===j?"#b66d26":sel?"#6fa69a":"#dce8df","KV "+group+", position "+j,()=>{
          position=j;head=group*6+head%6;el("head-choice").value=head;drawHead();
        });
      }
    }
    shape(svg,"text",{x:13,y:200,class:"kv-label"},"Selected query "+head);
    shape(svg,"text",{x:268,y:200,class:"kv-label"},"First token");
    shape(svg,"text",{x:611,y:200,class:"kv-label"},"New token");
  }
  function inspectPosition(){
    tensorMap();
    const sample=A.attention.heads[head], group=Math.floor(head/6), weight=sample.prob[position], value=A.attention.values[group][position][feature];
    el("head-title").textContent="Head "+head+" reads KV head "+group;
    el("head-desc").textContent="Position "+position+" holds “"+A.tokens[position]+"”. Six neighboring query heads share this key and value store.";
    drawProducts(sample.scores[position],group);
    const nums=el("head-numbers");nums.replaceChildren();
    for(const [key,val] of [["Scaled dot product",fixed(sample.scores[position])],["Share of attention",(100*weight).toFixed(2)+"%"],["Value at feature "+A.attention.columns[feature],fixed(value)],["Contribution to output",fixed(value*weight)]]){
      const n=add(nums,"div");add(n,"dt",key);add(n,"dd",val);
    }
    el("head-math").textContent="Each score is a 256-component dot product divided by 16. Multiply this position’s value by its share, then sum over all 17 positions. The resulting output at feature "+A.attention.columns[feature]+" is "+fixed(sample.out[feature])+".";
    Array.from(el("token-bars").children).forEach((b,i)=>b.setAttribute("aria-pressed",i===position));
  }
  function drawProducts(score,group) {
    const q=A.attention.queries[head], k=A.attention.keys[group][position];
    const products=q.map((x,i)=>x*k[i]);
    const max=Math.max(...products.map(Math.abs)) || 1;
    const svg=el("products");svg.replaceChildren();
    rect(svg,0,55.5,1024,1,"#73827e","Zero");
    products.forEach((product,i)=>{
      const height=52*Math.abs(product)/max;
      const color=i===feature?"#192b30":product>=0?"#338575":"#af602b";
      rect(svg,4*i,product>=0?56-height:56,3.5,Math.max(.4,height),color,"Feature "+i+" · "+fixed(product));
    });
    el("product-note").textContent="At feature "+feature+", Q = "+fixed(q[feature])+" and K = "+fixed(k[feature])+". Their product is "+fixed(products[feature])+". All 256 products sum to "+fixed(products.reduce((a,b)=>a+b,0))+". Dividing by √256 = 16 gives the scaled score "+fixed(score)+".";
  }
  el("products").onclick=e=>{
    const bounds=el("products").getBoundingClientRect();
    feature=Math.min(255,Math.max(0,Math.floor(256*(e.clientX-bounds.left)/bounds.width)));
    el("feature").value=feature;inspectPosition();
  };
  function drawWeights(){
    const components=[
      ["decoder_blocks","64 decoder blocks","#427c72","Every text block ran. The three gated MLP projections in each block account for 534,773,760 bytes of its weights."],
      ["model.language_model.embed_tokens.weight","Token embeddings","#9cb9a2","The embedding lookup selected the prompt rows during prefill and one row during decode. Each row contains 5,120 BF16 values."],
      ["lm_head.weight","Vocabulary head","#896c96","The output projection used this separate matrix to compute 248,320 vocabulary scores for the final position of each pass."],
      ["vision","Vision encoder","#bd9650","This branch takes image or video inputs. The captured prompt contained text only, so its weights had no operator calls in these passes."],
      ["multi_token_prediction","Multi-token prediction","#c6b6ae","The checkpoint includes additional weights for speculative prediction. The captured run used ordinary next-token inference."],
      ["model.language_model.norm.weight","Final normalization","#6374a1","The last normalization applied one learned BF16 weight to each of 5,120 features before the vocabulary head."]
    ];
    const total=Object.values(A.weight_inventory).reduce((sum,bytes)=>sum+bytes,0);
    const strip=el("weight-strip"),legend=el("weight-keys");
    function selectWeight(index) {
      const [key,label,,explanation]=components[index],bytes=A.weight_inventory[key];
      Array.from(legend.children).forEach((button,i)=>button.setAttribute("aria-pressed",i===index));
      el("weight-detail").textContent=label+" · "+bytes.toLocaleString("en-US")+" bytes ("+(100*bytes/total).toLocaleString("en-US",{maximumSignificantDigits:3})+"% of checkpoint payload). "+explanation;
    }
    components.forEach(([key,label,color],i)=>{
      const area=add(strip,"span");area.style.width=100*A.weight_inventory[key]/total+"%";area.style.background=color;
      const button=add(legend,"button");const swatch=add(button,"i");swatch.style.background=color;
      button.append(document.createTextNode(label));button.onclick=()=>selectWeight(i);
    });
    selectWeight(0);
  }
  function drawHead(){
    const sample=A.attention.heads[head], bars=el("token-bars");bars.replaceChildren();
    const largest=Math.max(...sample.prob);
    sample.prob.forEach((p,i)=>{
      const b=add(bars,"button");b.setAttribute("aria-label","Position "+i+", "+A.tokens[i]+", "+(100*p).toFixed(2)+" percent");
      b.title=b.getAttribute("aria-label");const mark=add(b,"i");mark.style.height=(p/largest*65)+"%";add(b,"b",A.tokens[i]);b.onclick=()=>{position=i;inspectPosition();};
    });
    inspectPosition();
  }
  for(let h=0;h<24;h++)el("head-choice").add(new Option(String(h),h));
  A.attention.columns.forEach((c,i)=>el("feature").add(new Option(String(c),i)));
  el("head-choice").onchange=e=>{head=+e.target.value;drawHead();};el("feature").onchange=e=>{feature=+e.target.value;inspectPosition();};
  drawTimeline();drawHead();drawWeights();
})();
