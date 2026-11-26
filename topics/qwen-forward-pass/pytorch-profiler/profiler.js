"use strict";
const $ = id => document.getElementById(id);
const P = window.profileData;
const format = new Intl.NumberFormat("en-US");
const num = value => format.format(value);
const fmt = us => Math.abs(us) >= 1e6 ? (us / 1e6).toFixed(2) + " s" : Math.abs(us) >= 1000 ? (us / 1000).toFixed(2) + " ms" : us.toFixed(us < 10 ? 2 : 1) + " µs";
const bytes = n => n >= 1048576 ? (n/1048576).toFixed(2)+" MiB" : n >= 1024 ? (n/1024).toFixed(2)+" KiB" : n+" B";
const pct = n => (100*n).toFixed(1)+"%";
const option = (value,label) => new Option(label,value);
const make = (tag,cls,content) => {const el=document.createElement(tag); if(cls) el.className=cls;if(content!=null) el.textContent=content;return el;};
function svg(tag,attrs={},label) {
    const el=document.createElementNS("http://www.w3.org/2000/svg",tag);
    Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));
    if(label!=null) el.textContent=label;
    return el;
}
function action(el,callback) {
    el.addEventListener("click",callback);
    el.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault(); callback();}});
}
const family = name => name.startsWith("Qwen/block.") && !name.includes("replay") ? (+name.split(".")[1]%4===3?"#995821":"#166762") : name.startsWith("Qwen")?"#24404a": /conv|thnn/.test(name)?"#655581": /matmul|::mm$|::bmm$|linear/.test(name)?"#166762":/softmax|sigmoid|silu/.test(name)?"#995821":"#587484";
const short = name=>name.replace("Qwen/block.","Block ").replace("aten::","");
const state={phase:"decode",lo:0,hi:1,event:0,order:2,all:false,rank:"",flow:0,allocation:0};
const d=()=>P[state.phase];
const replaying=()=>state.phase==="replay";
const rows=()=>d().rows;
const isBlock=r=>/^Qwen\/block\.\d+$/.test(r[0]);
const end=r=>r[1]+r[2];
const intersects=r=>r[1]<state.hi && end(r)>state.lo;
function getParents(list){
    const stack=[], parents=[];
    list.forEach((r,i)=>{
        while(stack.length && (list[stack.at(-1)][3]>=r[3] || end(list[stack.at(-1)])<r[1]+.001)) stack.pop();
        parents.push(stack.at(-1)??-1); stack.push(i);
    });
    return parents;
}
for(const dataset of Object.values(P)){dataset.parents=getParents(dataset.rows);}
function interval(lo,hi) {
    const duration=d().trace_us, width=Math.min(duration,Math.max(.2,hi-lo));
    state.lo=Math.min(Math.max(0,lo),duration-width); state.hi=state.lo+width;
}
function focus(i,zoom=true,scroll=false) {
    if(i<0) return; state.event=i;
    if(zoom){const r=rows()[i];interval(r[1],end(r));}
    renderWindow();
    if(scroll) $("timeline").scrollIntoView({block:"start"});
}
function setupPhase(phase) {
    state.phase=phase;
    const data=d(), replay=replaying();
    document.querySelectorAll("#phases button").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.phase===phase)));
    $("overviewWrap").hidden=replay; $("clocks").hidden=replay;
    $("subset").value="all";$("subset").disabled=replay; $("search").value="";state.rank="";state.all=false;
    $("phaseIntro").textContent=replay ?
        "One FP32 attention core from block 3. All "+num(data.count)+" measured host ranges are available, including the small operations inside each batched product." :
        (phase==="decode"?"One cached token follows 16 prompt tokens.":"Sixteen prompt tokens enter an empty cache.")+" The BF16 model ran on an Apple M4 Max CPU with eight intra-op threads. The view retains "+num(data.rows.length)+" of "+num(data.count)+" host ranges.";
    const blocks=data.rows.map((r,i)=>isBlock(r)?i:-1).filter(i=>i>=0);
    $("overview").replaceChildren();
    $("window").replaceChildren(option("full","Whole capture"),...blocks.map(i=>option(i,short(data.rows[i][0])+(Number(data.rows[i][0].split(".")[1])%4===3?" · full attention":" · recurrent"))));
    $("clockScope").replaceChildren(option("all","Whole capture"),...blocks.map(i=>option(i,short(data.rows[i][0]))));
    for(const i of blocks){
        const r=data.rows[i],btn=make("button",Number(r[0].split(".")[1])%4===3?"att":"");
        btn.style.left=pctFine(r[1]/data.trace_us);btn.style.width=pctFine(r[2]/data.trace_us);
        btn.title=short(r[0])+" · "+fmt(r[2]);btn.setAttribute("aria-label",btn.title);btn.dataset.id=i;btn.onclick=()=>focus(i);
        $("overview").append(btn);
    }
    $("overviewAxis").replaceChildren(make("span","","0 s"),make("span","",fmt(data.trace_us)));
    $("shortcuts").replaceChildren();
    const examples=replay?[["Repeated K","aten::repeat_interleave"],["Batched dot product","aten::bmm"],["Softmax","aten::_softmax"]]:
        [["Per-channel convolution","aten::_slow_conv2d_forward"],["Matrix multiply","aten::mm"],["Attention scores","aten::bmm"]];
    for(const [label,name] of examples){
        const btn=make("button","",label);btn.onclick=()=>focus(rows().findIndex(r=>r[0]===name));$("shortcuts").append(btn);
    }
    state.event=replay?0:blocks[0];
    const initial=data.rows[state.event];interval(initial[1],end(initial));
    renderWindow(); renderRank(); if(!replay) renderClock();
}
const pctFine=n=>(n*100)+"%";
function renderWindow() {
    const data=d(),list=rows(),selected=list[state.event], span=state.hi-state.lo;
    if(!intersects(selected)){state.event=list.findIndex(intersects); }
    let fitted=list.findIndex(r=>isBlock(r)&&Math.abs(state.lo-r[1])<.01&&Math.abs(span-r[2])<.01);
    const full=Math.abs(data.trace_us-span)<.01;
    const custom=$("window").querySelector('option[value="custom"]');
    if(custom) custom.remove();
    if(!full&&fitted<0){$("window").add(option("custom","Custom interval"));}
    $("window").value=full?"full":fitted<0?"custom":fitted;
    document.querySelectorAll("#overview button").forEach(b=>b.setAttribute("aria-pressed",String(+b.dataset.id===fitted)));
    $("leftEdge").textContent=fmt(state.lo);$("rightEdge").textContent=fmt(state.hi);
    $("pan").disabled=full; $("pan").value=full?0:1e6*state.lo/(data.trace_us-span);
    $("closer").disabled=span<=.21;$("further").disabled=full;
    const visible=list.map((r,i)=>intersects(r)?i:-1).filter(i=>i>=0);
    $("event").replaceChildren(...visible.map(i=>option(i,short(list[i][0])+" · "+fmt(list[i][2])+" · at "+fmt(list[i][1]))));
    $("event").value=state.event;
    drawTimeline(visible);renderEvent();
}
function drawTimeline(visible){
    const list=rows(),canvas=$("timelineSvg"),w=Math.max(240,Math.floor(canvas.getBoundingClientRect().width)),padding=39;
    const depthMax=Math.max(...visible.map(i=>list[i][3]));
    const rowHeight=25,height=(depthMax+1)*rowHeight+20;
    const span=state.hi-state.lo,extent=w-padding;
    canvas.setAttribute("viewBox","0 0 "+w+" "+height);canvas.replaceChildren();
    for(let t=0;t<=4;t++){
        const x=padding+t/4*extent;canvas.append(svg("line",{x1:x,x2:x,y1:0,y2:height,stroke:"#c1cfc4","stroke-dasharray":"2 3"}));
    }
    for(let layer=0;layer<=depthMax;layer++){
        canvas.append(svg("text",{x:0,y:layer*rowHeight+17,fill:"#44595b","font-size":10},layer===0?"Root":"L"+layer));
    }
    for(const i of visible){
        const r=list[i], x=padding+extent*(Math.max(r[1],state.lo)-state.lo)/span, width=extent*(Math.min(end(r),state.hi)-Math.max(r[1],state.lo))/span;
        const current=i===state.event;
        if(width<.45 && !current) continue;
        const y=r[3]*rowHeight+2;
        const group=svg("g",{class:"range-bar",role:"button","aria-label":r[0]+", "+fmt(r[2]),tabindex:width>25&&visible.length<140?0:-1,"data-index":i});
        group.append(svg("title",{},r[0]+" · "+fmt(r[2])+" inclusive • "+fmt(r[4])+" self"),
            svg("rect",{x:x+.25,y,width:Math.max(.4,width-.4),height:20,rx:2,fill:family(r[0]),stroke:current?"#142629":"none","stroke-width":current?2.7:0}));
        const title=short(r[0]);const chars=Math.floor((width-8)/6.8);
        if(chars>=3) group.append(svg("text",{x:x+4,y:y+14,fill:"white","font-size":11},title.length<=chars?title:title.slice(0,chars-1)+"…"));
        action(group,()=>focus(i,false));
        group.addEventListener("dblclick",()=>focus(i));canvas.append(group);
    }
}
function renderEvent(){
    const data=d(),r=rows()[state.event],parent=data.parents[state.event];
    $("eventTitle").textContent=r[0];$("eventContext").textContent="Host thread "+data.host_tid+" · record "+num(r[7])+" · begins at "+fmt(r[1])+(parent>=0?" inside "+rows()[parent][0]+".":".");
    $("parent").disabled=parent<0;$("zoomEvent").disabled=Math.abs((state.hi-state.lo)-r[2])<.01;
    const fraction=Math.max(0,Math.min(1,r[4]/r[2]));
    const own=make("i"),child=make("b");own.style.width=pctFine(fraction);child.style.width=pctFine(1-fraction);$("timeSplit").replaceChildren(own,child);
    $("timeValues").textContent=fmt(r[2])+" inclusive = "+fmt(r[4])+" self + "+fmt(r[2]-r[4])+" in recorded children.";
    const inputs=r[5],outputs=r[6],all=[...inputs,...(outputs||[])],largest=Math.max(1,...all.map(t=>t[2]));
    $("tensors").replaceChildren();
    for(const [kind,entries] of [["Input",inputs],["Output",outputs||[]]]){
        entries.forEach(([shape,type,n],j)=>{
            const row=make("div","tensor-row"+(kind==="Output"?" out":""));
            const label=make("small");label.append(make("span","",kind+" "+(j+1)+" · "+type+" ["+shape.join(" × ")+"]"),make("b","",bytes(n)));
            const track=make("div","track"),fill=make("i");fill.style.width=pctFine(n/largest);track.append(fill);row.append(label,track);$("tensors").append(row);
        });
    }
    $("payloadHint").textContent=all.length?"Logical bytes = element count × element size. Shared views can refer to the same storage. Hardware reads and writes were unmeasured."+(outputs===null?" This channel sample has only native input shapes.":""):"This range has no tensor operands in the capture. Select an operator to see its shapes and sizes.";
}
function aggregateReplay(){
    const agg={};for(const r of P.replay.rows){
        const a=agg[r[0]]??=[0,0,0,0,0,0];a[0]++;a[1]+=r[2];a[2]+=r[4];
    }return agg;
}
P.replay.aggregate=aggregateReplay();
function renderRank() {
    const data=d(), agg=data.aggregate, grouped=$("subset").value==="conv"&&!replaying(),offset=grouped?3:0,order=state.order;
    const records=Object.entries(agg), filter=$("search").value.trim().toLowerCase();
    const conv=agg["aten::conv1d"],convUs=conv?.[1]||0;
    $("convShare").hidden=replaying();$("convLabels").hidden=replaying();
    if(!replaying()){
        const fill=make("i");fill.style.width=pctFine(convUs/data.trace_us);$("convShare").replaceChildren(fill);
        $("convLabels").replaceChildren(make("span","","Grouped convolution · "+fmt(convUs)+" · "+pct(convUs/data.trace_us)),make("span","","Rest of capture · "+fmt(data.trace_us-convUs)));
        const under=records.reduce((a,item)=>a+item[1][3],0);
        $("summaryLead").textContent=num(under)+" of "+num(data.count)+" host ranges fall within "+conv[0]+" grouped convolutions. Each has 10,240 channels. The CPU path issued "+num(agg["aten::_slow_conv2d_forward"][0])+" single-channel convolution calls.";
    }else{$("summaryLead").textContent="All "+num(data.count)+" CPU ranges from the FP32 attention replay, including the enclosing application marker.";}
    const filtered=records.filter(([name,a])=>a[offset]>0&&name.toLowerCase().includes(filter)).sort((a,b)=>b[1][offset+order]-a[1][offset+order]);
    if(!filtered.some(e=>e[0]===state.rank)) state.rank=filtered[0]?.[0]||"";
    const shown=state.all?filtered:filtered.slice(0,10),max=filtered[0]?.[1][offset+order]||1;
    const box=$("ranking");box.replaceChildren();
    const value=n=>order===0?num(n):fmt(n);
    for(const [name,a] of shown){
        const btn=make("button","rankrow");btn.setAttribute("aria-pressed",String(name===state.rank));
        const bar=make("i","bar");bar.style.width=pctFine(a[offset+order]/max);
        btn.append(make("span","",name),bar,make("b","",value(a[offset+order])));
        btn.onclick=()=>{state.rank=name;renderRank();};box.append(btn);
    }
    if(!shown.length) box.textContent="No operator in this scope matches your search.";
    $("showAll").hidden=filtered.length<=10;$("showAll").textContent=state.all?"Show first 10":"Show all "+num(filtered.length)+" names";
    $("rankHint").textContent=order===2?"Self time removes every recorded child interval on this thread. The individual self times partition the selected capture scope.":order===1?"Inclusive time includes nested calls. The same elapsed interval can contribute to several bars.":"Each bar counts complete host spans bearing that name. An enclosing annotation and each of its children have separate records.";
    const a=agg[state.rank];$("rankDetail").textContent=a?state.rank+" · "+num(a[offset])+" calls · "+fmt(a[offset+1])+" inclusive · "+fmt(a[offset+2])+" self.":"";
    document.querySelectorAll("#measure button").forEach(b=>b.setAttribute("aria-pressed",String(+b.dataset.order===order)));
}
function renderClock(){
    const data=d(),chosen=$("clockScope").value,whole=chosen==="all";
    const [wall,cpu]=whole?[data.metrics.wall_us,data.metrics.cpu_us]:data.metrics.blocks[rows()[+chosen][0].replace("Qwen/","")];
    const instrumented=whole?data.trace_us:rows()[+chosen][2],max=Math.max(cpu,wall,instrumented);
    const box=$("clockChart");box.replaceChildren();
    for(const [label,n] of [["Wall time · module hooks",wall],["Process CPU · module hooks",cpu],["Wall time · instrumented trace",instrumented]]){
        const el=make("div","metric-line"),cap=make("small"),track=make("div","track"),bar=make("i");bar.style.width=pctFine(n/max);
        cap.append(make("span","",label),make("b","",fmt(n)));track.append(bar);el.append(cap,track);box.append(el);
    }
    $("clockConclusion").textContent=(cpu/wall).toFixed(2)+" average busy cores during this interval. The instrumented interval lasted "+(instrumented/wall).toFixed(2)+"× the wall time of the module-clock run.";
}
const stageRows=P.replay.rows.map((r,i)=>r[3]===1?i:-1).filter(i=>i>=0);
// Code-level tensor dependencies, ordered separately from host timestamps.
const stages=[
    ["Repeat keys",15,20,"Four KV heads expand to twenty-four. The first repeat_interleave produces the keys consumed by the score product."],
    ["Repeat values",15,105,"Value expansion uses a separate V input. Its result can be formed independently of the key expansion and score calculation."],
    ["Transpose keys",175,20,"The last two key dimensions swap to expose the feature dimension for the dot product."],
    ["Q × keys",335,20,"The first matmul reads the queries and transposed keys. Each query head produces 17 scores."],
    ["Scale",486,20,"Multiply each score by 1/16, the inverse square root of the 256-feature head size."],
    ["Softmax",620,20,"Normalize the 17 scores for each head. This result depends on the completed score calculation."],
    ["Mix values",667,105,"The second matmul needs both the normalized probabilities and the repeated V tensor. It produces 256 features for each of 24 heads."]
];
function flowChart(){
    const s=$("flow");s.replaceChildren();
    const defs=svg("defs"),marker=svg("marker",{id:"arrow",viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:5,markerHeight:5,orient:"auto"});
    marker.append(svg("path",{d:"M0 0L10 5L0 10Z",fill:"#527069"}));defs.append(marker);s.append(defs);
    const paths=["M140 42H172","M300 42H332","M460 42H483","M611 42H617","M682 65V87H729V102","M140 127H665"];
    for(const path of paths)s.append(svg("path",{d:path,fill:"none",stroke:"#527069","stroke-width":2,"marker-end":"url(#arrow)"}));
    stages.forEach(([name,x,y,info],index)=>{
        const g=svg("g",{class:"flow-node",tabindex:0,role:"button","aria-label":"Inspect "+name,"aria-pressed":index===state.flow});
        g.append(svg("rect",{x,y,width:125,height:45,rx:4,fill:index===state.flow?"#176762":"#e4e9df",stroke:"#637b70"}),
              svg("text",{x:x+62.5,y:y+27,"text-anchor":"middle","font-size":14,fill:index===state.flow?"#fff":"#19383a"},name));
        action(g,()=>{
            state.flow=index;if(!replaying()) setupPhase("replay");focus(stageRows[index],true);
            flowChart();$("timeline").scrollIntoView({block:"start"});
        });s.append(g);
    });
    const stage=stages[state.flow],r=P.replay.rows[stageRows[state.flow]];
    $("flowDetail").textContent=stage[3]+" Measured "+r[0]+" interval: "+fmt(r[2])+".";
}
function allocationChart(){
    const data=P.replay.allocations,plot=$("alloc");plot.setAttribute("viewBox","0 0 540 155");plot.replaceChildren();
    let running=0,max=0;const levels=data.map(([,delta])=>{running+=delta;max=Math.max(max,running);return running;});
    const x=t=>65+464*t/P.replay.trace_us,y=b=>119-94*b/max;
    let path="M65 119";
    data.forEach(([t],i)=>{path+="H"+x(t)+"V"+y(levels[i]);});
    path+="H529";plot.append(svg("path",{d:path+"V119H65Z",fill:"#cfe2d6"}),svg("path",{d:path,stroke:"#166762",fill:"none","stroke-width":2}));
    plot.append(svg("text",{x:4,y:27,"font-size":11},bytes(max)),svg("text",{x:20,y:118,"font-size":11},"0 B"),svg("text",{x:65,y:144,"font-size":11},"0 µs"),svg("text",{x:529,y:144,"text-anchor":"end","font-size":11},fmt(P.replay.trace_us)));
    data.forEach(([t,delta],i)=>{
        const dot=svg("circle",{cx:x(t),cy:y(levels[i]),r:i===state.allocation?5.8:4,fill:i===state.allocation?"#9b5120":"#166762",stroke:"white",class:"point",role:"button",tabindex:0,"aria-label":"Event "+(i+1)+", "+num(delta)+" bytes"});dot.append(svg("title",{},num(delta)+" B at "+fmt(t)));action(dot,()=>{state.allocation=i;allocationChart();});plot.append(dot);
    });
    $("allocSelect").value=state.allocation;
    const [t,delta]=data[state.allocation];
    const target=P.replay.rows.filter(r=>t>=r[1]&&t<=end(r)).at(-1);
    $("allocDetail").textContent=fmt(t)+" · "+(delta>0?"allocated ":"released ")+num(Math.abs(delta))+" bytes"+(target?" in "+target[0]:"")+". Net live "+num(levels[state.allocation])+" bytes after this event. Peak "+bytes(max)+".";
}
function valueChart(){
    const plot=$("values"),head=+$("head").value,feat=Math.min(255,Math.max(0,Math.floor(Number($("feature").value)||0)));
    const output=P.replay.actual[head],ref=P.replay.expected[head],bf=P.replay.native[head];
    const min=Math.min(...output),max=Math.max(...output),x=i=>15+510*i/255,y=v=>105-92*(v-min)/(max-min||1);
    plot.setAttribute("viewBox","0 0 540 144");plot.replaceChildren(svg("line",{x1:15,x2:525,y1:y(0),y2:y(0),stroke:"#b1c5b8"}),
        svg("polyline",{points:output.map((v,i)=>x(i)+","+y(v)).join(" "),fill:"none",stroke:"#166762","stroke-width":1.7}),
        svg("circle",{cx:x(feat),cy:y(output[feat]),r:5,fill:"#995821",stroke:"#fff"}),
        svg("text",{x:15,y:134,"font-size":11},"Feature 0"),svg("text",{x:525,y:134,"text-anchor":"end","font-size":11},"255"));
    $("comparison").textContent="Head "+head+", feature "+feat+" · replay "+output[feat].toPrecision(6)+", FP32 reference "+ref[feat].toPrecision(6)+", original BF16 model "+bf[feat].toPrecision(6)+". Across all 6,144 outputs, "+num(P.replay.exact)+" matched the FP32 reference exactly. Maximum absolute difference from the original BF16 model: "+P.replay.native_error.toPrecision(5)+".";
}
$("window").onchange=e=>{if(e.target.value==="full") focus(0);else if(e.target.value!=="custom") focus(+e.target.value);};
$("event").onchange=e=>focus(+e.target.value,false);
$("zoomEvent").onclick=()=>focus(state.event);
$("parent").onclick=()=>focus(d().parents[state.event],false);
$("reset").onclick=()=>focus(0);
const zoom=factor=>{const mid=(state.lo+state.hi)/2,half=(state.hi-state.lo)*factor/2;interval(mid-half,mid+half);renderWindow();};
$("closer").onclick=()=>zoom(.5);$("further").onclick=()=>zoom(2);
$("pan").oninput=e=>{const span=state.hi-state.lo,start=Number(e.target.value)/1e6*(d().trace_us-span);interval(start,start+span);renderWindow();};
$("phases").onclick=e=>{if(e.target.dataset.phase)setupPhase(e.target.dataset.phase);};
$("measure").onclick=e=>{if(e.target.dataset.order!=null){state.order=+e.target.dataset.order;state.rank="";renderRank();}};
$("subset").onchange=()=>{state.rank="";renderRank();};
$("search").oninput=()=>{state.rank="";renderRank();};
$("showAll").onclick=()=>{state.all=!state.all;renderRank();};
$("clockScope").onchange=renderClock;
$("allocSelect").replaceChildren(...P.replay.allocations.map(([t,delta],i)=>option(i,(i+1)+" · "+(delta>0?"+":"")+num(delta)+" B at "+fmt(t))));
$("allocSelect").onchange=e=>{state.allocation=+e.target.value;allocationChart();};
$("head").replaceChildren(...Array.from({length:24},(_,i)=>option(i,i)));
$("head").onchange=valueChart;$("feature").oninput=valueChart;
$("values").onclick=e=>{const bounds=e.currentTarget.getBoundingClientRect();$("feature").value=Math.max(0,Math.min(255,Math.round(((e.clientX-bounds.left)/bounds.width*540-15)/510*255)));valueChart();};
setupPhase(location.hash==="#replay"?"replay":"decode");flowChart();allocationChart();valueChart();
let resized;window.addEventListener("resize",()=>{clearTimeout(resized);resized=setTimeout(renderWindow,100);});
