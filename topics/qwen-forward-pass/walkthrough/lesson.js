(() => {
  const stages=window.QWEN_CHAPTERS, data=window.QWEN_EXAMPLES;
  const byId=id=>document.getElementById(id);
  const phases=document.querySelectorAll('[data-phase]');
  const plan=byId('stages'), tensorChoice=byId('tensor-choice');
  const headChoice=byId('query-head'), positionChoice=byId('cache-position');
  const stageFromHash=()=>stages.findIndex(s=>'#'+s.id===location.hash);
  let index=Math.max(0,stageFromHash()), phase='decode', selected, head=0, pos=16, timer;
  const label=s=>s.id.startsWith('rec-')?'Block 0 · recurrent':s.id.startsWith('full-')?'Block 3 · full attention':s.id==='tokens'?'Text input':'Model output';
  const options=(select,items)=>select.replaceChildren(...items.map(([name,value])=>new Option(name,value)));
  function setHash(id) {
    try { history.replaceState(null,'','#'+id); } catch { /* The player also works from a read-only file preview. */ }
  }
  function stop() {
    clearInterval(timer);timer=undefined;
    byId('play').textContent='Play';byId('play').setAttribute('aria-pressed','false');
  }
  function dims(shape) {
    const T=phase==='prefill'?16:1, L=phase==='prefill'?16:17;
    return shape.map(x=>x==='T'?T:x==='L'?L:x);
  }
  const engine=window.createQwenTensorView(byId('tensor-scene'),pick);
  function drawScene() {
    const s=stages[index], weights=data.attention[phase].weights[head];
    const nodes=s.nodes.map(n=>{
      const node={...n,shape:dims(n.shape)};
      if(s.id==='full-score'&&['p','v','k'].includes(n.id)){
        node.activeCell=pos;
        if(n.id==='p')node.heat=weights;
      }
      return node;
    });
    engine.setScene({nodes,edges:s.edges,selected});
    const n=nodes.find(n=>n.id===selected);
    byId('tensor-detail').textContent=n.label+' ['+n.shape.join(' × ')+']. '+n.detail;
    if(s.id==='full-score'&&['p','k','v'].includes(n.id)){
      byId('tensor-detail').textContent+=' Selected position '+pos+' (“'+data.tokens[pos]+'”) receives '+(100*weights[pos]).toFixed(2)+'% for query head '+head+'.';
    }
    tensorChoice.value=selected;
  }
  function pick(id,cell) {
    if(!stages[index].nodes.some(n=>n.id===id))return;
    if(stages[index].id==='full-score'&&Number.isInteger(cell)&&cell>=0&&cell<data.attention[phase].weights[head].length){
      pos=cell;renderHead();
    }
    stop();selected=id;drawScene();
  }
  function renderHead() {
    const a=data.attention[phase], weights=a.weights[head], scores=a.scores[head];
    positionChoice.value=pos;headChoice.value=head;
    const strip=byId('prob-strip'); strip.replaceChildren();
    weights.forEach((w,i)=>{
      const b=document.createElement('button');
      b.title='Index '+i+', “'+data.tokens[i]+'” · '+(w*100).toFixed(2)+'%';
      b.setAttribute('aria-label',b.title);b.setAttribute('aria-pressed',String(i===pos));
      b.style.height=(4+92*w/Math.max(...weights))+'%';
      b.onclick=()=>{stop();pos=i;renderHead();drawScene();strip.children[i].focus();};
      strip.append(b);
    });
    const peak=weights.indexOf(Math.max(...weights));
    byId('head-example').textContent='Query head '+head+' shares KV head '+Math.floor(head/6)+'. Index '+pos+' (“'+data.tokens[pos]+'”) scores '+scores[pos].toFixed(3)+' and receives '+(100*weights[pos]).toFixed(2)+'% of the attention. The highest weight is at index '+peak+' (“'+data.tokens[peak]+'”). Bars are scaled to the largest weight in this row.';
  }
  headChoice.onchange=()=>{stop();head=+headChoice.value;renderHead();drawScene();};
  positionChoice.onchange=()=>{stop();pos=+positionChoice.value;renderHead();drawScene();};
  function updateStep(i,writeHash=true) {
    index=i;const s=stages[index];
    selected=s.id==='full-score'?'p':s.nodes.find(n=>n.role==='output')?.id||s.nodes.at(-1).id;
    const heading=label(s);
    plan.value=i;
    byId('scene-name').textContent=heading;
    byId('position').textContent='Step '+(i+1)+' of '+stages.length+' · '+heading;
    byId('step-title').textContent=s.title;
    byId('action').textContent=s.action;byId('why').textContent=s.why;
    byId('shape-note').textContent=s.shape;byId('snippet').textContent=s.code;
    const note=byId('source-note');note.replaceChildren();
    if(s.actual){note.append('Model code, line '+s.actualLine+' · ');const code=document.createElement('code');code.textContent=s.actual;note.append(code,document.createElement('br'));}
    const link=document.createElement('a');link.href=data.model+'#L'+s.source[0]+'-L'+s.source[1];link.textContent='Read the model implementation, lines '+s.source[0]+'–'+s.source[1];note.append(link);
    byId('observed').textContent=data[phase][s.id];
    options(tensorChoice,s.nodes.map(n=>[n.label,n.id]));
    const full=s.id==='full-score';
    byId('live-attention').hidden=!full;
    if(full)renderHead();
    byId('back').disabled=i===0;byId('next').disabled=i===stages.length-1;
    byId('progress-text').textContent='Step '+(i+1)+' of '+stages.length;
    Array.from(byId('progress').children).forEach((b,j)=>b.setAttribute('aria-pressed',String(i===j)));
    drawScene();
    if(writeHash)setHash(s.id);
  }
  function setPhase(value) {
    stop();phase=value;
    const a=data.attention[phase];head=Math.min(head,a.weights.length-1);
    pos=a.weights[head].length-1;
    options(headChoice,a.weights.map((_,i)=>[''+i,i]));
    options(positionChoice,a.weights[head].map((_,i)=>[i+' · '+data.tokens[i],i]));
    phases.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.phase===phase)));
    byId('phase-description').textContent=(phase==='prefill'?'Prefill reads the 16-token prompt with an empty cache. B = 1 sequence, T = 16 current tokens and L = 16 cache positions after the pass.':'Decode reads the supplied token “ The” at index 16 using the previous 16 positions. B = 1 sequence, T = 1 current token and L = 17 cache positions after the pass.');
    updateStep(index,false);
  }
  options(plan,stages.map((s,i)=>[(i+1)+'. '+s.title,i]));
  stages.forEach((s,i)=>{
    const b=document.createElement('button');
    b.dataset.kind=s.id.startsWith('full-')?'full':'other';
    b.title=(i+1)+'. '+label(s)+' · '+s.title;b.setAttribute('aria-label',b.title);
    b.onclick=()=>{stop();updateStep(i);};byId('progress').append(b);
  });
  plan.onchange=()=>{stop();updateStep(+plan.value);};
  tensorChoice.onchange=()=>pick(tensorChoice.value);
  byId('back').onclick=()=>{stop();updateStep(Math.max(0,index-1));};
  byId('next').onclick=()=>{stop();updateStep(Math.min(stages.length-1,index+1));};
  byId('play').onclick=()=>{
    if(timer){stop();return;}
    if(index===stages.length-1)updateStep(0);
    byId('play').textContent='Pause';byId('play').setAttribute('aria-pressed','true');
    timer=setInterval(()=>{updateStep(index+1);if(index===stages.length-1)stop();},6500);
  };
  byId('camera-reset').onclick=()=>engine.reset();
  byId('zoom-in').onclick=()=>engine.zoom(1.22);byId('zoom-out').onclick=()=>engine.zoom(1/1.22);
  phases.forEach(b=>b.onclick=()=>setPhase(b.dataset.phase));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  function followStep(i){
    stop();updateStep(i);byId('walk').scrollIntoView({block:'start'});
  }
  document.querySelectorAll('a[href^="#"]').forEach(a=>{
    const i=stages.findIndex(s=>a.hash==='#'+s.id);
    if(i>=0)a.addEventListener('click',event=>{event.preventDefault();followStep(i);});
  });
  window.addEventListener('hashchange',()=>{const i=stageFromHash();if(i>=0)followStep(i);});
  setPhase('decode');
  if(stageFromHash()>=0)byId('walk').scrollIntoView({block:'start'});

  function formatBytes(bytes) {
    const unit=bytes>=2**30?'GiB':'MiB', divisor=unit==='GiB'?2**30:2**20;
    return (bytes/divisor).toLocaleString('en-US',{maximumFractionDigits:3})+' '+unit;
  }
  const length=byId('length');
  function compare(L) {
    const fixed=48*(48*128*128*4+10240*4*2);
    const perBlock=2*4*256*2;
    const growing=16*perBlock*L,hybrid=fixed+growing,pure=64*perBlock*L,max=Math.max(hybrid,pure);
    for(const [id,value] of [['state-bar',fixed],['kv-bar',growing],['pure-bar',pure]])byId(id).style.width=(100*value/max)+'%';
    byId('hybrid-bytes').textContent=formatBytes(hybrid);byId('pure-bytes').textContent=formatBytes(pure);
    const intro=L===16||L===17?'The measured state shapes give ':'At this calculated length, the shapes give ';
    byId('memory-result').textContent=intro+formatBytes(fixed)+' of fixed state plus '+formatBytes(growing)+' of KV history. '+(hybrid===pure?'Both layouts retain the same number of bytes.':'The hybrid uses '+formatBytes(Math.abs(hybrid-pure))+(hybrid>pure?' more':' less')+' cache than the all-GQA example.')+' Each additional token adds 64 KiB to the hybrid and 256 KiB to the all-GQA example.';
    document.querySelectorAll('[data-length]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.length===L)));
  }
  const valid=()=>Number.isInteger(length.valueAsNumber)&&length.valueAsNumber>=1&&length.valueAsNumber<=262144;
  length.oninput=()=>{if(valid())compare(length.valueAsNumber);};
  length.onchange=()=>{if(!valid()){length.value=Math.max(1,Math.min(262144,Math.round(length.valueAsNumber)||17));compare(+length.value);}};
  document.querySelectorAll('[data-length]').forEach(b=>b.onclick=()=>{length.value=b.dataset.length;compare(+length.value);});
  compare(+length.value);
})();
