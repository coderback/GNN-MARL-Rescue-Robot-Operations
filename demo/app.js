(function(){
const $=id=>document.getElementById(id);
const NS='http://www.w3.org/2000/svg';
const canvas=$('map'),ctx=canvas.getContext('2d');
const MODE_LABEL={basic:'Basic',extended:'Extended',novel:'Novel'};
const KIND_LABEL={robot:'Robot',drone:'Drone',charger:'Charger'};
const CAP={robot:100,drone:150,charger:1000};
const reduce=matchMedia('(prefers-reduced-motion: reduce)');
let run=null,t=0,playing=false,speed=6,last=0,mode='extended',runId=0,trails=true,colors={},seq=0;
const cache={};

function readColors(){const s=getComputedStyle(document.documentElement);
  ['--t0','--t1','--t2','--t3','--contour','--robot','--drone','--charger','--missing','--rescued','--ink','--base','--bg'].forEach(k=>colors[k]=s.getPropertyValue(k).trim())}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{readColors();draw()});

function sizeCanvas(){const r=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),px=Math.round(r.width*dpr);
  if(canvas.width!==px){canvas.width=px;canvas.height=px}return px}
function subOffset(i){const col=i%3,row=Math.floor(i/3)%3;return [(col-1)*.27,(row-1)*.27]}

function drawTerrain(c,n,g){
  const terr=run.terrain;
  for(let x=0;x<n;x++)for(let y=0;y<n;y++){ctx.fillStyle=colors['--t'+terr[x][y]];ctx.fillRect(x*g,(n-1-y)*g,g+1,g+1)}
  ctx.strokeStyle=colors['--contour'];ctx.lineWidth=Math.max(1,g*.035);ctx.beginPath();
  for(let x=0;x<n;x++)for(let y=0;y<n;y++){
    if(x+1<n&&terr[x][y]!==terr[x+1][y]){ctx.moveTo((x+1)*g,(n-1-y)*g);ctx.lineTo((x+1)*g,(n-y)*g)}
    if(y+1<n&&terr[x][y]!==terr[x][y+1]){ctx.moveTo(x*g,(n-1-y)*g);ctx.lineTo((x+1)*g,(n-1-y)*g)}}
  ctx.stroke();
  /* base */
  const [bx,by]=run.base;ctx.strokeStyle=colors['--base'];ctx.lineWidth=Math.max(1.5,g*.06);ctx.strokeRect(bx*g+g*.06,(n-1-by)*g+g*.06,g*.88,g*.88);
}
function drawPerson(p,step,n,g){
  if(p.born>step)return;
  const cx=(p.x+.5)*g,cy=(n-1-p.y+.5)*g,s=g*.2,done=p.rescued!=null&&p.rescued<=step;
  ctx.lineCap='round';
  if(done){ctx.strokeStyle=colors['--rescued'];ctx.lineWidth=Math.max(1.2,g*.05);ctx.beginPath();ctx.moveTo(cx-s,cy);ctx.lineTo(cx-s*.2,cy+s*.8);ctx.lineTo(cx+s,cy-s*.7);ctx.stroke();return}
  ctx.fillStyle=colors['--bg'];ctx.beginPath();ctx.arc(cx,cy,s*1.35,0,7);ctx.fill();
  ctx.strokeStyle=colors['--missing'];ctx.lineWidth=Math.max(1.6,g*.075);ctx.beginPath();ctx.moveTo(cx-s*.8,cy-s*.8);ctx.lineTo(cx+s*.8,cy+s*.8);ctx.moveTo(cx+s*.8,cy-s*.8);ctx.lineTo(cx-s*.8,cy+s*.8);ctx.stroke();
}
function shape(kind,x,y,s,fill,stroke){ctx.fillStyle=fill;ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.beginPath();
  if(kind==='robot'){ctx.rect(x-s,y-s,2*s,2*s)}
  else if(kind==='drone'){ctx.moveTo(x,y-s*1.25);ctx.lineTo(x+s*1.2,y+s*.95);ctx.lineTo(x-s*1.2,y+s*.95);ctx.closePath()}
  else{ctx.moveTo(x,y-s*1.3);ctx.lineTo(x+s*1.3,y);ctx.lineTo(x,y+s*1.3);ctx.lineTo(x-s*1.3,y);ctx.closePath()}
  ctx.fill();ctx.stroke()}
function draw(){
  if(!run)return;
  const px=sizeCanvas(),n=run.terrain.length,g=px/n,step=Math.floor(t),f=reduce.matches?0:t-step,nx=Math.min(step+1,run.series.rescued.length-1);
  ctx.clearRect(0,0,px,px);drawTerrain(ctx,n,g);
  run.persons.forEach(p=>drawPerson(p,step,n,g));
  run.tracks.forEach((a,i)=>{const [ox,oy]=subOffset(i),col=colors['--'+a.kind];
    const pos=k=>[(a.x[k]+.5+ox)*g,(n-1-a.y[k]+.5+oy)*g];
    if(trails){ctx.strokeStyle=col;ctx.lineWidth=Math.max(1,g*.03);ctx.globalAlpha=.42;ctx.beginPath();for(let k=Math.max(0,step-14);k<=step;k++){const [x,y]=pos(k);k===Math.max(0,step-14)?ctx.moveTo(x,y):ctx.lineTo(x,y)}ctx.stroke();ctx.globalAlpha=1}
    const [x0,y0]=pos(step),[x1,y1]=pos(nx);shape(a.kind,x0+(x1-x0)*f,y0+(y1-y0)*f,g*.115,col,colors['--bg'])});
  updatePanel(step);
}
const stateText=s=>s.replace(/_/g,' ');
let rows=[];
function buildPanel(){
  const list=$('agents');list.textContent='';rows=[];const count={robot:0,drone:0,charger:0};
  run.tracks.forEach(a=>{count[a.kind]++;const li=document.createElement('li');
    const icon=document.createElementNS(NS,'svg');icon.setAttribute('viewBox','0 0 12 12');icon.setAttribute('width','12');icon.setAttribute('height','12');icon.setAttribute('aria-hidden','true');
    const p=document.createElementNS(NS,'path');p.setAttribute('fill',`var(--${a.kind})`);p.setAttribute('d',a.kind==='robot'?'M1 1h10v10H1z':a.kind==='drone'?'M6 0l6 11H0z':'M6 0l6 6-6 6-6-6z');icon.append(p);
    const name=document.createElement('span');name.className='a-name';name.append(icon,` ${KIND_LABEL[a.kind]} ${count[a.kind]}`);
    const st=document.createElement('span');st.className='a-state';
    const bar=document.createElement('span');bar.className='a-bar';const fill=document.createElement('i');bar.append(fill);
    const num=document.createElement('span');num.className='a-num';
    li.append(name,st,bar,num);list.append(li);rows.push({st,fill,num,a})});
  const scrub=$('scrub');scrub.max=run.series.rescued.length-1;
  buildSpark();
  $('run-note').textContent=runNote();
}
function runNote(){const m=run.meta;return `${MODE_LABEL[m.mode]}, seed ${m.seed}: recorded with the real Mesa model on ${run.terrain.length}×${run.terrain.length} terrain, 500 steps.`}
function updatePanel(step){
  const s=run.series,res=s.rescued[step],active=run.persons.filter(p=>p.born<=step&&!(p.rescued!=null&&p.rescued<=step)).length;
  $('m-step').textContent=step;$('m-rescued').textContent=res;$('m-missing').textContent=active;$('m-messages').textContent=s.messages[step];
  $('m-persons').textContent=s.persons[step];
  rows.forEach(r=>{const a=r.a,b=a.b[step],cap=a.cap||CAP[a.kind];r.st.textContent=stateText(run.states[a.s[step]]);r.fill.style.width=Math.min(100,100*b/cap)+'%';r.fill.dataset.low=b<cap*.2?'1':'0';r.num.textContent=Math.round(b)});
  const sc=$('scrub');if(document.activeElement!==sc||playing)sc.value=step;
  sc.setAttribute('aria-valuetext',`Step ${step}, ${res} rescued`);
  const mk=$('spark-mark');if(mk){const x=step/(s.rescued.length-1)*400;mk.setAttribute('x1',x);mk.setAttribute('x2',x)}
  const eps=$('m-eps-wrap');if(s.epsilon){eps.hidden=false;$('m-eps').textContent=s.epsilon[step].toFixed(3)}else eps.hidden=true;
}
function buildSpark(){const host=$('spark');host.textContent='';const s=run.series.rescued,max=Math.max(1,run.series.persons[run.series.persons.length-1]),len=s.length-1;
  const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','0 0 400 44');svg.setAttribute('preserveAspectRatio','none');svg.setAttribute('role','img');
  svg.setAttribute('aria-label',`Rescues over the run: ${s[len]} by step ${len}`);
  const pl=document.createElementNS(NS,'polyline');pl.setAttribute('points',s.map((v,i)=>`${i/len*400},${42-40*v/max}`).join(' '));pl.setAttribute('class','spark-line');pl.setAttribute('vector-effect','non-scaling-stroke');
  const m=document.createElementNS(NS,'line');m.id='spark-mark';m.setAttribute('y1',0);m.setAttribute('y2',44);m.setAttribute('class','spark-mark');m.setAttribute('vector-effect','non-scaling-stroke');svg.append(pl,m);host.append(svg)}

function setPlaying(p){playing=p;const b=$('play');b.textContent=p?'Pause':'Play';b.setAttribute('aria-pressed',p);if(p){last=performance.now();requestAnimationFrame(tick)}}
function tick(now){if(!playing)return;const dt=Math.min(.25,(now-last)/1000);last=now;t+=dt*speed;const end=run.series.rescued.length-1;
  if(t>=end){t=end;draw();setPlaying(false);return}draw();requestAnimationFrame(tick)}
async function load(m,id,keepT){const key=`${m}-${id}`,token=++seq;$('status').textContent=`Loading recorded run ${MODE_LABEL[m]}, seed ${42+id*100}…`;
  try{if(!cache[key])cache[key]=await fetch(`runs/${key}.json`).then(r=>{if(!r.ok)throw new Error(r.status);return r.json()})}
  catch(e){$('status').textContent='This recorded run could not be loaded. Try another run.';return}
  if(token!==seq)return;run=cache[key];mode=m;runId=id;if(!keepT)t=0;t=Math.min(t,run.series.rescued.length-1);$('status').textContent='';
  buildPanel();draw();document.querySelectorAll('[name=mode]').forEach(r=>r.checked=r.value===m);
  $('mode-blurb').textContent=$('mode-blurb').dataset[m]}

document.querySelectorAll('[name=mode]').forEach(r=>r.addEventListener('change',e=>load(e.target.value,runId,true)));
$('seed').addEventListener('change',e=>load(mode,+e.target.value,true));
$('play').addEventListener('click',()=>{if(!run)return;if(t>=run.series.rescued.length-1)t=0;setPlaying(!playing)});
$('restart').addEventListener('click',()=>{t=0;draw()});
$('speed').addEventListener('change',e=>{speed=+e.target.value});
$('trails').addEventListener('change',e=>{trails=e.target.checked;draw()});
$('scrub').addEventListener('input',e=>{t=+e.target.value;draw()});
addEventListener('resize',draw);
readColors();

/* summary charts and tables */
function el(tag,attrs,parent,text){const n=document.createElementNS(NS,tag);for(const k in attrs)n.setAttribute(k,attrs[k]);if(text!=null)n.textContent=text;if(parent)parent.appendChild(n);return n}
const tip=$('tip');
function showTip(text,x,y){tip.textContent=text;tip.hidden=false;const r=tip.getBoundingClientRect();tip.style.left=Math.min(Math.max(8,x+12),innerWidth-r.width-8)+'px';tip.style.top=Math.max(8,y-r.height-10)+'px'}
function hideTip(){tip.hidden=true}
function legend(host,items){const d=document.createElement('p');d.className='legend';items.forEach(([c,t,dash])=>{const s=document.createElement('span');const i=document.createElement('i');i.style.borderTop=`2.5px ${dash?'dashed':'solid'} var(${c})`;s.append(i,t);d.append(s)});host.appendChild(d)}

function lineChart(host,cfg){
  const {xs,series,yMax,yTicks,xTicks,xLabel,fmt,title}=cfg;legend(host,series.map(s=>[s.color,s.label,s.dash]));
  const W=620,H=270,L=38,R=12,T=8,B=34,X=v=>L+(W-L-R)*(v-xs[0])/(xs[xs.length-1]-xs[0]),Y=v=>T+(H-T-B)*(1-v/yMax);
  const s=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':title},host);
  const g=el('g',{class:'grid'},s),a=el('g',{class:'axis'},s);
  yTicks.forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v)},g);el('text',{x:L-6,y:Y(v)+4,'text-anchor':'end'},a,fmt(v))});
  xTicks.forEach(v=>el('text',{x:X(v),y:H-16,'text-anchor':'middle'},a,v));
  el('text',{x:(L+W-R)/2,y:H-2,'text-anchor':'middle'},a,xLabel);
  series.forEach(se=>{el('polyline',{points:se.values.map((v,i)=>X(xs[i])+','+Y(v)).join(' '),fill:'none',stroke:`var(${se.color})`,'stroke-width':se.thin?1:2,opacity:se.thin?.35:1,'stroke-dasharray':se.dash?'6 4':'none'},s)});
  const cross=el('line',{y1:T,y2:H-B,stroke:'var(--ink2)',visibility:'hidden'},s);
  const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent',tabindex:0,class:'hitbox'},s);
  let idx=xs.length-1;
  function show(i,cx,cy){idx=Math.max(0,Math.min(xs.length-1,i));const x=X(xs[idx]);cross.setAttribute('x1',x);cross.setAttribute('x2',x);cross.setAttribute('visibility','visible');
    showTip(`${xLabel} ${xs[idx]}: `+series.filter(z=>!z.noTip).map(z=>`${z.label} ${fmt(z.values[idx])}`).join(', '),cx,cy)}
  hit.addEventListener('pointermove',e=>{const r=s.getBoundingClientRect(),vx=(e.clientX-r.left)/r.width*W,v=xs[0]+(vx-L)/(W-L-R)*(xs[xs.length-1]-xs[0]);let best=0;xs.forEach((q,i)=>{if(Math.abs(q-v)<Math.abs(xs[best]-v))best=i});show(best,e.clientX,e.clientY)});
  hit.addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');hideTip()});
  hit.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();const r=hit.getBoundingClientRect();show(idx+(e.key==='ArrowRight'?1:-1),r.left+r.width/2,r.top)}});
  hit.addEventListener('blur',()=>{cross.setAttribute('visibility','hidden');hideTip()});
}
fetch('data/summary.json').then(r=>r.json()).then(d=>{
  const c=[['basic','--c1','Basic',false],['extended','--c2','Extended',false],['novel','--c3','Novel',true]];
  lineChart($('c-rescues'),{xs:d.steps,series:c.map(([k,col,lab,dash])=>({label:lab,color:col,dash,values:d.meanRescued[k]})),yMax:Math.ceil(Math.max(...c.flatMap(([k])=>d.meanRescued[k]))/4)*4,
    yTicks:[0,4,8,12,16].filter(v=>v<=Math.ceil(Math.max(...c.flatMap(([k])=>d.meanRescued[k]))/4)*4),xTicks:[0,100,200,300,400,500],xLabel:'Step',fmt:v=>(+v).toFixed(1).replace(/\.0$/,''),title:'Mean people rescued over 500 steps across five recorded runs for each mode'});
  const body=$('runs-body');
  d.runs.forEach(r=>{const tr=document.createElement('tr');const th=document.createElement('th');th.scope='row';const b=document.createElement('button');b.type='button';b.className='linkish';b.textContent=`${MODE_LABEL[r.mode]}, seed ${r.seed}`;
    b.addEventListener('click',()=>{load(r.mode,r.runId,false);$('seed').value=r.runId;$('replay').scrollIntoView({block:'start'})});th.append(b);tr.append(th);
    [r.rescued,r.personsTotal,(r.rescueRate*100).toFixed(1)+'%',r.mode==='basic'?'n/a':r.messages].forEach(v=>{const td=document.createElement('td');td.className='n';td.textContent=v;tr.append(td)});
    body.append(tr)});
  /* training */
  const roll=d.training.rolling,xs=roll.map((_,i)=>i+1);
  lineChart($('c-train'),{xs,series:[{label:'Episode',color:'--ink2',thin:true,values:d.training.rates,noTip:false},{label:'Rolling mean of 25',color:'--c3',values:roll}],yMax:.5,yTicks:[0,.1,.2,.3,.4,.5],xTicks:[1,50,100,150,200,250],xLabel:'Episode',fmt:v=>(+v).toFixed(2),title:'Rescue rate per training episode over 250 episodes, with a rolling mean; no upward trend'});
});
load('extended',0,false).then(()=>{if(!reduce.matches)setPlaying(true)});
})();
