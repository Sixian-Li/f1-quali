// Execute the actual bilingual page applications with DOM/Plotly test doubles.
// This checks behavior and plot inputs, not browser rendering.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import vm from 'node:vm';

const root=resolve(import.meta.dirname,'..');
const hash=text=>createHash('sha256').update(text).digest('hex');
const originalDataHash='cfa7cd535d5e0dfb224b6abf936725735ba6b77a07b57444fe03ed5ffd37a199';
const originalPlotlyHash='122e3be346d66616944d0b83eaaf7242581508c3c1cfa0995a17af0d83eff770';
const paths={zh:process.argv[2]??resolve(root,'docs/index.html'),en:process.argv[3]??resolve(root,'docs/en.html')};
const pages=Object.fromEntries(Object.entries(paths).map(([lang,path])=>{
  const html=readFileSync(path,'utf8');
  const raw=html.match(/<script type="application\/json" id="ratings-data">(.*?)<\/script>/s)[1];
  const scripts=Array.from(html.matchAll(/<script>([\s\S]*?)<\/script>/g),m=>m[1]);
  const template=html.replace(/<script\b[^>]*>.*?<\/script>/gs,'');
  assert.equal(hash(raw),originalDataHash,'Every saved RM point and metadata must stay identical');
  assert.equal(hash(scripts[0]),originalPlotlyHash,'The bundled library is unchanged');
  assert.equal(scripts.length,2);
  assert(!/<script\b[^>]*\bsrc=/i.test(html));
  assert(!/<link\b[^>]*href=/i.test(html));
  assert(!/\b(fetch|XMLHttpRequest|WebSocket)\s*\(/.test(scripts[1]));
  assert(!html.includes('/Users/'));assert(!html.includes('artifacts/'));
  assert.match(html,/Permission is hereby granted, free of charge/);
  assert.match(template,/2026-09-25 11:00 UTC/);
  assert.match(template,/id="language-zh" href="index.html"/);
  assert.match(template,/id="language-en" href="en.html"/);
  assert.equal((template.match(/aria-current="page"/g)||[]).length,1);
  assert.match(template,new RegExp(`id="language-${lang}"[^>]*aria-current="page"`));
  assert.match(template,new RegExp(`<html lang="${lang==='zh'?'zh-CN':'en'}">`));
  if(lang==='en') {
    assert(!/[\u4e00-\u9fff]/.test(scripts[1]),'English app text must be fully translated');
    assert(!/[\u4e00-\u9fff]/.test(template.replace('中文','')),'Only the language name stays Chinese');
    assert.match(template,/Two-pass SMA/);assert.match(template,/neutral prior/);
  }
  return [lang,{html,template,javascript:scripts[1],data:JSON.parse(raw)}];
}));
assert.deepEqual(pages.zh.data,pages.en.data);
const data=pages.zh.data;
assert.deepEqual(data.models.map(m=>m.id),['RM']);
assert.equal(data.drivers.length,84);assert.equal(data.meta.pointCount,7248);
assert.equal(data.drivers.reduce((n,d)=>n+d.ratings.RM.length,0),7248);
assert.equal(data.drivers.filter(d=>d.current).length,22);

class Element {
  constructor(id) {this.id=id;this.value='';this.textContent='';this.innerHTML='';this.hidden=false;this.disabled=false;this.dataset={};this.attributes={};this.clientWidth=1100;this.scrollTop=0;this.listeners={};this.children=[];this.classList={toggle(){}};}
  addEventListener(type,fn){this.listeners[type]=fn;}
  on(type,fn){this.listeners[type]=fn;}
  append(element){this.children.push(element);}
  setAttribute(name,value){this.attributes[name]=String(value);}
  getAttribute(name){return this.attributes[name]??null;}
  focus(){}
  querySelector(){return null;}
  click(){return this.listeners.click?.({target:this});}
}
async function launch(lang,storage=new Map()) {
  const page=pages[lang];
  const nodes=new Map(Array.from(page.template.matchAll(/id="([a-z0-9-]+)"/g),m=>[m[1],new Element(m[1])]));
  nodes.set('ratings-data',new Element('ratings-data'));
  const get=id=>{assert(nodes.has(id),`Unknown element: ${id}`);return nodes.get(id);};
  get('ratings-data').textContent=JSON.stringify(page.data);get('driver-filter').value='all';
  const quick=['2010,2026','2019,2026','2026,2026'].map(years=>{const e=new Element();e.dataset.years=years;return e;});
  const errors=[],locales=[];
  const context={console:{error:e=>errors.push(String(e))},setTimeout,clearTimeout,Blob,URL,
    localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v)},
    document:{getElementById:get,activeElement:null,createElement:()=>new Element(),querySelectorAll:s=>{assert.equal(s,'[data-years]');return quick;}},
    Plotly:{register:locale=>locales.push(locale),async react(graph,traces,layout,config){
      for(const t of traces){assert.equal(t.x.length,t.y.length);if(t.customdata)assert.equal(t.x.length,t.customdata.length);assert.equal(t.connectgaps,false);}
      assert(layout.yaxis.range.every(Number.isFinite));
      graph.data=traces;graph.layout=layout;graph.config=config;
    },async relayout(graph,change){
      for(const [k,v] of Object.entries(change)){if(k==='yaxis.range')graph.layout.yaxis.range=v;}
      graph.listeners.plotly_relayout?.(change);
    }}
  };
  context.window=context;
  vm.runInNewContext(page.javascript,context,{filename:`rm_${lang}.js`,timeout:10000});
  const api=context.__f1Ratings;
  await api.settled;await new Promise(done=>setTimeout(done,0));
  assert(api.ready);assert.equal(errors.length,0);
  assert.equal(api.graph.config.locale,lang==='zh'?'zh-CN':'en-US');
  assert.equal(locales.length,lang==='zh'?1:0);
  if(lang==='zh')assert.equal(locales[0].dictionary['Zoom in'],'放大');
  const click=async id=>{get(id).click();await api.settled;};
  const change=async(id,value,event='change')=>{const e=get(id);e.value=String(value);e.listeners[event]({target:e});await api.settled;};
  return {api,get,context,errors,click,change,storage,quick};
}
const average=(values,n)=>values.map((_,i)=>{
  const window=values.slice(Math.max(0,i-n+1),i+1);return window.reduce((a,b)=>a+b,0)/window.length;
});
const rawBefore=JSON.stringify(data.drivers.map(d=>d.ratings));
for(const lang of ['zh','en']) {
  const {api,get,context,errors,click,change,storage,quick}=await launch(lang);
  const traces=()=>api.graph.data.filter(t=>t.meta.kind==='rating');
  assert.equal(api.state.model,'RM');assert.equal(api.state.mode,'raw');assert.equal(api.state.scale,'auto');
  assert.equal(api.state.selected.size,8);assert.equal(traces().length,8);
  assert.equal(api.state.from,2025);assert.equal(api.state.to,2026);
  assert.equal(api.state.firstWindow,3);assert.equal(api.state.secondWindow,3);
  await change('smoothing','smooth');await change('sma-first',7,'input');await change('sma-second',2,'input');
  for(const trace of traces()) {
    const driver=data.drivers.find(d=>d.id===trace.meta.driver);
    const expected=average(average(driver.ratings.RM.map(p=>p[1]),7),2);
    Array.from(trace.y).filter(v=>v!==null).forEach((v,i)=>assert(Math.abs(v-expected[i])<1e-10));
  }
  await change('sma-first',0,'input');assert.equal(api.state.firstWindow,7);
  assert.equal(get('smoothing-error').hidden,false);
  assert.match(get('smoothing-error').textContent,lang==='en'?/whole numbers/:/整数/);
  await change('sma-first',7,'input');assert.equal(get('smoothing-error').hidden,true);
  await click('ant-rus');assert.deepEqual(Array.from(api.state.selected).sort(),['george-russell','kimi-antonelli']);
  assert.equal(api.state.from,2026);assert.equal(api.state.to,2026);
  const rows=api.exportRows();assert.equal(rows.length,30);
  assert(rows.every(r=>r[0]==='RM'&&r[4]===2026&&r[15]===7&&r[16]===2));
  const endpoints=rows.filter(r=>r[6]==='current_after_round');assert.equal(endpoints.length,2);
  assert.equal(endpoints.find(r=>r[1]==='kimi-antonelli')[10].toFixed(2),'84.34');
  assert.equal(endpoints.find(r=>r[1]==='george-russell')[10].toFixed(2),'86.64');
  assert.match(api.csvText(),/"first_sma_window","second_sma_window"/);
  await context.Plotly.relayout(api.graph,{'xaxis.range[0]':'2026-06-01','xaxis.range[1]':'2026-09-26'});
  assert(api.exportRows().every(r=>Date.parse(r[7])>=Date.parse('2026-06-01')));
  const other=lang==='zh'?'en':'zh';
  await click(`language-${other}`);
  const switched=await launch(other,storage);
  for(const property of ['model','from','to','mode','scale','firstWindow','secondWindow'])assert.equal(switched.api.state[property],api.state[property]);
  assert.deepEqual(Array.from(switched.api.state.selected),Array.from(api.state.selected));
  assert.deepEqual(Array.from(switched.api.state.range),Array.from(api.state.range));
  assert.equal(switched.api.csvText(),api.csvText(),'CSV is language-independent, including smoothing');
  if(lang==='en') {
    for(const id of ['driver-count','source-note','mode-note','selection-summary','summary-body','driver-list','selected-chips']) {
      assert(!/[\u4e00-\u9fff]/.test(get(id).textContent+get(id).innerHTML),id);
    }
    assert.match(api.graph.layout.yaxis.title.text,/Qualifying rating/);
    assert(traces().every(t=>t.hovertemplate.includes('Raw')));
    assert(traces().flatMap(t=>t.customdata).filter(Boolean).every(d=>!/[\u4e00-\u9fff]/.test(d.join(' '))));
  }
  await change('y-scale','fixed');assert.deepEqual(Array.from(api.graph.layout.yaxis.range),[1,100]);
  await click('reset-zoom');assert.equal(new Date(api.state.range[0]).getUTCMonth(),0);
  await change('driver-filter','all');await change('driver-search','安东内利','input');
  assert.match(get('driver-list').innerHTML,/Kimi Antonelli/);assert.doesNotMatch(get('driver-list').innerHTML,/Lewis Hamilton/);
  await change('driver-search','Raikkonen','input');assert.match(get('driver-list').innerHTML,/Räikkönen/);
  await change('driver-search','','input');await click('clear');assert.equal(get('empty-chart').hidden,false);
  await click('select-visible');assert.equal(traces().length,84);
  quick[0].click();await api.settled;
  assert.equal(traces().find(t=>t.meta.driver==='andre-lotterer').x.length,1);
  const alonso=data.drivers.find(d=>d.id==='fernando-alonso');
  assert.equal(alonso.ratings.RM.find(p=>data.slots[p[0]].year===2021)[3],0);
  await click('favorites');assert.equal(api.state.selected.size,8);
  await change('sma-first',1,'input');await change('sma-second',1,'input');
  assert(api.exportRows().every(r=>Math.abs(r[10]-r[11])<1e-12));
  await change('smoothing','raw');assert(api.exportRows().every(r=>r[10]===r[13]&&r[12]==='raw'));
  assert.equal(errors.length,0);
  assert.equal(JSON.stringify(api.data.drivers.map(d=>d.ratings)),rawBefore);
}
console.log(JSON.stringify({status:'PASS',scope:'actual_application_logic_not_browser_render',languages:['zh-CN','en'],
  drivers:84,points_per_language:7248,original_RM_data_exact:true,Plotly_bundle_exact:true,
  language_switch_preserves_controls_and_zoom:true,localized_static_dynamic_and_accessibility_text:true,
  causal_smoothing_and_CSV_equal:true,all_drivers_gaps_search_and_shortcuts:true},null,2));
