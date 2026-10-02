// Execute the actual bilingual page applications with DOM/Plotly test doubles.
// This checks behavior and plot inputs, not browser rendering.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {basename,resolve} from 'node:path';
import vm from 'node:vm';

const root=resolve(import.meta.dirname,'..');
const hash=text=>createHash('sha256').update(text).digest('hex');
// Pinned from the previously published RM snapshot, before adding the display mapping.
const originalRecordsHash='542acd4f91676cf012c278616cd0ba86fb5098c02905817ffddadcd9d57a016c';
const mappingHash='6082ba7c2ec86228452b035b9f89c4a0a2bbfc6d6ea9a09452ba66eee1fcffe8';
const originalPlotlyHash='122e3be346d66616944d0b83eaaf7242581508c3c1cfa0995a17af0d83eff770';
const expectedMean=6.5,expectedSD=2;
const paths={zh:process.argv[2]??resolve(root,'docs/index.html'),en:process.argv[3]??resolve(root,'docs/en.html')};
const pages=Object.fromEntries(Object.entries(paths).map(([lang,path])=>{
  const html=readFileSync(path,'utf8');
  const raw=html.match(/<script type="application\/json" id="ratings-data">(.*?)<\/script>/s)[1];
  const scripts=Array.from(html.matchAll(/<script>([\s\S]*?)<\/script>/g),m=>m[1]);
  const template=html.replace(/<script\b[^>]*>.*?<\/script>/gs,'');
  const parsed=JSON.parse(raw);
  assert.equal(hash(JSON.stringify({drivers:parsed.drivers,slots:parsed.slots})),originalRecordsHash,'All original RM scores, default smoothing, identities and gaps are preserved');
  assert.equal(hash(JSON.stringify(parsed.mapping)),mappingHash,'The reviewed frozen mapping is unchanged');
  assert.equal(hash(scripts[0]),originalPlotlyHash,'The bundled library is unchanged');
  assert.equal(scripts.length,2);
  assert(!/<script\b[^>]*\bsrc=/i.test(html));
  assert(!/<link\b[^>]*href=/i.test(html));
  assert(!/\b(fetch|XMLHttpRequest|WebSocket)\s*\(/.test(scripts[1]));
  assert(!html.includes('/Users/'));assert(!html.includes('artifacts/'));
  assert.match(html,/Permission is hereby granted, free of charge/);
  assert.match(template,/2026-09-25 11:00 UTC/);
  assert(template.includes(`id="language-zh" href="${basename(paths.zh)}"`));
  assert(template.includes(`id="language-en" href="${basename(paths.en)}"`));
  assert.equal((template.match(/aria-current="page"/g)||[]).length,1);
  assert.match(template,new RegExp(`id="language-${lang}"[^>]*aria-current="page"`));
  assert.match(template,new RegExp(`<html lang="${lang==='zh'?'zh-CN':'en'}">`));
  if(lang==='en') {
    assert(!/[\u4e00-\u9fff]/.test(scripts[1]),'English app text must be fully translated');
    assert(!/[\u4e00-\u9fff]/.test(template.replace('中文','')),'Only the language name stays Chinese');
    assert.match(template,/Fixed 3→4 SMA/);assert.match(template,/neutral prior/);
  }
  assert(!/id="(?:smoothing|sma-first|sma-second)"/.test(template),'Fixed display has no smoothing controls');
  return [lang,{html,template,javascript:scripts[1],data:JSON.parse(raw)}];
}));
assert.deepEqual(pages.zh.data,pages.en.data);
const data=pages.zh.data;
assert.deepEqual(data.models.map(m=>m.id),['RM']);
assert.equal(data.drivers.length,84);assert.equal(data.meta.pointCount,7248);
assert.equal(data.drivers.reduce((n,d)=>n+d.ratings.RM.length,0),7248);
assert.equal(data.drivers.filter(d=>d.current).length,22);

class Element {
  constructor(id) {this.id=id;this.value='';this.textContent='';this.innerHTML='';this.hidden=false;this.disabled=false;this.dataset={};this.attributes={};this.style={};this.clientWidth=1100;this.scrollTop=0;this.listeners={};this.children=[];this.classList={toggle(){}};}
  addEventListener(type,fn){this.listeners[type]=fn;}
  on(type,fn){this.listeners[type]=fn;}
  append(element){this.children.push(element);}
  setAttribute(name,value){this.attributes[name]=String(value);}
  getAttribute(name){return this.attributes[name]??null;}
  getBoundingClientRect(){return this.id==='rating-tooltip'?{left:0,top:0,width:330,height:70+42*(this.innerHTML.match(/class="hover-row"/g)||[]).length}:{left:300,top:100,width:1100,height:600};}
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
  const context={console:{error:e=>errors.push(String(e))},setTimeout,clearTimeout,Blob,URL,innerWidth:1440,innerHeight:900,addEventListener(){},
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
const knots=data.mapping.knots;
const interpolate=value=>{
  let upper=knots.findIndex(k=>k[0]>=value);
  if(upper>=0 && knots[upper][0]===value)return knots[upper][1];
  if(upper<0)upper=knots.length-1;
  upper=Math.max(1,upper);
  const a=knots[upper-1],b=knots[upper];
  return a[1]+(value-a[0])*(b[1]-a[1])/(b[0]-a[0]);
};
assert.equal(data.mapping.mean,expectedMean);assert.equal(data.mapping.sd,expectedSD);
assert.equal(data.mapping.cap,null);assert.equal(data.mapping.sampleCount,7226);
assert.equal(data.mapping.excludedCurrentEndpoints,22);
assert.equal(data.mapping.id,'rm-normal-95a8937f20a9e29a');
assert.equal(data.mapping.retrospectiveDisplayOnly,true);
assert.equal(data.mapping.smoothingOrder,'smooth_RM_then_map');
assert.equal(knots.length,7139);
assert(knots.every((k,i)=>k.every(Number.isFinite)&&(i===0||(k[0]>knots[i-1][0]&&k[1]>knots[i-1][1]&&k[2]>knots[i-1][2]))));
const rawBefore=JSON.stringify(data.drivers.map(d=>d.ratings));
const expectedByDriver=new Map(data.drivers.map(d=>[d.id,
  average(average(d.ratings.RM.map(p=>p[1]),3),4).map(interpolate)]));
const expectedGlobal=Array.from(expectedByDriver.values()).flat();
const storageKey=`f1-rating-explorer:rm-normal-6p5_sd2:${data.meta.artifact}`;
for(const lang of ['zh','en']) {
  const {api,get,context,errors,click,change,storage,quick}=await launch(lang);
  const traces=()=>api.graph.data.filter(t=>t.meta.kind==='rating');
  const choose=async ids=>{
    await click('clear');
    for(const id of ids){get('driver-list').listeners.change({target:{dataset:{driver:id},checked:true}});await api.settled;}
  };
  const hover=(driverId,slotIndex)=>{
    const trace=traces().find(t=>t.meta.driver===driverId);
    const index=trace.customdata.findIndex(p=>p?.[0]===slotIndex);
    assert(index>=0);
    api.graph.listeners.plotly_hover({points:[{data:trace,customdata:trace.customdata[index]}],event:{clientX:1350,clientY:850}});
    return get('rating-tooltip');
  };
  assert.equal(api.state.model,'RM');assert.equal(api.state.scale,'auto');
  assert.equal(api.state.selected.size,8);assert.equal(traces().length,8);
  assert.equal(api.state.from,2025);assert.equal(api.state.to,2026);
  assert.equal(api.displayPolicy.firstWindow,3);assert.equal(api.displayPolicy.secondWindow,4);
  assert(Object.isFrozen(api.displayPolicy));
  for(const key of ['mode','firstWindow','secondWindow'])assert(!(key in api.state));
  for(const knot of knots)assert.equal(api.mapScore(knot[0]),knot[1]);
  assert(api.mapScore(knots.at(-1)[0]+.01)>knots.at(-1)[1]);
  assert(api.mapScore(knots[0][0]-.01)<knots[0][1]);
  const past=[12,25,17,40,15];
  assert.deepEqual(Array.from(api.smoothScores(past,3,4)),Array.from(api.smoothScores([...past,99],3,4)).slice(0,-1));
  for(const trace of traces()) {
    const expected=expectedByDriver.get(trace.meta.driver);
    Array.from(trace.y).filter(v=>v!==null).forEach((v,i)=>assert(Math.abs(v-expected[i])<1e-10));
  }
  // Reproduce the screenshot's three-driver comparison: one event header and one final score per driver.
  const trio=['charles-leclerc','carlos-sainz-jr','max-verstappen'];
  await choose(trio);await change('year-from',2023);await change('year-to',2023);
  const slotIndex=data.slots.findIndex(s=>s.year===2023&&s.round===4&&s.kind!=='current_after_round');
  const tip=hover('charles-leclerc',slotIndex);
  assert.equal(tip.hidden,false);
  assert.equal((tip.innerHTML.match(/class="hover-event"/g)||[]).length,1);
  assert.equal((tip.innerHTML.match(/Azerbaijan GP/g)||[]).length,1);
  assert.equal((tip.innerHTML.match(/UTC/g)||[]).length,1);
  assert.equal((tip.innerHTML.match(/class="hover-row"/g)||[]).length,3);
  assert(!/映射前|Original RM|SMA|均线/.test(tip.innerHTML));
  for(const id of trio){
    const d=data.drivers.find(d=>d.id===id),index=d.ratings.RM.findIndex(p=>p[0]===slotIndex);
    assert(tip.innerHTML.includes(d.name));
    assert(tip.innerHTML.includes(`class="hover-score">${expectedByDriver.get(id)[index].toFixed(2)}`));
    const last=expectedByDriver.get(id).at(-1).toFixed(2);
    const picker=get('driver-list').innerHTML.split('<label ').find(row=>row.includes(`data-driver="${id}"`));
    assert(picker.includes(`>${last}</span>`),'Picker uses the final smoothed rating');
    const visible=d.ratings.RM.map((p,i)=>({p,i})).filter(({p})=>data.slots[p[0]].year===2023);
    const summary=get('summary-body').innerHTML.split('<tr>').find(row=>row.includes(d.name));
    assert(summary.includes(`<strong>${expectedByDriver.get(id)[visible.at(-1).i].toFixed(2)}</strong>`),'Summary matches curve');
  }
  assert(parseFloat(tip.style.left)>=12&&parseFloat(tip.style.left)+330<=context.innerWidth-12);
  assert(parseFloat(tip.style.top)>=12&&parseFloat(tip.style.top)+tip.getBoundingClientRect().height<=context.innerHeight-12);
  if(lang==='en')assert(!/[\u4e00-\u9fff]/.test(tip.innerHTML));
  api.graph.listeners.plotly_unhover();assert.equal(tip.hidden,true);
  hover('charles-leclerc',slotIndex);
  await change('year-to',2024);assert.equal(tip.hidden,true,'Rerender clears stale hover');

  await click('ant-rus');assert.deepEqual(Array.from(api.state.selected).sort(),['george-russell','kimi-antonelli']);
  const rows=api.exportRows();assert.equal(rows.length,30);
  assert(rows.every(r=>r[0]==='RM'&&r[4]===2026&&r[15]===3&&r[16]===4));
  const endpoints=rows.filter(r=>r[6]==='current_after_round');assert.equal(endpoints.length,2);
  assert.equal(endpoints.find(r=>r[1]==='kimi-antonelli')[17].toFixed(2),'84.34');
  assert.equal(endpoints.find(r=>r[1]==='george-russell')[17].toFixed(2),'86.64');
  assert.match(api.csvText(),/"first_sma_window","second_sma_window","rm_raw_score","rm_smoothed_score"/);
  assert(rows.every(r=>r[19]===expectedMean&&r[20]===expectedSD&&r[21]===data.mapping.id&&r[22]==='smooth_RM_then_map'));
  for(const row of rows) {
    assert(Math.abs(row[10]-interpolate(row[17]))<1e-10);
    assert(Math.abs(row[11]-interpolate(row[18]))<1e-10);
    assert.equal(row[11],row[13]);assert.equal(row[12],'fixed_sma3_sma4');assert.equal(row[24],api.displayPolicy.id);
  }
  await context.Plotly.relayout(api.graph,{'xaxis.range[0]':'2026-06-01','xaxis.range[1]':'2026-09-26'});
  assert(api.exportRows().every(r=>Date.parse(r[7])>=Date.parse('2026-06-01')));
  const other=lang==='zh'?'en':'zh';await click(`language-${other}`);
  const switched=await launch(other,storage);
  for(const property of ['model','from','to','scale'])assert.equal(switched.api.state[property],api.state[property]);
  assert.deepEqual(Array.from(switched.api.state.selected),Array.from(api.state.selected));
  assert.deepEqual(Array.from(switched.api.state.range),Array.from(api.state.range));
  assert.equal(switched.api.csvText(),api.csvText(),'CSV and fixed display are language-independent');
  // Existing saved raw mode and custom windows cannot change the new rating definition.
  const legacy=JSON.parse(storage.get(storageKey));
  storage.set(storageKey,JSON.stringify({...legacy,mode:'raw',firstWindow:99,secondWindow:1}));
  const restored=await launch(lang,storage);
  assert.equal(restored.api.csvText(),api.csvText());
  assert.equal(restored.api.displayPolicy.secondWindow,4);
  assert(!('firstWindow' in JSON.parse(storage.get(storageKey))));
  if(lang==='en') {
    for(const id of ['driver-count','source-note','selection-summary','summary-body','driver-list','selected-chips'])assert(!/[\u4e00-\u9fff]/.test(get(id).textContent+get(id).innerHTML),id);
    assert.equal(api.graph.layout.yaxis.title.text,'Rating');
  }
  await change('y-scale','fixed');
  assert.deepEqual(Array.from(api.graph.layout.yaxis.range),[Math.floor(Math.min(...expectedGlobal)-.2),Math.ceil(Math.max(...expectedGlobal)+.2)]);
  await click('reset-zoom');assert.equal(new Date(api.state.range[0]).getUTCMonth(),0);
  await change('driver-filter','all');await change('driver-search','安东内利','input');
  assert.match(get('driver-list').innerHTML,/Kimi Antonelli/);assert.doesNotMatch(get('driver-list').innerHTML,/Lewis Hamilton/);
  await change('driver-search','Raikkonen','input');assert.match(get('driver-list').innerHTML,/Räikkönen/);
  await change('driver-search','','input');await click('clear');assert.equal(get('empty-chart').hidden,false);
  await click('select-visible');assert.equal(traces().length,84);
  quick[0].click();await api.settled;
  assert(traces().flatMap(t=>Array.from(t.y)).some(v=>v>10));
  assert(api.graph.layout.yaxis.range[1]>Math.max(...expectedGlobal));
  assert(api.graph.layout.yaxis.range[0]<Math.min(...expectedGlobal));
  for(const trace of traces()) {
    assert.equal(trace.mode,'lines');assert.equal(trace.line.shape,'spline');assert.equal(trace.hoverinfo,'none');
    assert.equal(trace.marker,undefined);
    const driver=data.drivers.find(d=>d.id===trace.meta.driver),expected=expectedByDriver.get(driver.id);
    const actual=Array.from(trace.y).filter(v=>v!==null);
    assert.equal(actual.length,expected.length);
    actual.forEach((v,i)=>assert(Math.abs(v-expected[i])<1e-10));
    assert.equal(trace.x.filter(v=>v===null).length,driver.ratings.RM.filter((p,i)=>i&&p[3]!==1).length);
    const bridge=api.graph.data.find(t=>t.meta.driver===driver.id&&t.meta.kind==='bridge');
    if(bridge){assert.equal(bridge.mode,'lines');assert.equal(bridge.line.dash,'dash');assert.equal(bridge.x.length,driver.ratings.RM.filter(p=>p[3]===2).length*3);}
  }
  const isolated=api.graph.data.filter(t=>t.meta.kind==='isolated');
  assert(isolated.some(t=>t.meta.driver==='andre-lotterer'&&t.x.length===1));
  for(const trace of isolated){
    const d=data.drivers.find(d=>d.id===trace.meta.driver);
    for(const [slot] of trace.customdata){const i=d.ratings.RM.findIndex(p=>p[0]===slot);assert(!(i&&d.ratings.RM[i][3]>0)&&!(d.ratings.RM[i+1]?.[3]>0));}
  }
  const alonso=data.drivers.find(d=>d.id==='fernando-alonso');
  assert.equal(alonso.ratings.RM.find(p=>data.slots[p[0]].year===2021)[3],0);
  const singleTip=hover('charles-leclerc',slotIndex);
  assert.equal((singleTip.innerHTML.match(/class="hover-row"/g)||[]).length,1,'Large selections retain closest-driver hover');
  await click('favorites');assert.equal(api.state.selected.size,8);
  const currentSlot=data.slots.findIndex(s=>s.kind==='current_after_round');
  context.innerHeight=360;
  const compact=hover('charles-leclerc',currentSlot);
  assert((compact.innerHTML.match(/class="hover-row"/g)||[]).length<=5);
  assert.match(compact.innerHTML,/class="hover-more"/);
  assert.equal(errors.length,0);
  assert.equal(JSON.stringify(api.data.drivers.map(d=>d.ratings)),rawBefore);
}
console.log(JSON.stringify({status:'PASS',scope:'actual_application_logic_not_browser_render',languages:['zh-CN','en'],
  drivers:84,points_per_language:7248,original_RM_data_exact:true,Plotly_bundle_exact:true,normal_mean:expectedMean,normal_sd:expectedSD,uncapped:true,reference_points:7226,all_knots_verified:true,
  fixed_sma_windows:[3,4],old_preferences_cannot_override:true,all_display_surfaces_match:true,shared_event_hover:true,
  spline_curves_without_repeated_markers:true,dashed_season_boundaries_and_absences_preserved:true,isolated_snapshots_visible:true,
  language_switch_preserves_controls_and_zoom:true,causal_smoothing_and_CSV_equal:true},null,2));
