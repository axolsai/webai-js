import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {readWorker} from './worker-manifests.mjs';
const base = 'https://f005.backblazeb2.com/file/hubters-webai-public/assets/benchmarks/';
const publicDir = 'test-app/public';
const files = fs.readdirSync(publicDir).filter(name=>name.startsWith('test-')).flatMap(dir=>fs.readdirSync(`${publicDir}/${dir}`).map(name=>`${publicDir}/${dir}/${name}`));
function local(file) { const result=files.find(p=>path.basename(p)===path.basename(file)); if(!result)throw Error(`Missing local fixture ${file}`);return result; }
function json(file) {return JSON.parse(fs.readFileSync(local(file),'utf8'));}
function keep(file) {const source=local(file), target=`model-samples/${path.basename(file)}`;fs.copyFileSync(source,target);return target;}
function asset(field, fixture, encoding='blob-url', sampleRate) {
 const file=keep(fixture.file), bytes=fs.readFileSync(file), sha1=crypto.createHash('sha1').update(bytes).digest('hex');
 if(bytes.length!==fixture.size||sha1!==fixture.sha1)throw Error(`Fixture mismatch ${file}`);
 const extension=path.extname(file);const mediaType={'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png','.mp3':'audio/mpeg','.wav':'audio/wav','.bin':'application/octet-stream'}[extension];
 if(!mediaType)throw Error(`Unknown asset type ${file}`);
 return {field,url:fixture.url,file,mediaType,size:bytes.length,sha1,encoding,...(sampleRate?{sampleRate}:{}),...(fixture.width?{width:fixture.width,height:fixture.height}:{}),...(fixture.durationSeconds?{durationSeconds:fixture.durationSeconds}:{})};
}
const inventory=[];
for(const filename of fs.readdirSync('model-workers-v4').filter(f=>f.endsWith('.worker.js'))) {
 const file=`model-workers-v4/${filename}`, {source,ast,declaration,manifest:m}=readWorker(file), id=m.model.id, fixture=m.benchmark.fixture;
 const fixtureFile=keep(fixture.file), isJson=fixture.file.endsWith('.json'), data=isJson?json(fixture.file):null;
 let input={}, assets=[], modelConfig={}, generateConfig={};
 if(data) {
  const candidate=data.input??data.cases?.[0]??data;
  if(id.startsWith('bge-reranker')) {input={query:data.input.texts[0],documents:data.input.texts.slice(1)};}
  else if(id.startsWith('sam-')||id==='slimsam-77-uniform') {
   input={input_points:data.input_points,input_labels:data.input_labels};
   assets=[asset('image',jsonFixtureForPortrait())];
  } else {
   const fieldNames=new Set(m.input.alternatives.flatMap(a=>a.fields?a.fields.map(f=>f.name):[a.name]));
   for(const [key,value] of Object.entries(candidate))if(fieldNames.has(key))input[key]=value;
  }
  if(id==='rubert-tiny-toxicity')input={text:data.cases.find(c=>c.expected==='safe').text};
  if(id==='speecht5-tts')assets.push(asset('speakerEmbedding',fixture.speakerEmbedding));
 } else {
  const audio=fixture.file.endsWith('.mp3')||fixture.file.endsWith('.wav');
  const flat=m.input.alternatives.flatMap(a=>a.fields??[a]);
  const field=audio?'audio':flat.some(f=>f.name==='image_blob_url')?'image_blob_url':'image';
  assets.push(asset(field,fixture,audio?'audio-pcm-f32':'blob-url',audio?16000:undefined));
  if(id.startsWith('vitmatte-'))assets.push(asset('trimap',fixture.trimap));
  if(id==='grounding-dino-tiny')input.candidate_labels=['person'];
  if(id==='siglip-base-patch16-224')input.candidate_labels=json(fixture.labels.file).input.candidate_labels;
 }
 // Only copy options the worker actually declares, and preserve their correct namespace.
 const options={...(data?.modelConfig??{}),...(data?.generation??{}),...(data?.generateConfig??{}),...(data?.voice?{voice:data.voice,speed:data.speed}:{})};
 for(const f of m.config.model)if(Object.hasOwn(options,f.name))modelConfig[f.name]=options[f.name];
 for(const f of m.config.generation)if(Object.hasOwn(options,f.name))generateConfig[f.name]=options[f.name];
 if(id.startsWith('mediapipe-'))generateConfig.useImageConversion=true;
 const sample={id:'standard',label:isJson?'Try sample text':fixture.file.match(/\.(wav|mp3)$/)?'Try sample audio':'Try sample image',description:`Standard sample for ${m.model.displayName}.`,userInput:input,assets,modelConfig,generateConfig,source:fixture.source??data?.source??fixture.url,license:fixture.license??data?.license??'Unverified: upstream fixture license not recorded',fixture:{file:fixtureFile,url:fixture.url,size:fixture.size,sha1:fixture.sha1},benchmarkReport:m.benchmark.report};
 if(!Object.keys(input).length&&!assets.length)throw Error(`Empty sample ${id}`);
 const property=declaration.initializer.properties.find(p=>p.name?.getText(ast)==='samples');
 let updated;
 if(property)updated=source.slice(0,property.getStart(ast))+`samples: ${JSON.stringify([sample],null,2)}`+source.slice(property.end);
 else {const end=declaration.initializer.end-1;updated=source.slice(0,end).replace(/,?\s*$/, ",")+`\n  samples: ${JSON.stringify([sample],null,2)},\n`+source.slice(end);}
 fs.writeFileSync(file,updated);
 const report=m.benchmark.report;
 if(report&&fs.existsSync(report)){const saved=JSON.parse(fs.readFileSync(report,'utf8'));saved.samples=[sample];fs.writeFileSync(report,JSON.stringify(saved,null,2)+'\n');}
 inventory.push({id,samples:[sample]});
}
function jsonFixtureForPortrait(){return {file:'modnet-portrait-pexels-5965592.jpg',url:base+'modnet-portrait-pexels-5965592.jpg',size:78124,sha1:'38212739a4a4f0be25d84863cf991d7fb6c6d818'};}
fs.writeFileSync('model-samples/catalog.json',JSON.stringify(inventory,null,2)+'\n');
console.log(`Updated samples for ${inventory.length} workers; inference and support matrices unchanged.`);
