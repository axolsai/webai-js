import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readWorker} from './worker-manifests.mjs';
const remote=process.argv.includes('--remote'), assets=new Map(), results=[];
const hash=bytes=>crypto.createHash('sha1').update(bytes).digest('hex');
for(const name of fs.readdirSync('model-workers-v4').filter(f=>f.endsWith('.worker.js'))) {
 const {manifest:m}=readWorker(`model-workers-v4/${name}`);
 const original=readWorker(name,execFileSync('git',['show',`HEAD:model-workers-v4/${name}`],{encoding:'utf8'})).manifest;
 const reused = ['runtime','model'].every(key=>JSON.stringify(original[key])===JSON.stringify(m[key])) && JSON.stringify(original.benchmark.results)===JSON.stringify(m.benchmark.results) && original.benchmark.fixture.sha1===m.benchmark.fixture.sha1 && JSON.stringify(original.benchmark.invalidatedBy)===JSON.stringify(m.benchmark.invalidatedBy);
 const errors=[]; const check=(value,message)=>{if(!value)errors.push(message);};
 check(m.samples?.length>0,'missing samples');
 check(reused,'benchmark invalidation: rerun affected cases');
 const provenanceGaps=[];
 for(const sample of m.samples??[]) {
  if(sample.license.startsWith('Unverified:'))provenanceGaps.push('Shared benchmark audio upstream source/license is not recorded.');
  for(const field of ['id','label','description','source','license','benchmarkReport'])check(typeof sample[field]==='string'&&sample[field].length>0,`missing ${field}`);
  const input={...sample.userInput};
  for(const a of sample.assets) {
   check(/^https:\/\//.test(a.url),`not remote HTTPS: ${a.field}`);
   check(['blob-url','audio-pcm-f32'].includes(a.encoding),`encoding ${a.field}`);
   const bytes=fs.readFileSync(a.file);check(bytes.length===a.size&&hash(bytes)===a.sha1,`local asset mismatch ${a.file}`);
   check(!assets.has(a.url)||assets.get(a.url).sha1===a.sha1,`conflicting remote checksum ${a.url}`);
   assets.set(a.url,a);input[a.field]=a.encoding==='audio-pcm-f32'?'prepared audio':'blob:prepared';
  }
  const fixture=fs.readFileSync(sample.fixture.file);check(fixture.length===sample.fixture.size&&hash(fixture)===sample.fixture.sha1,'local fixture mismatch');
  assets.set(sample.fixture.url,{...sample.fixture,field:'fixture'});
  // Flat legacy alternatives describe fields rather than alternative objects.
  const schemas=m.input.alternatives;
  const groups=schemas.some(s=>s.fields)?schemas.map(s=>s.fields??[s]):[schemas];
  const satisfied=groups.some(fields=>fields.every(f=>!f.required||Object.hasOwn(input,f.name)));
  check(satisfied,'sample does not supply a complete input alternative');
  for(const [key,fields] of [['modelConfig',m.config.model],['generateConfig',m.config.generation]])for(const [name,value] of Object.entries(sample[key])) {
   const field=fields.find(f=>f.name===name);check(Boolean(field),`unknown ${key}.${name}`);
   if(field?.enum)check(field.enum.includes(value),`invalid enum ${key}.${name}`);
   if(typeof value==='number'){if(field?.minimum!==undefined)check(value>=field.minimum,`below minimum ${name}`);if(field?.maximum!==undefined)check(value<=field.maximum,`above maximum ${name}`);}
  }
  const saved=JSON.parse(fs.readFileSync(sample.benchmarkReport,'utf8'));check(JSON.stringify(saved.samples)===JSON.stringify(m.samples),'report samples drift');
 }
 for(const [precision,p] of Object.entries(m.runtime.precisions))for(const device of p.supportedDevices)check(m.benchmark.results.some(r=>r.precision===precision&&r.device===device&&r.status==='pass'),`unsupported pair ${precision}/${device}`);
 results.push({id:m.model.id,samples:m.samples.length,benchmarkEvidenceReused:reused,provenanceGaps,passingConfigurations:m.benchmark.results.filter(r=>r.status==='pass').map(r=>`${r.precision}/${r.device}`),errors});
}
const remoteResults=[];
if(remote) {
 const entries=[...assets.entries()];let index=0;
 await Promise.all(Array.from({length:6},async()=>{while(index<entries.length){const [url,asset]=entries[index++];try{const response=await fetch(url,{signal:AbortSignal.timeout(45000)});assert(response.ok,`HTTP ${response.status}`);const bytes=Buffer.from(await response.arrayBuffer());assert.equal(bytes.length,asset.size);assert.equal(hash(bytes),asset.sha1);remoteResults.push({url,passed:true,size:bytes.length,sha1:asset.sha1});}catch(error){remoteResults.push({url,passed:false,error:error.message});}}}));
}
const report={checkedAt:new Date().toISOString(),scope:'Sample metadata, packaged fixtures, remote integrity and unchanged advertised support evidence; not a fresh inference matrix.',workers:results.length,assets:assets.size,remote,results,remoteResults,fullFixtureProvenancePassed:results.every(r=>!r.provenanceGaps.length),passed:results.every(r=>r.errors.length===0)&&remoteResults.every(r=>r.passed)};
fs.writeFileSync('benchmarks/model-samples-contract-audit.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({workers:report.workers,assets:report.assets,passed:report.passed,errors:results.filter(r=>r.errors.length),remoteFailures:remoteResults.filter(r=>!r.passed)},null,2));
if(!report.passed)process.exitCode=1;
