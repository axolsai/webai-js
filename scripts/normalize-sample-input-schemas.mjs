import fs from 'node:fs';
import {readWorker} from './worker-manifests.mjs';
for(const name of fs.readdirSync('model-workers-v4').filter(f=>f.endsWith('.worker.js'))) {
 const file=`model-workers-v4/${name}`, {source,ast,declaration,manifest:m}=readWorker(file);
 if(m.input.alternatives.every(a=>a.fields))continue;
 const alternatives=m.input.alternatives;
 let groups;
 if(m.model.id==='siglip-base-patch16-224')groups=[{name:'image and candidate labels',description:m.input.description,fields:alternatives.map(f=>({...f,...(f.name==='candidate_labels'?{minItems:2,maxItems:100}:{format:'url'})}))}];
 else if(m.model.id==='slimsam-77-uniform')groups=[{name:'image and points',description:m.input.description,fields:[{name:'image',type:'string',required:true,minLength:1,format:'url'},{name:'input_points',type:'number[][][]',required:true,minItems:1,description:'One prompt group with one or more finite [x,y] image-space points.'},{name:'input_labels',type:'number[][]',required:false,description:'When supplied, matches input_points and contains only 0 or 1.'}]}];
 else groups=alternatives.map(f=>({name:f.name,description:m.input.description,fields:[{...f,...(f.maximumLength?{maxLength:f.maximumLength}:{})}]}));
 for(const g of groups)for(const field of g.fields)delete field.maximumLength;
 const input={...m.input,alternatives:groups};
 const property=declaration.initializer.properties.find(p=>p.name?.getText(ast)==='input');
 fs.writeFileSync(file,source.slice(0,property.getStart(ast))+`input: ${JSON.stringify(input,null,2)}`+source.slice(property.end));
 const report=JSON.parse(fs.readFileSync(m.benchmark.report,'utf8'));if(report.input){report.input=input;fs.writeFileSync(m.benchmark.report,JSON.stringify(report,null,2)+'\n');}
}
