import fs from "node:fs";
import vm from "node:vm";

let source = fs.readFileSync(process.argv[2], "utf8");
source = source.replace(/^import[\s\S]*?;\s*$/gm, "");

const match = /const\s+MANIFEST\s*=/.exec(source);
if (!match) throw new Error("Worker does not declare MANIFEST");

const start = match.index;
let cursor = start + match[0].length;
let depth = 0;
let quote = null;
let escaped = false;

for (; cursor < source.length; cursor += 1) {
  const character = source[cursor];
  if (quote) {
    if (escaped) escaped = false;
    else if (character === "\\") escaped = true;
    else if (character === quote) quote = null;
    continue;
  }
  if (character === '"' || character === "'" || character === "`") quote = character;
  else if ("({[".includes(character)) depth += 1;
  else if (")}]".includes(character)) depth -= 1;
  else if (character === ";" && depth === 0) break;
}

const executable = `${source.slice(0, start)}globalThis.__manifest=${source.slice(start + match[0].length, cursor + 1)}`;
const context = {};
vm.createContext(context);
vm.runInContext(executable, context, { timeout: 1000 });
process.stdout.write(JSON.stringify(context.__manifest));
