import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
export function readWorker(file, source = fs.readFileSync(file, 'utf8')) {
  const ast = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true);
  const statements = [];
  for (const statement of ast.statements) {
    if (!ts.isImportDeclaration(statement)) statements.push(statement.getText(ast));
    if (ts.isVariableStatement(statement)) {
      const declaration = statement.declarationList.declarations.find(d => d.name.getText(ast) === 'MANIFEST');
      if (declaration) return { source, ast, declaration, manifest: vm.runInNewContext(`${statements.join('\n')}; JSON.parse(JSON.stringify(MANIFEST))`, {}, { timeout: 2000 }) };
    }
  }
  throw Error(`Missing MANIFEST in ${file}`);
}
