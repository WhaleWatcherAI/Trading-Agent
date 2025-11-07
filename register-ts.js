const fs = require('fs');
const path = require('path');
const ts = require('typescript');

const defaultCompilerOptions = {
  module: ts.ModuleKind.CommonJS,
  moduleResolution: ts.ModuleResolutionKind.NodeNext,
  target: ts.ScriptTarget.ES2020,
  esModuleInterop: true,
  allowJs: false,
  jsx: ts.JsxEmit.ReactJSX,
  resolveJsonModule: true,
  sourceMap: false,
};

require.extensions['.ts'] = function registerTs(module, filename) {
  const source = fs.readFileSync(filename, 'utf8');
  const { outputText, diagnostics } = ts.transpileModule(source, {
    compilerOptions: defaultCompilerOptions,
    fileName: filename,
  });

  if (diagnostics && diagnostics.length > 0) {
    const formatHost = {
      getCanonicalFileName: file => file,
      getCurrentDirectory: () => process.cwd(),
      getNewLine: () => '\n',
    };
    const message = ts.formatDiagnosticsWithColorAndContext(diagnostics, formatHost);
    console.warn(message);
  }

  return module._compile(outputText, filename);
};

require.extensions['.tsx'] = require.extensions['.ts'];
