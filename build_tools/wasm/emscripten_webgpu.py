# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Embed the WebGPU companion for Emscripten's asynchronous inline host."""
import argparse
import json
from pathlib import Path
import re

from wasm_binary_bundler import generate_merger, transform_companion

parser = argparse.ArgumentParser(description=__doc__)
for name in ('header', 'companion', 'host', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
declarations = re.findall(
    r'IREE_HAL_WEBGPU_IMPORT\("([^"]+)"\) extern (\w+)\s+'
    r'(iree_hal_webgpu_import_\w+)\(([^;]+)\);', args.header.read_text())
if len(declarations) != 38:
    parser.error('unexpected WebGPU import declarations')
companion = transform_companion(args.companion.read_text(), 'iree_webgpu', str(args.companion))
merger = generate_merger(['iree_webgpu']).replace('export function ', 'function ')
host = args.host.read_text().replace('export function ', 'function ')
code = ['#include <stdint.h>\n#include <emscripten.h>\n',
        'EM_JS(void, iree_webgpu_initialize_host_js, (void), {\n',
        "if (Module['ireeWebGPUHost']) return;\n", companion, merger, host,
        "Module['ireeWebGPUHost'] = createInlineHost({memory: wasmMemory,\n",
        "complete: (token, status) => Module['ireeAsyncHost'].context.complete(token, status)},\n",
        "context => createWasmImports(context).iree_webgpu);\n});\n",
        '__attribute__((constructor)) static void iree_webgpu_initialize_host_ctor(void) {iree_webgpu_initialize_host_js();}\n',
        "EM_JS(int, iree_webgpu_start_host_js, (void), {try {Module['ireeWebGPUHost'].start(); return 0;} catch {return 1;}});\n",
        "EM_JS(int, iree_webgpu_host_is_ready_js, (void), {return Module['ireeWebGPUHost'].ready();});\n",
        'int iree_hal_webgpu_emscripten_initialize(void) {return iree_webgpu_start_host_js();}\n',
        'int iree_hal_webgpu_emscripten_is_ready(void) {return iree_webgpu_host_is_ready_js();}\n']
for name, return_type, function, arguments in declarations:
    arguments = ' '.join(arguments.split())
    names = [] if arguments == 'void' else [a.split()[-1] for a in arguments.split(',')]
    values = '[' + ', '.join(names) + ']'
    body = "Module['ireeWebGPUHost'].call(" + json.dumps(name) + ', ' + values + ');'
    if return_type != 'void':
        body = 'return ' + body
    body = ('try { ' + body + ' } catch (error) { return '
            "Module['ireeWebGPUHost'].failure(" + json.dumps(name) + ', ' + values + ', error); }')
    code.append(f'EM_JS({return_type}, {function}, ({arguments}), {{ {body} }});\n')
args.output.write_text(''.join(code))
