// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapt the existing WebGPU companion to Emscripten memory and file descriptors.
export function createInlineHost(context, createImports) {
  let ready = false, failure, started = false, device;
  context.inline = true;
  context.gpu = navigator.gpu;
  const fileSpans = new Map();
  context.fileObjects = {
    get: fd => fileSpans.get(fd),
    register(buffer) {
      const path = '/tmp/iree-webgpu-' + crypto.randomUUID();
      const stream = FS.open(path, 'w+');
      FS.unlink(path);
      const bytes = new Uint8Array(buffer);
      if (FS.write(stream, bytes, 0, bytes.length, 0) !== bytes.length) {
        FS.close(stream);
        throw Error('Incomplete GPU file export');
      }
      return stream.fd;
    },
  };
  const imports = createImports(context);
  return {
    start() {
      if (started) return;
      started = true;
      (async () => {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw Error('No WebGPU adapter');
        device = await adapter.requestDevice();
        device.lost.then(info => {failure = Error('WebGPU device lost: ' + info.message);});
        device.addEventListener('uncapturederror', event => {
          failure = event.error;
          console.error('UNCAUGHT_GPU_ERROR', event.error.message);
        });
        context.preConfiguredDevice = context.insertWebGPUHandle(device);
        ready = true;
      })().catch(error => {failure = error;});
    },
    ready() { return failure ? -1 : ready ? 1 : 0; },
    close() { device?.destroy(); },
    call(name, args) {
      if (failure && !['device_destroy', 'buffer_destroy', 'handle_release'].includes(name)) throw failure;
      if (name === 'can_block') return 0;
      if (name === 'file_get_length') {
        const stream = FS.getStreamChecked(args[0]);
        const position = stream.position;
        const length = FS.llseek(stream, 0, 2);
        FS.llseek(stream, position, 0);
        return BigInt(length);
      }
      if (name === 'queue_write_buffer_from_file') {
        const data = new Uint8Array(Number(args[5]));
        if (FS.read(FS.getStreamChecked(args[3]), data, 0, data.length, Number(args[4])) !== data.length)
          throw Error('Incomplete file upload');
        // Adapt Emscripten descriptors to the existing companion's file objects.
        fileSpans.set(args[3], data.buffer);
        try {
          return imports[name](...args.slice(0, 4), 0, args[5]);
        } finally {fileSpans.delete(args[3]);}
      }
      if (name === 'file_write_from_mapped') {
        const data = new Uint8Array(Number(args[2]));
        fileSpans.set(args[3], data.buffer);
        try {
          imports[name](...args.slice(0, 4), 0);
          if (FS.write(FS.getStreamChecked(args[3]), data, 0, data.length, Number(args[4])) !== data.length)
            throw Error('Incomplete file download');
          return;
        } finally {fileSpans.delete(args[3]);}
      }
      return imports[name](...args);
    },
    failure(name, args, error) {
      failure = error;
      if (['request_adapter', 'adapter_request_device', 'buffer_map_async', 'queue_on_submitted_work_done'].includes(name)) {
        context.complete(args.at(-1), 13);
      }
      if (['execute_instructions', 'execute_recording'].includes(name)) return 13;
      if (name === 'buffer_export_mapped_to_file') return -1;
      if (name === 'file_get_length') return 0n;
      return 0;
    },
  };
}
