// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Emscripten Worker transport for the WebGPU companion.
import {ProactorEventHost} from '../../../async/platform/js/proactor_event_host.mjs';
import {ProactorRing} from '../../../async/platform/js/proactor_ring.mjs';

export function createWorkerHost(context, workerSource, {requestTimeoutMs = 30000} = {}) {
  let worker, control, payload, failed;
  let ring;
  const pending = new Set();
  const recordings = new Map();
  const stats = {calls: 0, inputBytes: 0, outputBytes: 0, byName: {}};
  const offset = p => typeof p === 'bigint' ? Number(p) : p >>> 0;
  const pointer = (p, n) => typeof p === 'bigint' ? BigInt(n) : n;
  function canBlock() {
    try {
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 0);
      return true;
    } catch { return false; }
  }
  function start() {
    if (failed) throw failed;
    if (worker) return;
    if (!canBlock()) throw Error('Blocking WebGPU calls require an isolated Worker');
    control = new SharedArrayBuffer(4096);
    payload = new SharedArrayBuffer(65536, {maxByteLength: 2 ** 32});
    const url = URL.createObjectURL(new Blob([workerSource], {type: 'application/javascript'}));
    worker = new Worker(url);
    URL.revokeObjectURL(url);
    const ringBuffer = ProactorRing.createSharedBuffer(256);
    const cancelBuffer = new SharedArrayBuffer(16);
    ring = new ProactorRing(ringBuffer, 256);
    context.attachWorker(ringBuffer, new Int32Array(cancelBuffer),
        message => worker.postMessage(message), token => pending.delete(token));
    worker.postMessage({initialize: true, control, payload, ringBuffer, cancelBuffer,
        timeOrigin: performance.timeOrigin, requestTimeoutMs});
    worker.onerror = event => {failed = Error(event.message || 'WebGPU Worker failed');};
  }
  function request(name, args, spans = [], table = null) {
    start();
    let length = 8;
    for (const span of spans) {
      length = Math.ceil(length / 8) * 8;
      span.remote = length;
      length += span.length;
      if (span.arg !== undefined) args[span.arg] = pointer(span.original, span.remote);
    }
    if (payload.byteLength < length) {
      payload.grow(2 ** Math.ceil(Math.log2(length)));
    }
    const heap = new Uint8Array(context.memory.buffer);
    const bytes = new Uint8Array(payload);
    for (const span of spans) {
      if (span.data) bytes.set(span.data, span.remote);
      else if (span.input) bytes.set(heap.subarray(span.local, span.local + span.length), span.remote);
    }
    if (table) {
      const view = new DataView(payload);
      for (let i = 0; i < table.blocks.length; i++) {
        const at = table.span.remote + i * table.width;
        if (table.width === 8) view.setBigUint64(at, BigInt(table.blocks[i].remote), true);
        else view.setUint32(at, table.blocks[i].remote, true);
      }
    }
    const words = new Int32Array(control);
    const nameBytes = new TextEncoder().encode(name);
    if (nameBytes.length > 64 || args.length > 16) throw Error('Invalid WebGPU command');
    words[4] = args.length; words[5] = 0; words[6] = nameBytes.length;
    words[7] = length;
    const outputs = spans.filter(span => span.output || span.returnData);
    if (outputs.length > 1) throw Error('Unexpected WebGPU output layout');
    words[10] = outputs[0]?.remote || 0;
    words[11] = outputs[0]?.length || 0;
    new Uint8Array(control, 64, nameBytes.length).set(nameBytes);
    const values = new BigUint64Array(control, 128, 16);
    for (let i = 0; i < args.length; i++) {
      values[i] = BigInt(args[i]);
      if (typeof args[i] === 'bigint') words[5] |= 1 << i;
    }
    Atomics.store(words, 0, 1);
    Atomics.notify(words, 0);
    if (Atomics.wait(words, 0, 1, requestTimeoutMs) === 'timed-out') {
      failed = Error('WebGPU Worker did not complete within its request deadline');
      worker.terminate();
      throw failed;
    }
    if (words[2]) {
      throw Error(new TextDecoder().decode(new Uint8Array(control, 512, words[3]).slice()));
    }
    for (const span of spans) {
      if (span.output) heap.set(bytes.subarray(span.remote, span.remote + span.length), span.local);
    }
    stats.calls++;
    return new BigInt64Array(control, 32, 1)[0];
  }
  function call(name, values) {
    stats.byName[name] = (stats.byName[name] || 0) + 1;
    if (name === 'can_block') return canBlock() ? 1 : 0;
    if (name === 'get_preconfigured_device') return 0;
    const args = [...values], spans = [];
    let table;
    function add(index, length, output = false) {
      length = Number(length);
      const original = args[index], local = offset(original);
      if (!Number.isSafeInteger(length) || length < 0 ||
          !Number.isSafeInteger(local) || local + length > context.memory.buffer.byteLength) {
        throw RangeError('WebGPU import range exceeds Wasm memory');
      }
      const span = {arg: index, original, local, length, input: !output, output};
      spans.push(span);
      return span;
    }
    switch (name) {
      case 'request_adapter': case 'adapter_request_device': add(1, 4, true); break;
      case 'buffer_get_mapped_range': add(3, args[2], true); stats.outputBytes += Number(args[2]); break;
      case 'buffer_set_mapped_range': add(3, args[2]); stats.inputBytes += Number(args[2]); break;
      case 'pass_set_bind_group': add(3, args[4] * 4); break;
      case 'device_create_compute_pipeline': add(2, args[3]); add(4, args[5]); break;
      case 'device_create_pipeline_layout': add(1, args[2] * 4); break;
      case 'device_create_bind_group_layout': add(1, args[2] * 24); break;
      case 'device_create_bind_group': add(2, args[3] * 24); break;
      case 'execute_instructions': case 'create_recording': {
        const base = name === 'execute_instructions' ? 2 : 1;
        const original = args[base], count = args[base + 1];
        const width = typeof original === 'bigint' ? 8 : 4;
        const view = new DataView(context.memory.buffer);
        const span = add(base, count * width), blocks = [];
        for (let i = 0; i < count; i++) {
          const at = offset(original) + i * width;
          const ptr = width === 8 ? view.getBigUint64(at, true) : view.getUint32(at, true);
          const words = i + 1 === count ? args[base + 3] : args[base + 2];
          const block = {original: ptr, local: offset(ptr), length: words * 4, input: true};
          if (block.local + block.length > context.memory.buffer.byteLength) throw RangeError('Invalid command block');
          spans.push(block); blocks.push(block);
        }
        table = {span, blocks, width};
        add(base + 4, args[base + 5] * 8);
        add(8, 16);
        break;
      }
      case 'execute_recording': add(2, (recordings.get(args[0]) || 0) * 8); break;
      case 'queue_write_buffer': add(3, args[4]); stats.inputBytes += Number(args[4]); break;
      case 'adapter_get_info': add(1, 16, true); break;
      case 'device_get_limits': add(1, 32, true); break;
      case 'file_get_length': return BigInt(FS.stat(FS.getStreamChecked(args[0]).path).size);
      case 'queue_write_buffer_from_file': {
        const bytes = new Uint8Array(Number(args[5]));
        const count = FS.read(FS.getStreamChecked(args[3]), bytes, 0, bytes.length, Number(args[4]));
        if (count !== bytes.length) throw Error('Incomplete GPU file upload');
        const result = request('queue_write_buffer', [args[0], args[1], args[2], 8, BigInt(bytes.length)],
          [{arg: 3, original: 8, length: bytes.length, data: bytes}]);
        stats.inputBytes += bytes.length;
        return Number(result);
      }
      case 'file_write_from_mapped': case 'buffer_export_mapped_to_file': {
        const span = {arg: 3, original: 8, local: 0, length: Number(args[2]), returnData: true};
        request('buffer_get_mapped_range', [args[0], args[1], args[2], 8], [span]);
        let stream;
        if (name === 'file_write_from_mapped') stream = FS.getStreamChecked(args[3]);
        else {
          const path = '/tmp/iree-webgpu-' + crypto.randomUUID();
          stream = FS.open(path, 'w+'); FS.unlink(path);
        }
        const bytes = new Uint8Array(payload, span.remote, span.length);
        const count = FS.write(stream, bytes, 0, bytes.length, name === 'file_write_from_mapped' ? Number(args[4]) : 0);
        if (count !== bytes.length) throw Error('Incomplete GPU file download');
        return name === 'file_write_from_mapped' ? undefined : stream.fd;
      }
    }
    const result = request(name, args, spans, table);
    if (name === 'create_recording' && result) recordings.set(Number(result), values[7]);
    if (name === 'handle_release') recordings.delete(values[0]);
    if (['request_adapter', 'adapter_request_device'].includes(name)) {
      context.complete(values.at(-1), Number(result));
    } else if (['buffer_map_async', 'queue_on_submitted_work_done'].includes(name)) {
      pending.add(values.at(-1));
    }
    return Number(result);
  }
  function failure(name, values, error) {
    failed ||= error;
    stats.lastError = String(error);
    if (['request_adapter', 'adapter_request_device', 'buffer_map_async', 'queue_on_submitted_work_done'].includes(name)) {
      context.complete(values.at(-1), 13);  // IREE_STATUS_INTERNAL.
    }
    if (['execute_instructions', 'execute_recording'].includes(name)) return 13;
    if (name === 'buffer_export_mapped_to_file') return -1;
    return 0;
  }
  return {call, stats, start, failure,
    ready() {return failed ? -1 : (control && Atomics.load(new Int32Array(control), 1) === 1 ? 1 : 0);},
    close() {
      failed ||= Error('WebGPU Worker was closed');
      worker?.terminate();
      const memory = {buffer: new ArrayBuffer(256 * 8)};
      const count = ring?.drain(memory, 0, 256) || 0;
      const entries = new Int32Array(memory.buffer);
      for (let i = 0; i < count; ++i) {
        pending.delete(entries[i * 2]);
        context.complete(entries[i * 2], entries[i * 2 + 1]);
      }
      for (const token of pending) context.complete(token, 13);
      pending.clear();
    }};
}

export function startGPUWorker(createImports) {
  const waiters = new Map();
  const pending = new Map();
  let eventHost;
  let deviceFailure;
  const context = {
    memory: {buffer: null},
    gpu: {
      async requestAdapter(options) {
        const adapter = await navigator.gpu.requestAdapter(options);
        if (!adapter) return null;
        return {info: adapter.info, async requestDevice(options) {
          const device = await adapter.requestDevice(options);
          device.lost.then(info => {
            deviceFailure = Error('WebGPU device lost: ' + info.message);
            for (const token of pending.keys()) context.complete(token, 14);
          });
          device.addEventListener('uncapturederror', event => {
            console.error('UNCAUGHT_GPU_ERROR', event.error.message);
          });
          return device;
        }};
      },
    },
    inline: true,
    complete(token, status) {
      const waiter = waiters.get(token);
      if (waiter) waiter(status);
      else if (pending.has(token)) {
        clearTimeout(pending.get(token));
        pending.delete(token);
        eventHost.ring.write(token, status);
      }
    },
  };
  const imports = createImports(context);
  onmessage = async ({data}) => {
    const words = new Int32Array(data.control);
    context.memory.buffer = new ArrayBuffer(65536);
    eventHost = new ProactorEventHost(data.ringBuffer, data.cancelBuffer, 256,
        () => BigInt(Math.round((performance.timeOrigin - data.timeOrigin + performance.now()) * 1e6)));
    eventHost.start();
    onmessage = event => eventHost.handleMessage(event.data);
    Atomics.store(words, 1, 1);
    while (true) {
      const state = Atomics.load(words, 0);
      if (state !== 1) {
        const waiting = Atomics.waitAsync(words, 0, state);
        if (waiting.async) await waiting.value;
        continue;
      }
      const name = new TextDecoder().decode(new Uint8Array(data.control, 64, words[6]).slice());
      const values = new BigUint64Array(data.control, 128, words[4]);
      const args = Array.from(values, (v, i) => words[5] & (1 << i) ? v : Number(v));
      const length = words[7] >>> 0;
      if (context.memory.buffer.byteLength < length) context.memory.buffer = new ArrayBuffer(length);
      new Uint8Array(context.memory.buffer, 0, length).set(new Uint8Array(data.payload, 0, length));
    try {
      const deferred = ['buffer_map_async', 'queue_on_submitted_work_done'].includes(name);
      if (deferred) {
        const token = args.at(-1);
        pending.set(token, setTimeout(() => context.complete(token, 13), data.requestTimeoutMs));
      }
      if (deviceFailure && !['device_destroy', 'buffer_destroy', 'handle_release'].includes(name)) throw deviceFailure;
      const asynchronous = ['request_adapter', 'adapter_request_device'].includes(name);
      let result;
      if (asynchronous) {
        result = await new Promise(resolve => {
          waiters.set(args.at(-1), resolve);
          imports[name](...args);
        });
        waiters.delete(args.at(-1));
      } else result = imports[name](...args);
      new BigInt64Array(data.control, 32, 1)[0] = BigInt(result || 0);
      const outOffset = words[10] >>> 0, outLength = words[11] >>> 0;
      if (outLength) new Uint8Array(data.payload, outOffset, outLength).set(new Uint8Array(context.memory.buffer, outOffset, outLength));
      words[2] = 0;
    } catch (error) {
      if (['buffer_map_async', 'queue_on_submitted_work_done'].includes(name)) {
        context.complete(args.at(-1), 13);
        words[2] = 0;
      } else {
        const bytes = new TextEncoder().encode(error.stack || String(error)).subarray(0, words.byteLength - 512);
        new Uint8Array(data.control, 512, bytes.length).set(bytes);
        words[2] = 1; words[3] = bytes.length;
      }
    } finally {
      Atomics.store(words, 0, 2);
      Atomics.notify(words, 0);
    }
    }
  };
}
