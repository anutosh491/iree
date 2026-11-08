# Emscripten 6.x / LLVM 23.1.2 source branch

This branch starts at IREE `9c7d2ae876b372b4b3304fec07e8f29df35979b1`.
It is the direct parent of LLVM integration `a869dc3c`, which changed IREE's
LLVM dependency from LLVM 23 to LLVM 24. The LLVM submodule here is pinned
to the `llvmorg-23.1.2` release commit. Emscripten-forge builds use the installed
LLVM/MLIR/Clang/LLD 23.1.2 SDK with `IREE_BUILD_BUNDLED_LLVM=OFF`.

`source-lock.json` records merged upstream backports, the final merged VM ownership
fix, remaining local changes and dependency revisions. This is a compatibility
branch; it is not an upstream-supported IREE/LLVM release combination.

## Dependencies

For the external-SDK compiler build, initialize only the required submodules:

```sh
git submodule update --init third_party/flatcc third_party/stablehlo \
  third_party/torch-mlir third_party/webgpu-headers
```

The StableHLO submodule uses our LLVM-23-compatible fork with optional tests
already applied. Configure with `STABLEHLO_ENABLE_TESTS=OFF`; its default
remains ON and compiler helper libraries remain available.

WebGPU code generation also uses the pinned Dawn/Tint, Abseil and SPIR-V
dependencies listed in `source-lock.json` and the WebGPUSPIRV CMake files.
Apply `patches/abseil-emscripten-version-macros.patch` to the pinned Abseil
source before configuring. It backports upstream Abseil commit `dcd3be56`.
Packaging must acquire these sources explicitly and set the corresponding
`FETCHCONTENT_SOURCE_DIR_*` values; it must not rely on a developer's cache.

## Scope and validation

The compiler includes Torch and StableHLO frontends, Wasm CPU code generation and WebGPU code generation.
The native build helpers and target libraries use this same source selection.
Python CLI bundling can be disabled while keeping the library APIs.

Emscripten builds embed the existing JS proactor companion in the library with
`EM_JS`. Loading the library initializes one host context per Emscripten instance;
no Python- or notebook-specific host setup is needed for these imports. The compiler and WebGPU runtime use the JS backend's inline completion mode in the kernel Worker.
The clock uses Emscripten's libc `CLOCK_MONOTONIC` implementation.

The LLVM-CPU path emits WebAssembly side modules and links them with
LLD in process under Emscripten. IREE's existing system-library loader supplies
`dlopen`/`dlsym`; it does not need a replacement loader. Enable
`IREE_TARGET_BACKEND_LLVM_CPU` and `IREE_TARGET_BACKEND_LLVM_CPU_WASM`, and use
the inline-dynamic execution model. The tested runtime heap alignment is 256.
This adds an IREE CPU route alongside the existing JAX/XLA and PyTorch routes;
it does not automatically select IREE for `jax.jit`.

CPU-enabled builds additionally need native `iree-c-embed-data` and LLVM 23.1.2
`clang`/`llvm-link` tools to prepare device bitcode during the build. Their
explicit paths and Clang resource headers can be supplied to the installed-SDK
CMake configuration. These executables are build dependencies.

The compiler package has passed wasm32/wasm64 browser import and in-process
compilation checks. The runtime Python bridge has also been exercised on both
widths with published PyTorch packages, direct Turbine export, FP32 arithmetic,
linear training and convolution gradients. A combined recipe build is still a
separate gate; these experiments reuse the already-built compiler package.

## Asynchronous WebGPU Python API

Enable `IREE_BUILD_RUNTIME_PYTHON_BINDINGS`, disable
`IREE_BUILD_RUNTIME_PYTHON_TOOLS`, set `IREE_CMAKE_PLUGIN_PATHS` to this directory,
and select `IREE_EXTERNAL_HAL_DRIVERS=webgpu`. Emscripten filesystem integration
also needs `IREE_FILE_IO_ENABLE=1` in the C and C++ compiler definitions.

The package embeds IREE's existing WebGPU companion in the Python kernel Worker.
Its inline proactor drives completion callbacks while Python yields to the event
loop. This needs no shared memory, COOP/COEP headers, pthreads, JSPI, Asyncify,
PyJS calls, or notebook JavaScript setup.

```python
import iree.runtime as rt
from iree.runtime.webgpu import get_webgpu_device

device = await get_webgpu_device()
config = rt.Config(device=device)
with rt.AsyncSession(device) as execution:
    inputs = await execution.upload(host_inputs)
    function = execution.bind(exported_function, abi="coarse-fences")
    pending = function.submit(inputs)
    result = await execution.download(pending.value, wait_fences=[pending.fence])
```

`exported_function` is a function loaded through `SystemContext`. Compile with
`--iree-execution-model=async-external`. Torch imports already export a
`main$async` function with the coarse-fences calling convention. An explicit
`abi="coarse-fences"` declaration is needed for exports without runtime ABI
reflection; the adapter does not infer calling conventions from function names.

Submissions expose a value and a completion fence. Pass that fence as a dependency
when chaining operations without a host wait. This keeps parameters, optimizer
state and intermediate arrays on the GPU. Bound the submission window to control
in-flight memory. Await results when the host needs them or before releasing the
execution session. Cancelling a Python await does not cancel submitted GPU work.
Closing a session stops its Python waits.

`AsyncSession` owns its completion bridge and event-loop lifetime. Native builds
retain the existing threaded bridge; single-threaded builds register callbacks on
IREE's existing async semaphore timepoints. Future cancellation is delivered on
the owning Python event loop in both cases.

The validated scope is one cached WebGPU device, repeated resident FP32 execution,
host/file transfers, bounded PyTorch/Turbine training, cleanup, cancellation and
failure propagation. Arithmetic and lifecycle checks cover wasm32 and wasm64;
full MNIST training covers wasm64. Multi-device concurrency and the complete PJRT
contract remain separate work. This is not a PyTorch GPU Tensor backend.

The working compiler configuration uses O3, 256-byte resource alignment and
mutable-binding aliasing; it disables constant evaluation and constant-expression
hoisting. Existing SPIR-V, Tint/WGSL and runtime fixes remain in the source stack.
The old synchronous Worker transport is preserved in source revision
`a682ad82e3c3ea65d8f3ff004906f7be72baed95`, which is the rollback point for the
published header-dependent runtime.

The saved MLIR fixed-array alias and vector-gather changes belong to the LLVM/MLIR
SDK. Numerical LAPACK/FFT/callback adapters and single-device frontend rewrites
remain outside this generic compiler/runtime port.

## Refreshed CPU and WebGPU configuration

`source-lock.json` records the required build options for both routes. Use
`local-sync` and the system-library loader for threadless CPU execution, and the
inline asynchronous WebGPU host for GPU execution. Local-task and embedded ELF
remain disabled. The 256-byte host heap alignment is an explicit package policy.

Use MLIR 23.1.2 build 9 or later for the finite erfc and SPIR-V alias fixes.
The embedded Torch-MLIR dependency also needs the FMA backport from its separate
integration branch before publishing the final recipe revision. The source lock
tracks that pending pin; it must be resolved by the source commit script.

The refreshed full package still needs CI. Saved notebook checks used development
artifacts; they do not establish clean artifact parity on both pointer widths.
