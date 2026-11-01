# Emscripten 6.x / LLVM 23.1.2 source branch

This branch starts at IREE `9c7d2ae876b372b4b3304fec07e8f29df35979b1`.
It is the direct parent of LLVM integration `a869dc3c`, which changed IREE's
LLVM dependency from LLVM 23 to LLVM 24. The LLVM submodule here is pinned
to the `llvmorg-23.1.2` release commit. Emscripten-forge builds use the installed
LLVM/MLIR/Clang/LLD 23.1.2 SDK with `IREE_BUILD_BUNDLED_LLVM=OFF`.

`source-lock.json` records merged upstream backports, the borrowed VM ownership
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

The compiler includes Torch and StableHLO frontends and WebGPU code generation.
The native build helpers and target libraries use this same source selection.
Python CLI bundling can be disabled while keeping the library APIs.

Emscripten builds embed the existing JS proactor companion in the library with
`EM_JS`. Loading the library initializes one host context per Emscripten instance;
no Python- or notebook-specific host setup is needed for these imports. The compiler starts with the JS backend's inline mode. The optional WebGPU
runtime connects its event Worker to the existing proactor completion ring,
timer host and cancellation protocol when initialized.
The clock uses Emscripten's libc `CLOCK_MONOTONIC` implementation.

The optional LLVM-CPU path emits WebAssembly side modules and links them with
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

## Experimental runtime Python bridge

Enable `IREE_BUILD_RUNTIME_PYTHON_BINDINGS`, disable
`IREE_BUILD_RUNTIME_PYTHON_TOOLS`, set `IREE_CMAKE_PLUGIN_PATHS` to this directory,
and select `IREE_EXTERNAL_HAL_DRIVERS=webgpu`. Emscripten filesystem integration
also needs `IREE_FILE_IO_ENABLE=1` in the C and C++ compiler definitions.

The package embeds the existing WebGPU companion and owns its event Worker.
Initialize it asynchronously once, then use the usual runtime API:

```python
import iree.runtime as rt
from iree.runtime.webgpu import get_webgpu_device

device = await get_webgpu_device()
config = rt.Config(device=device)
```

This requires a cross-origin-isolated browser Worker. The kernel's Wasm memory
remains unshared and pthreads are not required. The Worker transports commands
and transfers; IREE owns GPU buffers and compiled execution. No PyJS or Pyodide
API is required by this integration.

The tested scope is a single device with static FP32 computations, host transfers,
repeated resident updates and cleanup. Finite waits, joined fences, failed transfer dependencies, Worker startup and
shutdown failures, and device loss have focused browser checks. Multi-device
concurrency and the complete PJRT contract remain separate work. This is not a general PyTorch GPU Tensor backend. The working compiler
configuration uses O3, 256-byte resource alignment and mutable-binding aliasing;
it disables constant evaluation and constant-expression hoisting. The resource
configuration printer fix allows these resource settings to survive MLIR
round-tripping, but the updated Wasm compiler must still be built and checked.

The saved MLIR fixed-array alias and vector-gather changes belong to the LLVM/MLIR
SDK; they are not implemented by replacing archives in this branch. Numerical
LAPACK/FFT/callback adapters and single-device frontend rewrites remain preserved
in their existing workspaces, outside the generic compiler/runtime port.
