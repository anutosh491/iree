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

The branch has configuration and focused source checks against published
wasm32/wasm64 SDKs. Clean package builds, Python dynamic loading and the complete
browser training notebook remain separate validation gates. Existing generated
headers and cross-generation tooling were used by the source checks.

Browser host imports, Worker coordination and resident-array bindings still
need their own installed integration. This branch does not provide a complete
JAX WebGPU PJRT plugin or a general PyTorch GPU Tensor backend. The working
consumer disables constant evaluation and constant-expression hoisting; WebGPU
code generation still uses O3.

The saved MLIR fixed-array alias and vector-gather changes belong to the LLVM/MLIR
SDK; they are not implemented by replacing archives in this branch. Numerical
LAPACK/FFT/callback adapters and single-device frontend rewrites remain preserved
in their existing workspaces, outside the generic compiler/runtime port.
