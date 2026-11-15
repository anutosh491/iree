# Emscripten integration sources

This branch keeps the LLVM/MLIR/Clang/LLD 23.1.2 library baseline with
Emscripten 6.0.8, Python 3.14.3, Wasm exceptions and no pthreads.
Use the installed SDK (`IREE_BUILD_BUNDLED_LLVM=OFF`); MLIR build 10
contains the selected scalar/vector alias and integer-bitcast backports.

`source-lock.json` records the compatible dependencies and frontend pin.
Native helper tools and Wasm compiler/runtime packages must use the same
IREE revision. The recipe must use the embedded Torch-MLIR revision
recorded by both the gitlink and source lock. The standalone Torch-MLIR
package remains independent of this embedded compiler frontend.

## Browser WebGPU

Python/IREE and the JavaScript WebGPU companion run in the kernel Worker.
Queued operations carry explicit dependencies; Python yields at async
completion and readback boundaries. This path needs neither a second GPU
Worker nor SharedArrayBuffer, COOP/COEP, JSPI or Asyncify.

The package owns browser-host setup and event-loop progress. The async
Python API and Turbine's optional tensor/device adapter are separate
consumer interfaces. Generic fixes remain separate from this integration.

## Wasm CPU

The LLVM CPU backend targets WebAssembly, links in process with LLD and
uses the local-sync runtime. Keep host heap alignment at its default
64 bytes; WebGPU resource alignment is a separate device requirement.
Target-aware half-helper specialization preserves native retargeting.

## Validation

The selected implementations have focused compiler/runtime checks and
wasm64 notebook evidence. Replaying this source stack does not replace
fresh package CI and browser checks against the resulting artifacts.
Complete wasm32 model coverage remains a separate verification step.
