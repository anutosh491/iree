// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_WEBGPU_EMSCRIPTEN_H_
#define IREE_HAL_DRIVERS_WEBGPU_EMSCRIPTEN_H_

#ifdef __cplusplus
extern "C" {
#endif

// Starts the event Worker. The caller must yield until initialization completes.
int iree_hal_webgpu_emscripten_initialize(void);
int iree_hal_webgpu_emscripten_is_ready(void);

#ifdef __cplusplus
}
#endif

#endif
