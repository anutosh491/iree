# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(EMSCRIPTEN)
  iree_register_external_hal_driver(
    NAME webgpu
    SOURCE_DIR "${IREE_SOURCE_DIR}/runtime/src/iree/hal/drivers/webgpu"
    BINARY_DIR "${IREE_BINARY_DIR}/runtime/webgpu"
    DRIVER_TARGET iree::hal::drivers::webgpu::registration
    REGISTER_FN iree_hal_webgpu_driver_module_register
  )
endif()
