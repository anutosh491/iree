# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Experimental WebGPU initialization for Emscripten Worker hosts."""
import asyncio
from . import _binding
from .system_setup import get_device


async def get_webgpu_device(*, timeout: float = 30):
    """Initialize the browser event Worker and return IREE's WebGPU device."""
    start = getattr(_binding, '_webgpu_initialize', None)
    if start is None:
        raise RuntimeError('This runtime was not built with Emscripten WebGPU support')
    if not _binding._webgpu_can_block():
        raise RuntimeError('WebGPU requires a cross-origin-isolated Web Worker')
    if start():
        raise RuntimeError('WebGPU Worker could not start')
    async def wait_until_ready():
        while True:
            state = _binding._webgpu_is_ready()
            if state < 0:
                raise RuntimeError('WebGPU Worker failed to start')
            if state:
                return
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait_until_ready(), timeout=timeout)
    return get_device('webgpu')
