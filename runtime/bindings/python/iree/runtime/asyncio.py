# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Asyncio integration for coarse-fence functions and queued transfers."""

import asyncio

import numpy as np

from ._binding import (
    BufferUsage,
    FileHandle,
    HalBufferView,
    HalDeviceLoopBridge,
    HalFence,
    InvokeContext,
    MemoryType,
    VmVariantList,
)
from .array_interop import DeviceArray
from .dtypes import map_dtype_to_hal_element_type
from .function import Invocation, _extract_vm_sequence_to_python

__all__ = ["AsyncSession", "AsyncResult"]


class AsyncResult:
    """A submitted value and its completion fence.

    ``value`` may be passed to another submission ordered by ``fence``. Await
    the result before the host accesses it. Cancelling an await does not cancel
    GPU work or release its input owners before completion.
    """

    def __init__(self, session, value, semaphore, owners=()):
        self.value = value
        self.fence = HalFence.create_at(semaphore, 1)
        self._session = session
        self._semaphore = semaphore
        self._owners = owners
        self._future = None

    async def result(self):
        self._session._check_loop()
        if self._future is None:
            self._future = self._session._bridge.on_semaphore(
                self._semaphore, 1, (self.value, self._owners)
            )
        value, _ = await asyncio.shield(self._future)
        return value

    def __await__(self):
        return self.result().__await__()


class AsyncSession:
    """Owns an asyncio completion bridge for one IREE device and event loop.

    Use this inside a running event loop. ``close()`` stops Python waits; it
    does not cancel submitted device work. Await work that must finish before
    leaving the context. Submissions retain device resources through IREE's
    normal queue ownership rules.
    """

    def __init__(self, device):
        self.device = device
        self._loop = asyncio.get_running_loop()
        self._bridge = HalDeviceLoopBridge(device, self._loop)
        self._closed = False

    def _check_loop(self):
        if self._closed:
            raise RuntimeError("AsyncSession is closed")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("AsyncSession belongs to another event loop")

    def close(self):
        if not self._closed:
            self._closed = True
            self._bridge.stop()

    def __enter__(self):
        self._check_loop()
        return self

    def __exit__(self, *exc):
        self.close()

    def bind(self, function, *, abi=None):
        """Bind a function using the coarse-fences calling convention.

        An explicit ``abi='coarse-fences'`` declaration is needed if the
        compiler's export does not include calling-convention reflection.
        """
        self._check_loop()
        model = function.vm_function.reflection.get("iree.abi.model")
        if abi is not None and model is not None and abi != model:
            raise ValueError("Explicit ABI disagrees with function reflection")
        if (abi or model) != "coarse-fences":
            raise ValueError("Async invocation requires the coarse-fences ABI")
        if function._device != self.device:
            raise ValueError("Function belongs to another IREE device")
        return _AsyncFunction(self, function)

    def enqueue_upload(self, value):
        self._check_loop()
        array = np.asarray(value, order="C")
        buffer = self.device.allocator.allocate_buffer(
            memory_type=int(MemoryType.DEVICE_LOCAL),
            allowed_usage=int(BufferUsage.DEFAULT),
            allocation_size=array.nbytes,
        )
        view = HalBufferView(
            buffer, array.shape, map_dtype_to_hal_element_type(array.dtype)
        )
        source = FileHandle.wrap_memory(array, readable=True, writable=False)
        semaphore = self.device.create_semaphore(0)
        self.device.queue_read(
            source,
            buffer,
            wait_semaphores=HalFence.create_at(semaphore, 0),
            signal_semaphores=HalFence.create_at(semaphore, 1),
        )
        return AsyncResult(
            self,
            DeviceArray(self.device, view, override_dtype=array.dtype),
            semaphore,
            (array, source),
        )

    async def upload(self, value):
        return await self.enqueue_upload(value)

    def enqueue_download(self, array, *, wait_fences=()):
        self._check_loop()
        if array._device != self.device:
            raise ValueError("Array belongs to another IREE device")
        result = np.empty(array.shape, dtype=array.dtype)
        target = FileHandle.wrap_memory(result, readable=False, writable=True)
        semaphore = self.device.create_semaphore(0)
        self.device.queue_write(
            array._buffer_view.get_buffer(),
            target,
            wait_semaphores=(
                HalFence.join(wait_fences)
                if wait_fences
                else HalFence.create_at(semaphore, 0)
            ),
            signal_semaphores=HalFence.create_at(semaphore, 1),
        )
        return AsyncResult(self, result, semaphore, (array, target))

    async def download(self, array, *, wait_fences=()):
        return await self.enqueue_download(array, wait_fences=wait_fences)


class _AsyncFunction:
    def __init__(self, session, function):
        self._session = session
        self._function = function

    def submit(self, *args, wait_fences=(), **kwargs):
        self._session._check_loop()
        function = self._function
        arguments = function._arg_packer.pack(
            InvokeContext(function._device), args, kwargs
        )
        semaphore = function._device.create_semaphore(0)
        arguments.push_ref(
            HalFence.join(wait_fences)
            if wait_fences
            else HalFence.create_at(semaphore, 0)
        )
        arguments.push_ref(HalFence.create_at(semaphore, 1))
        outputs = VmVariantList(
            len(function._ret_descs) if function._ret_descs is not None else 1
        )
        function._invoke(arguments, outputs)
        aligned = outputs
        if function._has_inlined_results:
            aligned = VmVariantList(1)
            aligned.push_list(outputs)
        returns = _extract_vm_sequence_to_python(
            Invocation(function._device), aligned, function._ret_descs
        )
        value = returns[0] if len(returns) == 1 else tuple(returns) if returns else None
        return AsyncResult(
            self._session, value, semaphore, (args, kwargs, arguments, outputs)
        )

    async def __call__(self, *args, **kwargs):
        return await self.submit(*args, **kwargs)
