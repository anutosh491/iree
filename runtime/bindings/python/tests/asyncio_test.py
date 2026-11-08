# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import gc
import unittest
import weakref
from types import SimpleNamespace

import numpy as np

import iree.runtime as rt


class AsyncSessionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if "local-sync" not in rt.query_available_drivers():
            self.skipTest("local-sync driver is not available")
        self.device = rt.get_device("local-sync")
        self.session = rt.AsyncSession(self.device)
        self.addCleanup(self.session.close)

    async def test_transfers(self):
        for dtype in (np.float32, np.float16, np.bool_):
            with self.subTest(dtype=dtype):
                values = np.asarray(np.arange(24).reshape(4, 6) % 2, dtype=dtype)[
                    :, ::2
                ]
                array = await self.session.upload(values)
                np.testing.assert_array_equal(
                    await self.session.download(array), values
                )

    async def test_wait_cancellation_retains_owners_until_completion(self):
        class Owner:
            pass

        owner = Owner()
        weak_owner = weakref.ref(owner)
        semaphore = self.device.create_semaphore(0)
        pending = rt.AsyncResult(self.session, "complete", semaphore, (owner,))
        waiter = asyncio.create_task(pending.result())
        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        del owner, pending, waiter
        gc.collect()
        self.assertIsNotNone(weak_owner())
        semaphore.signal(1)

        async def wait_for_owner_release():
            while weak_owner() is not None:
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_for_owner_release(), 1)
        gc.collect()
        self.assertIsNone(weak_owner())

    async def test_ready_result_can_be_awaited_twice(self):
        pending = rt.AsyncResult(self.session, "ready", self.device.create_semaphore(1))
        self.assertEqual(await pending, "ready")
        self.assertEqual(await pending, "ready")

    async def test_close_cancels_waits_and_rejects_new_work(self):
        pending = rt.AsyncResult(
            self.session, "pending", self.device.create_semaphore(0)
        )
        waiter = asyncio.create_task(pending.result())
        await asyncio.sleep(0)
        self.session.close()
        self.session.close()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.session.enqueue_upload(np.ones(4, dtype=np.float32))

    async def test_calling_convention_is_explicit(self):
        function = SimpleNamespace(
            _device=self.device,
            vm_function=SimpleNamespace(reflection={"iree.abi.model": "other"}),
        )
        with self.assertRaisesRegex(ValueError, "coarse-fences"):
            self.session.bind(function)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.session.bind(function, abi="coarse-fences")


if __name__ == "__main__":
    unittest.main()
