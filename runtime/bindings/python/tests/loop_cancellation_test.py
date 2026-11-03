# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import unittest

from iree.runtime import HalDeviceLoopBridge, get_device, query_available_drivers


class LoopCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def test_stop_cancels_pending_future_on_its_event_loop(self):
        if "local-sync" not in query_available_drivers():
            self.skipTest("local-sync driver is not available")
        loop = asyncio.get_running_loop()
        device = get_device("local-sync")
        bridge = HalDeviceLoopBridge(device, loop)
        try:
            future = bridge.on_semaphore(device.create_semaphore(0), 1, "pending")

            async def wait():
                return await future

            waiter = asyncio.create_task(wait())
            await asyncio.sleep(0)
            bridge.stop()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(waiter, 1)
        finally:
            bridge.stop()


if __name__ == "__main__":
    unittest.main()
