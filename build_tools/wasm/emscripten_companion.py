# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Embed an existing Wasm JS companion in an Emscripten C source template."""

import argparse
from pathlib import Path

from wasm_binary_bundler import generate_merger, transform_companion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    template = args.template.read_text()
    marker = "@IREE_WASM_IMPORTS@"
    if template.count(marker) != 1:
        parser.error(f"{args.template} must contain exactly one {marker}")
    companion = transform_companion(
        args.source.read_text(), args.module, str(args.source)
    )
    merger = generate_merger([args.module]).replace("export function ", "function ")
    args.output.write_text(template.replace(marker, companion + merger))


if __name__ == "__main__":
    main()
