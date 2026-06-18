# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PyTorch scatter-style operations.

Tests: out[idx]=src, torch.scatter, torch.index_copy, torch.index_add,
torch.scatter_reduce, torch.index_fill, torch.masked_scatter.

CHARACTERIZATION tests, validated on the hardware build. After the work-division
indirect-store fix and the OUTPUT-layout fix, the index-tensor scatters compile
to a real op spec with IndirectAccess on the output (SCATTER_OP_SPEC). Two forms
still crash, for reasons unrelated to the indirect-store path:

  * index_fill   -> CRASHED: scalar fill value lowers to aten.scalar_tensor, a
                   rank-0 Constant whose codegen fails ("Error in codegen for
                   ComputedBuffer ... ops.constant"). A scalar-constant gap.
  * masked_scatter -> CRASHED: mask-based (not index-tensor); lowers to a CPU
                   FallbackKernel + device copy-back, which raises.

All cases run under classify_compile (which patches sdsc), so they stop at the
sdsc handoff and never reach the deeptools backend.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    CRASHED,
    SCATTER_OP_SPEC,
    IndirectAccessTestCase,
    capture_op_specs,
    classify_compile,
)

from torch_spyre._inductor import config  # noqa: E402

# Observed lowering outcome per scatter op (validated on the hardware build).
EXPECTED = {
    "index_put": SCATTER_OP_SPEC,  # VALIDATED
    "scatter": SCATTER_OP_SPEC,  # VALIDATED
    "index_copy": SCATTER_OP_SPEC,  # VALIDATED
    "index_add": SCATTER_OP_SPEC,  # VALIDATED
    "scatter_reduce": SCATTER_OP_SPEC,  # VALIDATED
    "index_fill": CRASHED,  # VALIDATED -- scalar-constant codegen gap
    "masked_scatter": CRASHED,  # VALIDATED -- mask-based CPU fallback path
}


@config.patch({"sencores": 1})
class TestTorchScatterOps(IndirectAccessTestCase):
    def test_index_put(self):
        """``out[idx] = src`` -- the standard advanced-indexing store."""
        M, N, P = 128, 256, 3
        out = torch.zeros(M, N, dtype=torch.float16)
        src = torch.rand(P, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(out, src, idx):
            out[idx] = src
            return out

        out_dev, src_dev, idx_dev = out.to("spyre"), src.to("spyre"), idx.to("spyre")
        self.name_dims(out_dev, {"M": M, "N": N})
        self.name_dims(src_dev, {"P": P, "N": N})
        self.name_dims(idx_dev, {"P": P})

        label, _ = classify_compile(kernel, out_dev, src_dev, idx_dev)
        self.assertEqual(label, EXPECTED["index_put"])

    def test_scatter(self):
        """``torch.scatter(out, 0, index, src)`` -- out-of-place scatter."""
        M, N, P = 128, 256, 3
        out = torch.zeros(M, N, dtype=torch.float16)
        src = torch.rand(P, N, dtype=torch.float16)
        index = torch.randint(0, M, (P, N), dtype=torch.int64)

        def kernel(out, src, index):
            return torch.scatter(out, 0, index, src)

        out_dev = out.to("spyre")
        src_dev = src.to("spyre")
        index_dev = index.to("spyre")
        self.name_dims(out_dev, {"M": M, "N": N})
        self.name_dims(src_dev, {"P": P, "N": N})
        self.name_dims(index_dev, {"P": P, "N": N})

        label, _ = classify_compile(kernel, out_dev, src_dev, index_dev)
        self.assertEqual(label, EXPECTED["scatter"])

    def test_index_copy(self):
        """``torch.index_copy(out, 0, idx, src)`` -- copy rows by 1-D index."""
        M, N, P = 128, 256, 3
        out = torch.zeros(M, N, dtype=torch.float16)
        src = torch.rand(P, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(out, src, idx):
            return torch.index_copy(out, 0, idx, src)

        out_dev, src_dev, idx_dev = out.to("spyre"), src.to("spyre"), idx.to("spyre")
        self.name_dims(out_dev, {"M": M, "N": N})
        self.name_dims(src_dev, {"P": P, "N": N})
        self.name_dims(idx_dev, {"P": P})

        label, _ = classify_compile(kernel, out_dev, src_dev, idx_dev)
        self.assertEqual(label, EXPECTED["index_copy"])

    def test_index_add(self):
        """``out.index_add_(0, idx, src)`` -- accumulating indirect store."""
        M, N, P = 128, 256, 3
        out = torch.zeros(M, N, dtype=torch.float16)
        src = torch.rand(P, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(out, src, idx):
            return out.index_add_(0, idx, src)

        out_dev, src_dev, idx_dev = out.to("spyre"), src.to("spyre"), idx.to("spyre")
        self.name_dims(out_dev, {"M": M, "N": N})
        self.name_dims(src_dev, {"P": P, "N": N})
        self.name_dims(idx_dev, {"P": P})

        label, _ = classify_compile(kernel, out_dev, src_dev, idx_dev)
        self.assertEqual(label, EXPECTED["index_add"])

    def test_scatter_reduce(self):
        """``out.scatter_reduce_(0, index, src, "sum")`` -- reducing scatter."""
        M, N, P = 128, 256, 3
        out = torch.zeros(M, N, dtype=torch.float16)
        src = torch.rand(P, N, dtype=torch.float16)
        index = torch.randint(0, M, (P, N), dtype=torch.int64)

        def kernel(out, src, index):
            return out.scatter_reduce_(0, index, src, "sum")

        out_dev = out.to("spyre")
        src_dev = src.to("spyre")
        index_dev = index.to("spyre")
        self.name_dims(out_dev, {"M": M, "N": N})
        self.name_dims(src_dev, {"P": P, "N": N})
        self.name_dims(index_dev, {"P": P, "N": N})

        label, _ = classify_compile(kernel, out_dev, src_dev, index_dev)
        self.assertEqual(label, EXPECTED["scatter_reduce"])

    def test_index_fill(self):
        """``out.index_fill_(0, idx, value)`` -- scalar fill.

        Crashes in codegen of the rank-0 scalar Constant (aten.scalar_tensor) --
        a scalar-constant gap separate from the indirect-store path.
        """
        M, N, P = 128, 256, 3
        out = torch.rand(M, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(out, idx):
            return out.index_fill_(0, idx, 0.0)

        out_dev, idx_dev = out.to("spyre"), idx.to("spyre")
        self.name_dims(out_dev, {"M": M, "N": N})
        self.name_dims(idx_dev, {"P": P})

        label, _ = classify_compile(kernel, out_dev, idx_dev)
        self.assertEqual(label, EXPECTED["index_fill"])

    def test_masked_scatter(self):
        """``torch.masked_scatter(out, mask, src)`` -- mask-driven store.

        Mask-based (not index-tensor): lowers to a CPU FallbackKernel + device
        copy-back, which raises. Separate from the indirect-store path.
        """
        M, N = 64, 64
        out = torch.zeros(M, N, dtype=torch.float16)
        mask = torch.randint(0, 2, (M, N), dtype=torch.bool)
        src = torch.rand(M, N, dtype=torch.float16)

        def kernel(out, mask, src):
            return torch.masked_scatter(out, mask, src)

        out_dev, mask_dev, src_dev = out.to("spyre"), mask.to("spyre"), src.to("spyre")
        self.name_dims(out_dev, {"M": M, "N": N})

        label, _ = classify_compile(kernel, out_dev, mask_dev, src_dev)
        self.assertEqual(label, EXPECTED["masked_scatter"])

    def test_scatter_method_produces_output_indirect(self):
        """``y.scatter_(0, index, src.exp())`` produces an op spec whose output
        carries IndirectAccess (detailed check via capture)."""
        M, N, P = 128, 256, 3
        y = torch.zeros(M, N, dtype=torch.float16)
        src = torch.rand(P, N, dtype=torch.float16)
        index = torch.randint(0, M, (P, N), dtype=torch.int32)

        def kernel(y, src, index):
            return y.scatter_(0, index, src.exp())

        y_dev, src_dev, index_dev = y.to("spyre"), src.to("spyre"), index.to("spyre")
        self.name_dims(y_dev, {"M": M, "N": N})
        self.name_dims(src_dev, {"P": P, "N": N})
        self.name_dims(index_dev, {"P": P, "N": N})
        with capture_op_specs() as captured:
            torch.compile(kernel)(y_dev, src_dev, index_dev)
        self.assert_scatter_op_spec(captured)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
