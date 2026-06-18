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

"""Tests for PyTorch gather-style operations.

Tests these operations: x[idx], torch.index_select, torch.gather, torch.embedding

These are CHARACTERIZATION tests - they record what the compiler currently does.
When behavior changes, update the EXPECTED dict below.

Current behavior (validated on hardware):
  * x[idx]        -> GATHER_OP_SPEC  (true indirect gather)
  * index_select  -> GATHER_OP_SPEC  (same as x[idx])
  * gather        -> GATHER_OP_SPEC  (same as x[idx])
  * embedding     -> NO_SPYRE_OP     (Currently fallback to CPU)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    GATHER_OP_SPEC,
    NO_SPYRE_OP,
    IndirectAccessTestCase,
    capture_op_specs,
    classify_compile,
    op_spec_has_indirect_input,
)

# Expected compilation outcomes for each operation (validated on hardware)
EXPECTED = {
    "advanced_index": GATHER_OP_SPEC,  # VALIDATED
    "index_select": GATHER_OP_SPEC,  # VALIDATED
    "gather": GATHER_OP_SPEC,  # VALIDATED
    "embedding": NO_SPYRE_OP,  # VALIDATED
}


class TestTorchGatherOps(IndirectAccessTestCase):
    # -- Test each operation's expected behavior -------------------------
    def test_advanced_indexing(self):
        """Test x[idx] - the standard advanced indexing gather."""
        M, N, P = 128, 256, 32
        x = torch.rand(M, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(x, idx):
            return x[idx]

        x_dev, idx_dev = x.to("spyre"), idx.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(idx_dev, {"P": P})

        label, _ = classify_compile(kernel, x_dev, idx_dev)
        self.assertEqual(label, EXPECTED["advanced_index"])

    def test_index_select(self):
        """Test torch.index_select(x, 0, idx) - select rows by 1-D index."""
        M, N, P = 128, 256, 32
        x = torch.rand(M, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(x, idx):
            return torch.index_select(x, 0, idx)

        x_dev, idx_dev = x.to("spyre"), idx.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(idx_dev, {"P": P})

        label, _ = classify_compile(kernel, x_dev, idx_dev)
        self.assertEqual(label, EXPECTED["index_select"])

    def test_gather(self):
        """Test torch.gather(x, 1, index) - index has same rank as x."""
        M, N, K = 128, 256, 64
        x = torch.rand(M, N, dtype=torch.float16)
        index = torch.randint(0, N, (M, K), dtype=torch.int64)

        def kernel(x, index):
            return torch.gather(x, 0, index)

        x_dev, index_dev = x.to("spyre"), index.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(index_dev, {"M": M, "K": K})

        label, _ = classify_compile(kernel, x_dev, index_dev)
        self.assertEqual(label, EXPECTED["gather"])

    def test_embedding(self):
        """Test torch.embedding(weight, idx) - lookup rows in embedding table."""
        V, E, P = 256, 128, 32
        weight = torch.rand(V, E, dtype=torch.float16)
        idx = torch.randint(0, V, (P,), dtype=torch.int32)

        def kernel(weight, idx):
            return torch.embedding(weight, idx)

        weight_dev, idx_dev = weight.to("spyre"), idx.to("spyre")
        self.name_dims(weight_dev, {"V": V, "E": E})
        self.name_dims(idx_dev, {"P": P})

        label, _ = classify_compile(kernel, weight_dev, idx_dev)
        self.assertEqual(label, EXPECTED["embedding"])

    # -- Test gather with fused operations -------------------------------
    def test_advanced_indexing_with_exp(self):
        """Test x[idx].exp() - gather fused with unary operation.

        Should produce gather OpSpec with IndirectAccess on input.
        """
        M, N, P = 128, 256, 32
        x = torch.rand(M, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(x, idx):
            return x[idx].exp()

        x_dev, idx_dev = x.to("spyre"), idx.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(idx_dev, {"P": P})

        with capture_op_specs() as captured:
            torch.compile(kernel)(x_dev, idx_dev)

        op_specs = self.assert_reaches_op_spec(captured)
        self.assertIn("exp", [s.op for s in op_specs])
        self.assertTrue(any(op_spec_has_indirect_input(s) for s in op_specs))

    # -- Test unsupported operations fall back to CPU --------------------
    def test_advanced_indexing_then_unsupported_unary_falls_back(self):
        """Test x[idx].sin() - unsupported op falls back to CPU.

        Gather still compiles to Spyre, but sin runs on CPU.
        """
        M, N, P = 128, 256, 32
        x = torch.rand(M, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(x, idx):
            return x[idx].sin()

        x_dev, idx_dev = x.to("spyre"), idx.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(idx_dev, {"P": P})

        label, op_specs = classify_compile(kernel, x_dev, idx_dev)
        self.assertEqual(label, GATHER_OP_SPEC)
        self.assertNotIn("sin", [s.op for s in op_specs])

    # -- Test different data types ---------------------------------------
    def test_advanced_indexing_int64_index(self):
        """Test that int64 indices work the same as int32."""
        M, N, P = 128, 256, 32
        x = torch.rand(M, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P,), dtype=torch.int64)

        def kernel(x, idx):
            return x[idx]

        x_dev, idx_dev = x.to("spyre"), idx.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(idx_dev, {"P": P})

        label, _ = classify_compile(kernel, x_dev, idx_dev)
        self.assertEqual(label, EXPECTED["advanced_index"])

    # -- Test higher-dimensional indexing --------------------------------
    def test_advanced_indexing_2d_index(self):
        """Test x[idx] with 2-D index produces 3-D output.

        Should still classify as gather OpSpec.
        """
        M, N, P, Q = 128, 256, 3, 192
        x = torch.rand(M, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P, Q), dtype=torch.int32)

        def kernel(x, idx):
            return x[idx]

        x_dev, idx_dev = x.to("spyre"), idx.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(idx_dev, {"P": P, "Q": Q})

        label, _ = classify_compile(kernel, x_dev, idx_dev)
        self.assertEqual(label, EXPECTED["advanced_index"])

    # -- fused unary on gather ops keeps the gather signature ------------
    def _named_xi(self, P=32):
        M, N = 128, 256
        x = torch.rand(M, N, dtype=torch.float16)
        idx = torch.randint(0, M, (P,), dtype=torch.int32)
        x_dev, idx_dev = x.to("spyre"), idx.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(idx_dev, {"P": P})
        return x_dev, idx_dev

    def test_advanced_indexing_with_tanh(self):
        """Test x[idx].tanh() produces gather OpSpec."""
        x_dev, idx_dev = self._named_xi()
        label, op_specs = classify_compile(lambda x, i: x[i].tanh(), x_dev, idx_dev)
        self.assertEqual(label, GATHER_OP_SPEC)
        self.assertIn("tanh", [s.op for s in op_specs])

    def test_advanced_indexing_with_abs(self):
        """Test x[idx].abs() produces gather OpSpec."""
        x_dev, idx_dev = self._named_xi()
        label, op_specs = classify_compile(lambda x, i: x[i].abs(), x_dev, idx_dev)
        self.assertEqual(label, GATHER_OP_SPEC)
        self.assertIn("abs", [s.op for s in op_specs])

    def test_index_select_with_exp(self):
        """Test index_select with exp produces gather OpSpec."""
        x_dev, idx_dev = self._named_xi()
        label, op_specs = classify_compile(
            lambda x, i: torch.index_select(x, 0, i).exp(), x_dev, idx_dev
        )
        self.assertEqual(label, GATHER_OP_SPEC)
        self.assertIn("exp", [s.op for s in op_specs])

    def test_index_select_larger_index(self):
        """Test index_select with larger index size produces gather OpSpec."""
        x_dev, idx_dev = self._named_xi(P=96)
        label, _ = classify_compile(
            lambda x, i: torch.index_select(x, 0, i), x_dev, idx_dev
        )
        self.assertEqual(label, EXPECTED["index_select"])

    def test_advanced_indexing_single_row(self):
        """Test x[idx] with single-row index produces gather OpSpec."""
        x_dev, idx_dev = self._named_xi(P=1)
        label, _ = classify_compile(lambda x, i: x[i], x_dev, idx_dev)
        self.assertEqual(label, EXPECTED["advanced_index"])


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
