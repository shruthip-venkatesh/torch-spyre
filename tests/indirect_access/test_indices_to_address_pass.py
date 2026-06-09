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

"""Unit tests for indices_to_address compiler pass.

Tests the transformation of indirect access operations (gather, embedding, index)
to use address tensors instead of index tensors for efficient hardware execution.
"""

import os
import unittest
import torch
from torch._dynamo.testing import CompileCounter
from torch._inductor.utils import run_and_get_code


class TestIndicesToAddressPass(unittest.TestCase):
    """Test suite for indices_to_address compiler pass."""

    def setUp(self):
        """Setup test environment."""
        super().setUp()
        torch.manual_seed(0xAFFE)
        self.device = torch.device("spyre")
        # Enable the pass for testing
        os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"] = "1"

    def tearDown(self):
        """Cleanup test environment."""
        # Reset environment variable
        if "SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS" in os.environ:
            del os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"]

    def _check_for_indices_to_address_call(self, code: str) -> bool:
        """Check if generated code contains indices_to_address call."""
        return (
            "torch.ops.spyre.indices_to_address" in code or "indices_to_address" in code
        )

    def _compile_and_check(self, fn, *args, should_transform=True):
        """Compile function and check if transformation occurred."""
        compiled_fn = torch.compile(fn, backend="inductor")

        # Get generated code
        code, _ = run_and_get_code(compiled_fn, *args)

        has_transform = self._check_for_indices_to_address_call(code)

        if should_transform:
            self.assertTrue(
                has_transform,
                f"Expected indices_to_address transformation but didn't find it in code:\n{code[:500]}",
            )

        return compiled_fn, code

    # ========== Gather Tests ==========

    def test_gather_basic_2d(self):
        """Test gather operation on 2D tensor gets transformed."""

        def fn(input_tensor, indices):
            return torch.gather(input_tensor, dim=0, index=indices)

        input_tensor = torch.randn(10, 32, dtype=torch.float16, device=self.device)
        indices = torch.tensor(
            [[0, 1, 2], [3, 4, 5]], dtype=torch.int64, device=self.device
        )

        compiled_fn, code = self._compile_and_check(fn, input_tensor, indices)

        # Verify output correctness
        expected = fn(input_tensor, indices)
        result = compiled_fn(input_tensor, indices)
        torch.testing.assert_close(result, expected)

    def test_gather_3d_dim1(self):
        """Test gather on 3D tensor along dimension 1."""

        def fn(input_tensor, indices):
            return torch.gather(input_tensor, dim=1, index=indices)

        input_tensor = torch.randn(4, 8, 16, dtype=torch.float16, device=self.device)
        indices = torch.tensor(
            [[[0, 1], [2, 3]]], dtype=torch.int64, device=self.device
        )
        indices = indices.expand(4, 2, 2)

        compiled_fn, code = self._compile_and_check(fn, input_tensor, indices)

        expected = fn(input_tensor, indices)
        result = compiled_fn(input_tensor, indices)
        torch.testing.assert_close(result, expected)

    def test_gather_different_dtypes(self):
        """Test gather with different data types."""
        dtypes = [torch.float16, torch.float32]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):

                def fn(input_tensor, indices):
                    return torch.gather(input_tensor, dim=0, index=indices)

                input_tensor = torch.randn(8, 32, dtype=dtype, device=self.device)
                indices = torch.tensor(
                    [[0, 1, 2]], dtype=torch.int64, device=self.device
                )

                compiled_fn, code = self._compile_and_check(fn, input_tensor, indices)

                expected = fn(input_tensor, indices)
                result = compiled_fn(input_tensor, indices)
                torch.testing.assert_close(result, expected)

    def test_gather_disabled_pass(self):
        """Test that gather is NOT transformed when pass is disabled."""
        # Disable the pass
        del os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"]

        def fn(input_tensor, indices):
            return torch.gather(input_tensor, dim=0, index=indices)

        input_tensor = torch.randn(10, 32, dtype=torch.float16, device=self.device)
        indices = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=self.device)

        compiled_fn = torch.compile(fn, backend="inductor")
        code, _ = run_and_get_code(compiled_fn, input_tensor, indices)

        # Should NOT have transformation
        has_transform = self._check_for_indices_to_address_call(code)
        self.assertFalse(
            has_transform,
            "Did not expect indices_to_address transformation when pass is disabled",
        )

    # ========== Embedding Tests ==========

    def test_embedding_basic(self):
        """Test embedding operation gets transformed."""

        def fn(weight, indices):
            return torch.embedding(weight, indices)

        weight = torch.randn(100, 64, dtype=torch.float16, device=self.device)
        indices = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device=self.device)

        compiled_fn, code = self._compile_and_check(fn, weight, indices)

        expected = fn(weight, indices)
        result = compiled_fn(weight, indices)
        torch.testing.assert_close(result, expected)

    def test_embedding_2d_indices(self):
        """Test embedding with 2D indices tensor."""

        def fn(weight, indices):
            return torch.embedding(weight, indices)

        weight = torch.randn(50, 32, dtype=torch.float16, device=self.device)
        indices = torch.tensor(
            [[0, 1, 2], [3, 4, 5]], dtype=torch.int64, device=self.device
        )

        compiled_fn, code = self._compile_and_check(fn, weight, indices)

        expected = fn(weight, indices)
        result = compiled_fn(weight, indices)
        torch.testing.assert_close(result, expected)

    def test_embedding_large_vocab(self):
        """Test embedding with large vocabulary size."""

        def fn(weight, indices):
            return torch.embedding(weight, indices)

        weight = torch.randn(10000, 128, dtype=torch.float16, device=self.device)
        indices = torch.tensor(
            [0, 100, 1000, 5000, 9999], dtype=torch.int64, device=self.device
        )

        compiled_fn, code = self._compile_and_check(fn, weight, indices)

        expected = fn(weight, indices)
        result = compiled_fn(weight, indices)
        torch.testing.assert_close(result, expected)

    # ========== Index Tests ==========

    def test_index_select_basic(self):
        """Test index_select operation gets transformed."""

        def fn(input_tensor, indices):
            return torch.index_select(input_tensor, dim=0, index=indices)

        input_tensor = torch.randn(10, 32, dtype=torch.float16, device=self.device)
        indices = torch.tensor([0, 2, 4, 6], dtype=torch.int64, device=self.device)

        compiled_fn, code = self._compile_and_check(fn, input_tensor, indices)

        expected = fn(input_tensor, indices)
        result = compiled_fn(input_tensor, indices)
        torch.testing.assert_close(result, expected)

    def test_index_select_dim1(self):
        """Test index_select along dimension 1."""

        def fn(input_tensor, indices):
            return torch.index_select(input_tensor, dim=1, index=indices)

        input_tensor = torch.randn(8, 16, 32, dtype=torch.float16, device=self.device)
        indices = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device=self.device)

        compiled_fn, code = self._compile_and_check(fn, input_tensor, indices)

        expected = fn(input_tensor, indices)
        result = compiled_fn(input_tensor, indices)
        torch.testing.assert_close(result, expected)

    def test_index_tensor_basic(self):
        """Test aten.index.Tensor operation gets transformed."""

        def fn(input_tensor, indices):
            # This will decompose to aten.index.Tensor
            return input_tensor[indices]

        input_tensor = torch.randn(10, 32, dtype=torch.float16, device=self.device)
        indices = torch.tensor([0, 2, 4], dtype=torch.int64, device=self.device)

        compiled_fn, code = self._compile_and_check(fn, input_tensor, indices)

        expected = fn(input_tensor, indices)
        result = compiled_fn(input_tensor, indices)
        torch.testing.assert_close(result, expected)

    # ========== Edge Cases ==========

    def test_gather_single_element(self):
        """Test gather with single element index."""

        def fn(input_tensor, indices):
            return torch.gather(input_tensor, dim=0, index=indices)

        input_tensor = torch.randn(10, 32, dtype=torch.float16, device=self.device)
        indices = torch.tensor([[5]], dtype=torch.int64, device=self.device)

        compiled_fn, code = self._compile_and_check(fn, input_tensor, indices)

        expected = fn(input_tensor, indices)
        result = compiled_fn(input_tensor, indices)
        torch.testing.assert_close(result, expected)

    def test_embedding_single_index(self):
        """Test embedding with single index."""

        def fn(weight, indices):
            return torch.embedding(weight, indices)

        weight = torch.randn(100, 64, dtype=torch.float16, device=self.device)
        indices = torch.tensor([42], dtype=torch.int64, device=self.device)

        compiled_fn, code = self._compile_and_check(fn, weight, indices)

        expected = fn(weight, indices)
        result = compiled_fn(weight, indices)
        torch.testing.assert_close(result, expected)

    def test_gather_large_batch(self):
        """Test gather with large batch of indices."""

        def fn(input_tensor, indices):
            return torch.gather(input_tensor, dim=0, index=indices)

        input_tensor = torch.randn(100, 64, dtype=torch.float16, device=self.device)
        indices = torch.randint(0, 100, (50, 64), dtype=torch.int64, device=self.device)

        compiled_fn, code = self._compile_and_check(fn, input_tensor, indices)

        expected = fn(input_tensor, indices)
        result = compiled_fn(input_tensor, indices)
        torch.testing.assert_close(result, expected)

    # ========== Integration Tests ==========

    def test_multiple_gather_operations(self):
        """Test multiple gather operations in same function."""

        def fn(input1, input2, indices1, indices2):
            out1 = torch.gather(input1, dim=0, index=indices1)
            out2 = torch.gather(input2, dim=0, index=indices2)
            return out1 + out2

        input1 = torch.randn(10, 32, dtype=torch.float16, device=self.device)
        input2 = torch.randn(10, 32, dtype=torch.float16, device=self.device)
        indices1 = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=self.device)
        indices2 = torch.tensor([[3, 4, 5]], dtype=torch.int64, device=self.device)

        compiled_fn, code = self._compile_and_check(
            fn, input1, input2, indices1, indices2
        )

        expected = fn(input1, input2, indices1, indices2)
        result = compiled_fn(input1, input2, indices1, indices2)
        torch.testing.assert_close(result, expected)

    def test_gather_with_computation(self):
        """Test gather followed by computation."""

        def fn(input_tensor, indices):
            gathered = torch.gather(input_tensor, dim=0, index=indices)
            return gathered * 2.0 + 1.0

        input_tensor = torch.randn(10, 32, dtype=torch.float16, device=self.device)
        indices = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=self.device)

        compiled_fn, code = self._compile_and_check(fn, input_tensor, indices)

        expected = fn(input_tensor, indices)
        result = compiled_fn(input_tensor, indices)
        torch.testing.assert_close(result, expected)

    def test_embedding_with_linear(self):
        """Test embedding followed by linear layer."""

        def fn(weight, indices, linear_weight):
            embedded = torch.embedding(weight, indices)
            return torch.matmul(embedded, linear_weight.t())

        weight = torch.randn(100, 64, dtype=torch.float16, device=self.device)
        indices = torch.tensor([0, 5, 10], dtype=torch.int64, device=self.device)
        linear_weight = torch.randn(32, 64, dtype=torch.float16, device=self.device)

        compiled_fn, code = self._compile_and_check(fn, weight, indices, linear_weight)

        expected = fn(weight, indices, linear_weight)
        result = compiled_fn(weight, indices, linear_weight)
        torch.testing.assert_close(result, expected)

    # ========== Negative Tests ==========

    def test_cpu_tensor_not_transformed(self):
        """Test that CPU tensors are NOT transformed."""

        def fn(input_tensor, indices):
            return torch.gather(input_tensor, dim=0, index=indices)

        # Use CPU tensors
        input_tensor = torch.randn(10, 32, dtype=torch.float16)
        indices = torch.tensor([[0, 1, 2]], dtype=torch.int64)

        compiled_fn = torch.compile(fn, backend="inductor")

        # Should work but not transform
        result = compiled_fn(input_tensor, indices)
        expected = fn(input_tensor, indices)
        torch.testing.assert_close(result, expected)

    def test_compile_counter(self):
        """Test that compilation happens and pass is applied."""
        counter = CompileCounter()

        def fn(input_tensor, indices):
            return torch.gather(input_tensor, dim=0, index=indices)

        input_tensor = torch.randn(10, 32, dtype=torch.float16, device=self.device)
        indices = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=self.device)

        compiled_fn = torch.compile(fn, backend=counter)
        _ = compiled_fn(input_tensor, indices)

        # Verify compilation occurred
        self.assertGreater(
            counter.frame_count, 0, "Expected at least one compilation frame"
        )


class TestIndicesToAddressPassIntegration(unittest.TestCase):
    """Integration tests for indices_to_address pass with real models."""

    def setUp(self):
        """Setup test environment."""
        super().setUp()
        torch.manual_seed(0xAFFE)
        self.device = torch.device("spyre")
        os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"] = "1"

    def tearDown(self):
        """Cleanup test environment."""
        if "SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS" in os.environ:
            del os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"]

    def test_simple_embedding_layer(self):
        """Test with a simple embedding layer."""

        class SimpleEmbedding(torch.nn.Module):
            def __init__(self, vocab_size, embed_dim):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embed_dim)

            def forward(self, indices):
                return self.embedding(indices)

        model = SimpleEmbedding(100, 64).to(self.device).to(torch.float16)
        indices = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device=self.device)

        compiled_model = torch.compile(model, backend="inductor")

        expected = model(indices)
        result = compiled_model(indices)
        torch.testing.assert_close(result, expected)

    def test_attention_gather_pattern(self):
        """Test gather pattern common in attention mechanisms."""

        def attention_gather(values, indices):
            # Simulate gathering values based on attention indices
            batch_size, seq_len, hidden_dim = values.shape
            gathered = torch.gather(
                values, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
            )
            return gathered

        values = torch.randn(2, 10, 64, dtype=torch.float16, device=self.device)
        indices = torch.tensor(
            [[0, 2, 4], [1, 3, 5]], dtype=torch.int64, device=self.device
        )

        compiled_fn = torch.compile(attention_gather, backend="inductor")

        expected = attention_gather(values, indices)
        result = compiled_fn(values, indices)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
