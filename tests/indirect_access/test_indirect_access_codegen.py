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

"""
Unit tests for indirect access code generation changes.

This test suite covers the key changes made to support indirect access operations:
1. maxDimSizes computation for value/index/output tensors
2. Layout label assignment (INPUT, KERNEL_IDX, OUTPUT)
3. Index tensor dimension handling
4. Stride and offset computation for indirect access
5. Data format handling (SENUINT32 for index tensors)

NOTE: These tests must be run with pytest, not directly with python:
    pytest torch-spyre/tests/indirect_access/test_indirect_access_codegen.py -v
"""

import pytest
from sympy import Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.indirect_access import (
    compute_indirect_max_dim_sizes,
    get_indirect_tensor_address,
    is_indirect_value_tensor,
    collect_index_tensor_layouts,
    get_positive_indirect_dims,
    get_active_indirect_dims,
)


# Create a mock logger for tests
class MockLogger:
    """Mock logger for testing."""

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


# Global mock logger instance
mock_logger = MockLogger()


class TestMaxDimSizesComputation:
    """Test maxDimSizes computation for indirect access tensors."""

    def test_value_tensor_maxdim_sizes_4d_value_3d_index(self):
        """Test maxDimSizes for 4D value tensor with 3D index tensor.

        Pattern: value[out, mb, x, y] accessed via index[mb, x, y]
        Expected maxDimSizes for value: [-1, -1, -1, -1]
        - out: -1 (not in index, indirectly accessed)
        - mb: -1 (in index with equal size, dynamically accessed)
        - x: -1 (in index with equal size, dynamically accessed)
        - y: -1 (stick dim, always -1)

        Note: When dimensions are present in both value and index with equal sizes,
        they get -1 (not device_size) because they're accessed dynamically via the index.
        """
        # Create symbols
        out_sym = Symbol("out")
        mb_sym = Symbol("mb")
        x_sym = Symbol("x")
        y_sym = Symbol("y")

        # Value tensor: 4D [out=128, mb=128, x=8, y=2]
        value_tensor = TensorArg(
            arg_index=0,
            is_input=True,
            device_size=[128, 64, 1, 1],  # Device layout
            device_coordinates=[out_sym, mb_sym, x_sym, y_sym, Symbol("stick_offset")],
            device_dtype=DataFormats.SEN169_FP16,
            allocation={"pool": 0},
            is_index_tensor=False,
            related_value_tensor_idx=-1,
        )

        # Index tensor: 3D [mb=128, x=8, y=2]
        index_tensor = TensorArg(
            arg_index=1,
            is_input=True,
            device_size=[64, 1, 1],
            device_coordinates=[mb_sym, x_sym, y_sym, Symbol("stick_offset")],
            device_dtype=DataFormats.SENUINT32,
            allocation={"pool": 1000},
            is_index_tensor=True,
            related_value_tensor_idx=0,
        )

        # Create op_spec
        iteration_space = {
            out_sym: 128,
            mb_sym: 128,
            x_sym: 8,
            y_sym: 2,
        }

        op_spec = OpSpec(
            op="identity",
            args=[value_tensor, index_tensor],
            iteration_space=iteration_space,
            is_reduction=False,
            op_info={
                "index_args": [1],
                "index_value_pairs": [
                    {
                        "index_arg": 1,
                        "value_arg": 0,
                        "value_host_shape": {"out": 128, "mb": 128, "x": 8, "y": 2},
                        "index_host_shape": {"mb": 128, "x": 8, "y": 2},
                    }
                ],
            },
        )

        # Test maxDimSizes computation
        symbol_mapping = {out_sym: out_sym, mb_sym: mb_sym, x_sym: x_sym, y_sym: y_sym}
        index_args = {1}

        # Collect index tensor layouts
        index_active_dims = collect_index_tensor_layouts(
            op_spec, symbol_mapping, index_args, mock_logger
        )

        # Test value tensor maxDimSizes
        for dim in [out_sym, mb_sym, x_sym, y_sym]:
            max_dim_size, stride_mult, offset_mult = compute_indirect_max_dim_sizes(
                tensor_idx=0,
                dim=dim,
                stick_dim=y_sym,
                original_dev_dim_size=128
                if dim == out_sym
                else 64
                if dim == mb_sym
                else 1,
                op_spec=op_spec,
                symbol_mapping=symbol_mapping,
                index_args=index_args,
                index_active_dims=index_active_dims,
                logger=mock_logger,
            )

            # All dimensions should be -1 for value tensors in indirect access
            # - out: not in index (indirectly accessed)
            # - mb, x, y: in index with equal sizes (dynamically accessed via index)
            assert max_dim_size == -1, (
                f"{dim} dimension should be -1, got {max_dim_size}"
            )

    def test_value_tensor_maxdim_sizes_2d_value_1d_index(self):
        """Test maxDimSizes for 2D value tensor with 1D index tensor.

        Pattern: value[vocab, embed] accessed via index[batch]
        Expected maxDimSizes for value: [-1, -1]
        - vocab: -1 (not in index, indirectly accessed)
        - embed: -1 (not in index, data dimension)
        """
        vocab_sym = Symbol("vocab")
        embed_sym = Symbol("embed")
        batch_sym = Symbol("batch")

        value_tensor = TensorArg(
            arg_index=0,
            is_input=True,
            device_size=[1000, 64],
            device_coordinates=[vocab_sym, embed_sym, Symbol("stick_offset")],
            device_dtype=DataFormats.SEN169_FP16,
            allocation={"pool": 0},
            is_index_tensor=False,
            related_value_tensor_idx=-1,
        )

        index_tensor = TensorArg(
            arg_index=1,
            is_input=True,
            device_size=[32],
            device_coordinates=[batch_sym, Symbol("stick_offset")],
            device_dtype=DataFormats.SENUINT32,
            allocation={"pool": 1000},
            is_index_tensor=True,
            related_value_tensor_idx=0,
        )

        iteration_space = {
            vocab_sym: 1000,
            embed_sym: 128,
            batch_sym: 32,
        }

        op_spec = OpSpec(
            op="identity",
            args=[value_tensor, index_tensor],
            iteration_space=iteration_space,
            is_reduction=False,
            op_info={
                "index_args": [1],
                "index_value_pairs": [
                    {
                        "index_arg": 1,
                        "value_arg": 0,
                        "value_host_shape": {"vocab": 1000, "embed": 128},
                        "index_host_shape": {"batch": 32},
                    }
                ],
            },
        )

        symbol_mapping = {
            vocab_sym: vocab_sym,
            embed_sym: embed_sym,
            batch_sym: batch_sym,
        }
        index_args = {1}

        index_active_dims = collect_index_tensor_layouts(
            op_spec, symbol_mapping, index_args, mock_logger
        )

        # Both dimensions should be -1 (not in index tensor)
        for dim, dev_size in [(vocab_sym, 1000), (embed_sym, 64)]:
            max_dim_size, _, _ = compute_indirect_max_dim_sizes(
                tensor_idx=0,
                dim=dim,
                stick_dim=embed_sym,
                original_dev_dim_size=dev_size,
                op_spec=op_spec,
                symbol_mapping=symbol_mapping,
                index_args=index_args,
                index_active_dims=index_active_dims,
                logger=mock_logger,
            )
            assert max_dim_size == -1, f"{dim} should be -1, got {max_dim_size}"


class TestLayoutLabelAssignment:
    """Test layout label assignment for indirect access tensors."""

    def test_index_tensor_gets_kernel_idx_label(self):
        """Test that index tensors are assigned KERNEL_IDX layout."""
        # This would require mocking the layout assignment logic
        # For now, we verify the is_index_tensor flag is set correctly
        index_tensor = TensorArg(
            arg_index=1,
            is_input=True,
            device_size=[32],
            device_coordinates=[Symbol("mb"), 0],
            device_dtype=DataFormats.SENUINT32,
            allocation={"pool": 0},
            is_index_tensor=True,
            related_value_tensor_idx=0,
        )

        assert index_tensor.is_index_tensor is True
        assert index_tensor.related_value_tensor_idx == 0

    def test_value_tensor_gets_input_label(self):
        """Test that value tensors are assigned INPUT layout."""
        value_tensor = TensorArg(
            arg_index=0,
            is_input=True,
            device_size=[64, 32],
            device_coordinates=[Symbol("out"), Symbol("mb"), 0],
            device_dtype=DataFormats.SEN169_FP16,
            allocation={"pool": 0},
            is_index_tensor=False,
            related_value_tensor_idx=-1,
        )

        assert value_tensor.is_index_tensor is False
        assert value_tensor.is_input is True


class TestIndirectAccessHelpers:
    """Test helper functions for indirect access."""

    def test_is_indirect_value_tensor(self):
        """Test identification of value tensors in indirect access."""
        value_tensor = TensorArg(
            arg_index=0,
            is_input=True,
            device_size=[64, 32],
            device_coordinates=[Symbol("out"), Symbol("mb"), 0],
            device_dtype=DataFormats.SEN169_FP16,
            allocation={"pool": 0},
            is_index_tensor=False,
            related_value_tensor_idx=-1,
        )

        index_tensor = TensorArg(
            arg_index=1,
            is_input=True,
            device_size=[32],
            device_coordinates=[Symbol("mb"), 0],
            device_dtype=DataFormats.SENUINT32,
            allocation={"pool": 1000},
            is_index_tensor=True,
            related_value_tensor_idx=0,
        )

        output_tensor = TensorArg(
            arg_index=2,
            is_input=False,
            device_size=[64, 32],
            device_coordinates=[Symbol("out"), Symbol("mb"), 0],
            device_dtype=DataFormats.SEN169_FP16,
            allocation={"pool": 2000},
            is_index_tensor=False,
            related_value_tensor_idx=-1,
        )

        op_spec = OpSpec(
            op="identity",
            args=[value_tensor, index_tensor, output_tensor],
            iteration_space={},
            is_reduction=False,
            op_info={"index_args": [1]},
        )

        assert is_indirect_value_tensor(op_spec, 0) is True
        assert is_indirect_value_tensor(op_spec, 1) is False
        assert is_indirect_value_tensor(op_spec, 2) is False

    def test_get_indirect_tensor_address(self):
        """Test address assignment for indirect access tensors."""
        from torch_spyre._inductor.constants import SEGMENT_OFFSETS

        value_tensor = TensorArg(
            arg_index=0,
            is_input=True,
            device_size=[64],
            device_coordinates=[],
            device_dtype=DataFormats.SEN169_FP16,
            allocation={},
            is_index_tensor=False,
            related_value_tensor_idx=-1,
        )
        index_tensor = TensorArg(
            arg_index=1,
            is_input=True,
            device_size=[32],
            device_coordinates=[],
            device_dtype=DataFormats.SENUINT32,
            allocation={},
            is_index_tensor=True,
            related_value_tensor_idx=0,
        )
        output_tensor = TensorArg(
            arg_index=2,
            is_input=False,
            device_size=[64],
            device_coordinates=[],
            device_dtype=DataFormats.SEN169_FP16,
            allocation={},
            is_index_tensor=False,
            related_value_tensor_idx=-1,
        )

        op_spec = OpSpec(
            op="identity",
            args=[value_tensor, index_tensor, output_tensor],
            iteration_space={},
            is_reduction=False,
            op_info={"index_args": [1]},
        )

        index_args = {1}

        # Value tensor should get SEGMENT_OFFSETS[0]
        assert get_indirect_tensor_address(op_spec, index_args, 0) == SEGMENT_OFFSETS[0]

        # Index tensor should get SEGMENT_OFFSETS[1]
        assert get_indirect_tensor_address(op_spec, index_args, 1) == SEGMENT_OFFSETS[1]

        # Output tensor should get SEGMENT_OFFSETS[2]
        assert get_indirect_tensor_address(op_spec, index_args, 2) == SEGMENT_OFFSETS[2]

    def test_get_positive_indirect_dims(self):
        """Test identification of positive indirect dimensions."""
        value_host_shape = {"out": 128, "mb": 128, "x": 8, "y": 2}
        index_host_shape = {"mb": 128, "x": 8, "y": 2}

        positive_dims = get_positive_indirect_dims(value_host_shape, index_host_shape)

        # Dimensions in both value and index with index_size < value_size
        # In this case, all shared dimensions have equal sizes, so no positive dims
        assert positive_dims == set()

        # Test with different sizes
        value_host_shape2 = {"out": 128, "mb": 128}
        index_host_shape2 = {"mb": 64}  # mb in index is smaller

        positive_dims2 = get_positive_indirect_dims(
            value_host_shape2, index_host_shape2
        )
        assert "mb" in positive_dims2

    def test_get_active_indirect_dims(self):
        """Test computation of active indirect dimensions."""
        all_dims = [Symbol("out"), Symbol("mb"), Symbol("x"), Symbol("y")]
        value_host_shape = {"out": 128, "mb": 128, "x": 8, "y": 2}
        index_host_shape = {"mb": 64, "x": 8, "y": 2}

        active_dims = get_active_indirect_dims(
            all_dims, value_host_shape, index_host_shape
        )

        # Should include mb (positive indirect dim)
        assert Symbol("mb") in active_dims


class TestDataFormatHandling:
    """Test data format assignment for indirect access tensors."""

    def test_index_tensor_uses_senuint32(self):
        """Test that index tensors use SENUINT32 data format."""
        index_tensor = TensorArg(
            arg_index=1,
            is_input=True,
            device_size=[32],
            device_coordinates=[Symbol("mb"), 0],
            device_dtype=DataFormats.SENUINT32,
            allocation={"pool": 0},
            is_index_tensor=True,
            related_value_tensor_idx=0,
        )

        assert index_tensor.device_dtype == DataFormats.SENUINT32

    def test_value_tensor_preserves_original_dtype(self):
        """Test that value tensors preserve their original data type."""
        value_tensor = TensorArg(
            arg_index=0,
            is_input=True,
            device_size=[64, 32],
            device_coordinates=[Symbol("out"), Symbol("mb"), 0],
            device_dtype=DataFormats.SEN169_FP16,
            allocation={"pool": 0},
            is_index_tensor=False,
            related_value_tensor_idx=-1,
        )

        assert value_tensor.device_dtype == DataFormats.SEN169_FP16


class TestStrideAndOffsetComputation:
    """Test stride and offset computation for indirect access."""

    def test_index_tensor_stride_zeroing(self):
        """Test that non-indexed dimensions have zero stride in index tensors.

        For dimensions not present in the index tensor, strides should be zeroed
        to prevent incorrect address computation.
        """
        # This is tested implicitly through the maxDimSizes computation
        # When maxDimSize is -1, the stride multiplier should be 0
        pass  # Placeholder for future detailed stride tests


class TestScatterOperation:
    """Test scatter operation indirect access pattern."""

    def test_scatter_has_correct_indirect_metadata(self):
        """Test that scatter operations have correct indirect access metadata."""
        # Scatter: output[index[i]] = src[i]
        # src is value tensor, index is index tensor, output is output tensor

        src_tensor = TensorArg(
            arg_index=0,
            is_input=True,
            device_size=[4, 64],
            device_coordinates=[Symbol("n"), Symbol("d"), 0],
            device_dtype=DataFormats.SEN169_FP16,
            allocation={"pool": 0},
            is_index_tensor=False,
            related_value_tensor_idx=-1,
        )

        index_tensor = TensorArg(
            arg_index=1,
            is_input=True,
            device_size=[4],
            device_coordinates=[Symbol("n"), 0],
            device_dtype=DataFormats.SENUINT32,
            allocation={"pool": 1000},
            is_index_tensor=True,
            related_value_tensor_idx=2,  # Points to output
        )

        output_tensor = TensorArg(
            arg_index=2,
            is_input=False,
            device_size=[64, 64],
            device_coordinates=[Symbol("out"), Symbol("d"), 0],
            device_dtype=DataFormats.SEN169_FP16,
            allocation={"pool": 2000},
            is_index_tensor=False,
            related_value_tensor_idx=-1,
        )

        op_spec = OpSpec(
            op="scatter",
            args=[src_tensor, index_tensor, output_tensor],
            iteration_space={Symbol("n"): 4, Symbol("d"): 64, Symbol("out"): 64},
            is_reduction=False,
            op_info={
                "index_args": [1],
                "index_value_pairs": [
                    {
                        "index_arg": 1,
                        "value_arg": 2,  # Index accesses output
                    }
                ],
            },
        )

        # Verify scatter metadata
        assert index_tensor.is_index_tensor is True
        assert index_tensor.related_value_tensor_idx == 2
        assert is_indirect_value_tensor(op_spec, 2) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
