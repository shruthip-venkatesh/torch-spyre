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

"""Unit tests for detect_indirect_access.py"""

import unittest
from unittest.mock import Mock, patch
from sympy import Symbol

from torch._inductor.dependencies import MemoryDep
from torch_spyre._inductor.detect_indirect_access import (
    detect_indirect_access,
    _find_indirect_pattern,
    _mark_indirect_access,
)


class TestFindIndirectPattern(unittest.TestCase):
    """Test _find_indirect_pattern function"""

    def test_no_reads(self):
        """Test with empty reads list"""
        result = _find_indirect_pattern([])
        self.assertEqual(result, (None, None))

    def test_single_read(self):
        """Test with single read (need at least 2 for indirect access)"""
        read = Mock()
        read.index = Mock()
        read.index.free_symbols = set()

        result = _find_indirect_pattern([read])
        self.assertEqual(result, (None, None))

    def test_no_tmp_variables(self):
        """Test with multiple reads but no tmp variables in index"""
        read1 = Mock()
        read1.index = Mock()
        read1.index.free_symbols = {Symbol("x"), Symbol("y")}

        read2 = Mock()
        read2.index = Mock()
        read2.index.free_symbols = {Symbol("z")}

        result = _find_indirect_pattern([read1, read2])
        self.assertEqual(result, (None, None))

    def test_indirect_pattern_detected(self):
        """Test when indirect pattern is detected"""
        # First read - index tensor (no tmp in index)
        read1 = Mock(spec=MemoryDep)
        read1.index = Mock()
        read1.index.free_symbols = {Symbol("x")}

        # Second read - value tensor (has tmp in index)
        read2 = Mock(spec=MemoryDep)
        read2.index = Mock()
        read2.index.free_symbols = {Symbol("tmp0"), Symbol("y")}

        result = _find_indirect_pattern([read1, read2])
        # read1 is index (pos 0), read2 is value (pos 1)
        self.assertEqual(result, (0, 1))

    def test_indirect_pattern_reversed_order(self):
        """Test when value tensor comes before index tensor"""
        # First read - value tensor (has tmp in index)
        read1 = Mock(spec=MemoryDep)
        read1.index = Mock()
        read1.index.free_symbols = {Symbol("tmp0")}

        # Second read - index tensor (no tmp in index)
        read2 = Mock(spec=MemoryDep)
        read2.index = Mock()
        read2.index.free_symbols = {Symbol("x")}

        result = _find_indirect_pattern([read1, read2])
        # read2 is index (pos 1), read1 is value (pos 0)
        self.assertEqual(result, (1, 0))

    def test_multiple_tmp_variables(self):
        """Test with multiple tmp variables"""
        read1 = Mock(spec=MemoryDep)
        read1.index = Mock()
        read1.index.free_symbols = {Symbol("x")}

        read2 = Mock(spec=MemoryDep)
        read2.index = Mock()
        read2.index.free_symbols = {Symbol("tmp0"), Symbol("tmp1")}

        result = _find_indirect_pattern([read1, read2])
        self.assertEqual(result, (0, 1))


class TestMarkIndirectAccess(unittest.TestCase):
    """Test _mark_indirect_access function"""

    def test_mark_indirect_access_basic(self):
        """Test basic marking of indirect access"""
        # Create mock operation
        op = Mock()
        op.get_name = Mock(return_value="test_op")
        op.data = Mock()
        op.data.op_info = None

        # Create mock reads as MemoryDep instances
        read1 = Mock(spec=MemoryDep)
        read1.name = "index_tensor"

        read2 = Mock(spec=MemoryDep)
        read2.name = "value_tensor"

        reads = [read1, read2]

        # Mark indirect access: index at pos 0, value at pos 1
        _mark_indirect_access(op, reads, index_tensor_pos=0, value_tensor_pos=1)

        # Verify op_info was set correctly
        self.assertIsNotNone(op.data.op_info)
        self.assertEqual(op.data.op_info["op"], "identity")
        self.assertEqual(
            op.data.op_info["index_args"], [1]
        )  # Index is second after reordering
        self.assertEqual(
            op.data.op_info["tensor_names"], ["value_tensor", "index_tensor"]
        )  # Reordered: value first
        self.assertEqual(
            op.data.op_info["index_value_pairs"],
            [{"index_arg": 1, "value_arg": 0}],  # After reordering
        )

    def test_mark_indirect_access_with_extra_tensors(self):
        """Test marking with additional tensors"""
        op = Mock()
        op.get_name = Mock(return_value="test_op")
        op.data = Mock()
        op.data.op_info = None

        read1 = Mock(spec=MemoryDep)
        read1.name = "index_tensor"

        read2 = Mock(spec=MemoryDep)
        read2.name = "value_tensor"

        read3 = Mock(spec=MemoryDep)
        read3.name = "extra_tensor"

        reads = [read1, read2, read3]

        _mark_indirect_access(op, reads, index_tensor_pos=0, value_tensor_pos=1)

        # Verify tensor_names includes all tensors with value and index first
        self.assertEqual(
            op.data.op_info["tensor_names"],
            ["value_tensor", "index_tensor", "extra_tensor"],
        )


class TestDetectIndirectAccess(unittest.TestCase):
    """Test detect_indirect_access function"""

    def test_empty_operations(self):
        """Test with empty operations list"""
        # Should not raise any errors
        detect_indirect_access([])

    def test_non_computed_buffer(self):
        """Test with non-ComputedBuffer operation"""
        op = Mock()
        op.__class__.__name__ = "SomeOtherOp"

        # Should skip this operation
        detect_indirect_access([op])

    def test_operation_with_existing_op_info(self):
        """Test operation that already has indirect access metadata"""
        from torch._inductor.ir import ComputedBuffer, Pointwise

        op = Mock(spec=ComputedBuffer)
        op.data = Mock(spec=Pointwise)
        # Set op_info with index_args to indicate indirect access
        op.data.op_info = {"index_args": [0], "existing": "info"}

        # Should skip this operation (already has indirect access)
        detect_indirect_access([op])

        # op_info should remain unchanged
        self.assertEqual(op.data.op_info, {"index_args": [0], "existing": "info"})

    def test_operation_with_single_read(self):
        """Test operation with only one read (need at least 2)"""
        from torch._inductor.ir import ComputedBuffer, Pointwise

        op = Mock(spec=ComputedBuffer)
        op.data = Mock(spec=Pointwise)
        op.data.op_info = None

        read_writes = Mock()
        read_writes.reads = [Mock()]
        op.get_read_writes = Mock(return_value=read_writes)

        # Should skip this operation
        detect_indirect_access([op])

    @patch("torch_spyre._inductor.detect_indirect_access._mark_indirect_access")
    @patch("torch_spyre._inductor.detect_indirect_access._find_indirect_pattern")
    def test_indirect_pattern_detected_and_marked(
        self, mock_find_pattern, mock_mark_access
    ):
        """Test when indirect pattern is detected and marked"""
        from torch._inductor.ir import ComputedBuffer, Pointwise

        # Setup mock operation
        op = Mock(spec=ComputedBuffer)
        op.data = Mock(spec=Pointwise)
        op.data.op_info = None

        read1 = Mock()
        read2 = Mock()
        read_writes = Mock()
        read_writes.reads = [read1, read2]
        op.get_read_writes = Mock(return_value=read_writes)

        # Mock pattern detection to return valid positions
        mock_find_pattern.return_value = (0, 1)

        # Run detection
        detect_indirect_access([op])

        # Verify pattern was searched for
        mock_find_pattern.assert_called_once_with([read1, read2])

        # Verify marking was called
        mock_mark_access.assert_called_once_with(op, [read1, read2], 0, 1)


if __name__ == "__main__":
    unittest.main()
