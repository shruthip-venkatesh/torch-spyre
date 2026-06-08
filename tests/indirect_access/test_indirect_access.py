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

"""Unit tests for indirect_access_utils.py"""

import unittest
from unittest.mock import Mock
from sympy import Symbol

from torch_spyre._inductor.indirect_access import (
    relabel_indirect_metadata_dims,
    get_indirect_pairs,
    find_indirect_pair_by_index_arg,
    find_indirect_pair_by_value_arg,
    get_relabeled_pair_metadata,
)
from torch_spyre._inductor.op_spec import OpSpec


class TestRelabelIndirectMetadataDims(unittest.TestCase):
    """Test relabel_indirect_metadata_dims function"""

    def test_none_metadata(self):
        """Test with None metadata"""
        result = relabel_indirect_metadata_dims(None, {})
        self.assertIsNone(result)

    def test_empty_metadata(self):
        """Test with empty metadata"""
        result = relabel_indirect_metadata_dims({}, {})
        self.assertEqual(result, {})

    def test_no_symbol_mapping(self):
        """Test with metadata but no symbol mapping"""
        metadata = {"x": 10, "y": 20}
        result = relabel_indirect_metadata_dims(metadata, {})
        self.assertEqual(result, {"x": 10, "y": 20})

    def test_symbol_mapping(self):
        """Test with symbol mapping"""
        x_sym = Symbol("x")
        y_sym = Symbol("y")
        a_sym = Symbol("a")
        b_sym = Symbol("b")

        metadata = {"x": 10, "y": 20, "z": 30}
        symbol_mapping = {x_sym: a_sym, y_sym: b_sym}

        result = relabel_indirect_metadata_dims(metadata, symbol_mapping)

        # x -> a, y -> b, z stays z
        self.assertEqual(result, {"a": 10, "b": 20, "z": 30})

    def test_partial_symbol_mapping(self):
        """Test with partial symbol mapping"""
        x_sym = Symbol("x")
        a_sym = Symbol("a")

        metadata = {"x": 10, "y": 20}
        symbol_mapping = {x_sym: a_sym}

        result = relabel_indirect_metadata_dims(metadata, symbol_mapping)
        self.assertEqual(result, {"a": 10, "y": 20})


class TestGetIndirectPairs(unittest.TestCase):
    """Test get_indirect_pairs function"""

    def test_no_op_info(self):
        """Test with OpSpec that has no op_info"""
        op_spec = Mock(spec=OpSpec)
        op_spec.op_info = None

        result = get_indirect_pairs(op_spec)
        self.assertEqual(result, [])

    def test_empty_op_info(self):
        """Test with empty op_info"""
        op_spec = Mock(spec=OpSpec)
        op_spec.op_info = {}

        result = get_indirect_pairs(op_spec)
        self.assertEqual(result, [])

    def test_no_index_value_pairs(self):
        """Test with op_info but no index_value_pairs"""
        op_spec = Mock(spec=OpSpec)
        op_spec.op_info = {"other_key": "value"}

        result = get_indirect_pairs(op_spec)
        self.assertEqual(result, [])

    def test_with_index_value_pairs(self):
        """Test with index_value_pairs in op_info"""
        pairs = [
            {"index_arg": 0, "value_arg": 1},
            {"index_arg": 2, "value_arg": 3},
        ]
        op_spec = Mock(spec=OpSpec)
        op_spec.op_info = {"index_value_pairs": pairs}

        result = get_indirect_pairs(op_spec)
        self.assertEqual(result, pairs)


class TestFindIndirectPairByIndexArg(unittest.TestCase):
    """Test find_indirect_pair_by_index_arg function"""

    def test_no_pairs(self):
        """Test with no pairs"""
        op_spec = Mock(spec=OpSpec)
        op_spec.op_info = {}

        result = find_indirect_pair_by_index_arg(op_spec, 0)
        self.assertIsNone(result)

    def test_pair_not_found(self):
        """Test when pair with given index_arg is not found"""
        pairs = [
            {"index_arg": 0, "value_arg": 1},
            {"index_arg": 2, "value_arg": 3},
        ]
        op_spec = Mock(spec=OpSpec)
        op_spec.op_info = {"index_value_pairs": pairs}

        result = find_indirect_pair_by_index_arg(op_spec, 5)
        self.assertIsNone(result)

    def test_pair_found(self):
        """Test when pair with given index_arg is found"""
        pairs = [
            {"index_arg": 0, "value_arg": 1},
            {"index_arg": 2, "value_arg": 3},
        ]
        op_spec = Mock(spec=OpSpec)
        op_spec.op_info = {"index_value_pairs": pairs}

        result = find_indirect_pair_by_index_arg(op_spec, 2)
        self.assertEqual(result, {"index_arg": 2, "value_arg": 3})


class TestFindIndirectPairByValueArg(unittest.TestCase):
    """Test find_indirect_pair_by_value_arg function"""

    def test_no_pairs(self):
        """Test with no pairs"""
        op_spec = Mock(spec=OpSpec)
        op_spec.op_info = {}

        result = find_indirect_pair_by_value_arg(op_spec, 0)
        self.assertIsNone(result)

    def test_pair_not_found(self):
        """Test when pair with given value_arg is not found"""
        pairs = [
            {"index_arg": 0, "value_arg": 1},
            {"index_arg": 2, "value_arg": 3},
        ]
        op_spec = Mock(spec=OpSpec)
        op_spec.op_info = {"index_value_pairs": pairs}

        result = find_indirect_pair_by_value_arg(op_spec, 5)
        self.assertIsNone(result)

    def test_pair_found(self):
        """Test when pair with given value_arg is found"""
        pairs = [
            {"index_arg": 0, "value_arg": 1},
            {"index_arg": 2, "value_arg": 3},
        ]
        op_spec = Mock(spec=OpSpec)
        op_spec.op_info = {"index_value_pairs": pairs}

        result = find_indirect_pair_by_value_arg(op_spec, 3)
        self.assertEqual(result, {"index_arg": 2, "value_arg": 3})


class TestGetRelabeledPairMetadata(unittest.TestCase):
    """Test get_relabeled_pair_metadata function"""

    def test_none_pair(self):
        """Test with None pair"""
        result = get_relabeled_pair_metadata(None, {})

        expected = {
            "value_host_shape": None,
            "value_host_stride": None,
            "index_host_shape": None,
            "index_host_stride": None,
        }
        self.assertEqual(result, expected)

    def test_empty_pair(self):
        """Test with empty pair"""
        result = get_relabeled_pair_metadata({}, {})

        expected = {
            "value_host_shape": None,
            "value_host_stride": None,
            "index_host_shape": None,
            "index_host_stride": None,
        }
        self.assertEqual(result, expected)

    def test_pair_with_metadata(self):
        """Test with pair containing metadata"""
        x_sym = Symbol("x")
        a_sym = Symbol("a")

        pair = {
            "value_host_shape": {"x": 10, "y": 20},
            "value_host_stride": {"x": 1, "y": 10},
            "index_host_shape": {"x": 5},
            "index_host_stride": {"x": 1},
        }
        symbol_mapping = {x_sym: a_sym}

        result = get_relabeled_pair_metadata(pair, symbol_mapping)

        expected = {
            "value_host_shape": {"a": 10, "y": 20},
            "value_host_stride": {"a": 1, "y": 10},
            "index_host_shape": {"a": 5},
            "index_host_stride": {"a": 1},
        }
        self.assertEqual(result, expected)

    def test_pair_with_partial_metadata(self):
        """Test with pair containing partial metadata"""
        pair = {
            "value_host_shape": {"x": 10},
            "index_host_shape": {"y": 5},
        }

        result = get_relabeled_pair_metadata(pair, {})

        expected = {
            "value_host_shape": {"x": 10},
            "value_host_stride": None,
            "index_host_shape": {"y": 5},
            "index_host_stride": None,
        }
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
