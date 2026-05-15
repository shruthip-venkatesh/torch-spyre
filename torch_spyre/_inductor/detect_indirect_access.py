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
Detect and mark indirect access patterns in Pointwise operations.

This pass identifies operations where one tensor is loaded and its value
is used as an index to load from another tensor (indirect access pattern).
"""

import logging
from torch._inductor.ir import ComputedBuffer, Pointwise, Operation
from torch._inductor.dependencies import MemoryDep
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("detect_indirect_access")


def detect_indirect_access(operations: list[Operation]) -> None:
    """
    Detect indirect access patterns in Pointwise operations and mark them
    with appropriate op_info metadata.
    
    An indirect access pattern is detected when:
    1. A Pointwise operation has multiple loads
    2. One load's result is used in the index expression of another load
    
    When detected, we add op_info with:
    - index_args: list of positions of index tensors
    - index_value_pairs: mapping of which index accesses which value tensor
    """
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
            
        if not isinstance(op.data, Pointwise):
            continue
            
        # Check if this operation already has op_info (e.g., from indirect_gather lowering)
        if hasattr(op.data, 'op_info') and op.data.op_info:
            continue
            
        # Analyze the inner_fn to detect indirect access pattern
        # The pattern we're looking for is:
        #   tmp0 = ops.load(index_tensor, index_expr)
        #   tmp1 = ops.load(value_tensor, expr_containing_tmp0)
        
        # Get the read dependencies
        reads = op.get_read_writes().reads
        if len(reads) < 2:
            # Need at least 2 reads for indirect access
            continue
            
        # Try to detect if any read uses another read's result in its index
        # This is a heuristic based on the index expressions
        has_indirect_pattern = False
        index_tensor_pos = None
        value_tensor_pos = None
        
        for i, read in enumerate(reads):
            if not isinstance(read, MemoryDep):
                continue
                
            # Check if this read's index contains symbols that might come from another load
            # In the gather case, we see patterns like:
            #   read0: index at i1 + 2*i0 (loading from index tensor)
            #   read1: index at i1 + 128*tmp0 (loading from value tensor using loaded index)
            
            index_expr = read.index
            free_symbols = index_expr.free_symbols
            
            # Check if the index expression has more complex terms that suggest
            # it's using a loaded value (e.g., large coefficients, non-loop variables)
            for symbol in free_symbols:
                symbol_str = str(symbol)
                # If we see symbols like 'tmp0', 'tmp1', etc., it's likely an indirect access
                if symbol_str.startswith('tmp'):
                    has_indirect_pattern = True
                    # The current read is the value tensor (being indexed by loaded value)
                    value_tensor_pos = i
                    # Find the index tensor (the one being loaded to get the index)
                    for j, other_read in enumerate(reads):
                        if j != i and isinstance(other_read, MemoryDep):
                            index_tensor_pos = j
                            break
                    break
            
            if has_indirect_pattern:
                break
        
        # For indirect_gather, the tensor order should be: value, index, output
        # But we detected: index at pos 0, value at pos 1
        # So we need to swap them in the tensor_names list
        
        if has_indirect_pattern and index_tensor_pos is not None and value_tensor_pos is not None:
            # Mark this operation with indirect access metadata
            # Use "identity" operation for indirect gather (same as indirect_gather lowering)
            
            # Get tensor names in the order they appear in reads (which matches value.arguments order)
            tensor_names = [read.name for read in reads if isinstance(read, MemoryDep)]
            
            # Keep the original order from reads - don't swap!
            # The index_args positions should refer to the actual positions in value.arguments
            # In the typical gather case:
            #   reads[0] = index_tensor (arg1_1)
            #   reads[1] = value_tensor (arg0_1)
            # So index_args should be [0] (position of index in the original order)
            
            op_info = {
                "op": "identity",
                "index_args": [index_tensor_pos],
                "index_value_pairs": [
                    {"index_arg": index_tensor_pos, "value_arg": value_tensor_pos}
                ],
                "tensor_names": tensor_names
            }
            
            # Pointwise is a frozen dataclass, so we need to use object.__setattr__
            # to bypass the frozen restriction
            if not hasattr(op.data, 'op_info') or op.data.op_info is None:
                object.__setattr__(op.data, 'op_info', {})
            op.data.op_info.update(op_info)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Detected indirect access pattern in {op.get_name()}: "
                    f"index_tensor_pos={index_tensor_pos}, value_tensor_pos={value_tensor_pos}, "
                    f"op_info={op.data.op_info}"
                )

# Made with Bob
