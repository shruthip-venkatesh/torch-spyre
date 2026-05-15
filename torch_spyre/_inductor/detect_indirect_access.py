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
        if not isinstance(op, ComputedBuffer) or not isinstance(op.data, Pointwise):
            continue
            
        # Skip if operation already has op_info (e.g., from indirect_gather lowering)
        if hasattr(op.data, 'op_info') and op.data.op_info:
            continue
            
        # Get read dependencies - need at least 2 for indirect access
        reads = op.get_read_writes().reads
        if len(reads) < 2:
            continue
            
        # Detect indirect access pattern by looking for tmp variables in index expressions
        index_tensor_pos, value_tensor_pos = _find_indirect_pattern(reads)
        
        if index_tensor_pos is not None and value_tensor_pos is not None:
            _mark_indirect_access(op, reads, index_tensor_pos, value_tensor_pos)


def _find_indirect_pattern(reads) -> tuple[int | None, int | None]:
    """
    Find indirect access pattern in read dependencies.
    
    Returns:
        Tuple of (index_tensor_pos, value_tensor_pos) or (None, None) if not found
    """
    for i, read in enumerate(reads):
        if not isinstance(read, MemoryDep):
            continue
            
        # Check if index expression contains tmp variables (loaded values)
        index_expr = read.index
        for symbol in index_expr.free_symbols:
            if str(symbol).startswith('tmp'):
                # Current read is value tensor (indexed by loaded value)
                value_tensor_pos = i
                # Find index tensor (first other MemoryDep)
                for j, other_read in enumerate(reads):
                    if j != i and isinstance(other_read, MemoryDep):
                        return j, value_tensor_pos
    return None, None


def _mark_indirect_access(op, reads, index_tensor_pos: int, value_tensor_pos: int) -> None:
    """Mark operation with indirect access metadata."""
    tensor_names = [read.name for read in reads if isinstance(read, MemoryDep)]
    
    op_info = {
        "op": "identity",
        "index_args": [index_tensor_pos],
        "index_value_pairs": [
            {"index_arg": index_tensor_pos, "value_arg": value_tensor_pos}
        ],
        "tensor_names": tensor_names
    }
    
    # Pointwise is frozen dataclass - use object.__setattr__ to bypass
    if not hasattr(op.data, 'op_info') or op.data.op_info is None:
        object.__setattr__(op.data, 'op_info', {})
    op.data.op_info.update(op_info)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Detected indirect access in {op.get_name()}: "
            f"index_pos={index_tensor_pos}, value_pos={value_tensor_pos}"
        )
