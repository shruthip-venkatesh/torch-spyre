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
from torch._inductor.ir import ComputedBuffer, Pointwise, Scatter, Operation
from torch._inductor.dependencies import MemoryDep
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("detect_indirect_access")


def detect_indirect_access(operations: list[Operation]) -> None:
    """
    Detect indirect access patterns in Pointwise and Scatter operations and mark them
    with appropriate op_info metadata.
    
    An indirect access pattern is detected when:
    1. A Pointwise operation (read-indirect) has multiple loads where one load's
       result is used in the index expression of another load
    2. A Scatter operation (write-indirect) has an output_indexer that uses loaded indices
    
    When detected, we add op_info with:
    - index_args: list of positions of index tensors
    - index_value_pairs: mapping of which index accesses which value tensor
    """
    logger.debug(f"detect_indirect_access: Processing {len(operations)} operations")
    
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        
        logger.debug(f"Checking operation {op.get_name()}, type={type(op.data).__name__}")

        # Check Scatter BEFORE Pointwise: Scatter is a subclass of Pointwise, so
        # isinstance(op.data, Pointwise) would match Scatter ops if checked first,
        # and _find_indirect_pattern would never find the write-indirect pattern.
        if isinstance(op.data, Scatter):
            logger.debug(f"Found Scatter operation: {op.get_name()}")

            # Skip if operation already has op_info
            if hasattr(op.data, 'op_info') and op.data.op_info:
                logger.debug(f"Scatter {op.get_name()} already has op_info, skipping")
                continue

            # Scatter needs at least 2 reads: index tensor (from output_indexer) and
            # src tensor (from inner_fn).
            reads = op.get_read_writes().reads
            logger.debug(f"Scatter {op.get_name()} has {len(reads)} read dependencies")

            if len(reads) < 2:
                logger.debug(f"Scatter {op.get_name()} has fewer than 2 reads, skipping")
                continue

            logger.debug(f"Marking scatter {op.get_name()} with indirect access")
            _mark_scatter_indirect_access(op, reads)

        # Handle plain Pointwise operations (read-indirect like gather).
        # Must come after the Scatter check above.
        elif isinstance(op.data, Pointwise):
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
    all_tensor_names = [read.name for read in reads if isinstance(read, MemoryDep)]
    
    # Reorder tensor_names to match custom op: [value, index, ...]
    # This ensures the OpSpec args are in the correct order
    value_name = all_tensor_names[value_tensor_pos]
    index_name = all_tensor_names[index_tensor_pos]
    
    tensor_names = [value_name, index_name]
    # Add any remaining tensors
    for name in all_tensor_names:
        if name not in tensor_names:
            tensor_names.append(name)
    
    # Update positions based on new ordering
    new_value_pos = 0  # Value is now first
    new_index_pos = 1  # Index is now second
    
    op_info = {
        "op": "identity",
        "index_args": [new_index_pos],
        "index_value_pairs": [
            {"index_arg": new_index_pos, "value_arg": new_value_pos}
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
            f"tensor_names={tensor_names} (reordered: value first, index second), "
            f"index_arg={new_index_pos}, value_arg={new_value_pos}"
        )


def _mark_scatter_indirect_access(op, reads) -> None:
    """Mark Scatter operation with indirect access metadata for write-indirect pattern.

    For Scatter (e.g. index_copy → index_put → ir.Scatter), PyTorch evaluates:
      1. output_indexer(vars)  — loads the index tensor  → reads[0] = INDEX tensor
      2. inner_fn/loader(vars) — loads the src tensor    → reads[1] = SRC tensor

    The scatter semantics are: output[index[i]] = src[i]
      - src  (pos 0 in tensor_names): read sequentially
      - index (pos 1 in tensor_names): determines the output write address
      - output (pos 2, added implicitly in store()): the write-indirect target

    index_value_pairs maps the index tensor (pos 1) to the scatter target (output,
    pos 2 = len(tensor_names)).  superdsc.py uses is_scatter=True to assign the
    correct layout labels: output gets "OUTPUT" (not "INPUT") and src gets "INPUT".
    """
    all_tensor_names = [read.name for read in reads if isinstance(read, MemoryDep)]

    if len(all_tensor_names) < 2:
        return  # Need at least index and src tensors

    # reads[0] = index tensor (output_indexer evaluated first)
    # reads[1] = src tensor   (inner_fn/loader evaluated second)
    index_name = all_tensor_names[0]
    source_name = all_tensor_names[1]

    # tensor_names order: [src=0, index=1]; output added at pos 2 by store()
    tensor_names = [source_name, index_name]
    for name in all_tensor_names[2:]:
        tensor_names.append(name)

    source_pos = 0
    index_pos = 1
    # output_pos = 2 (len(tensor_names)) — added implicitly by store()
    output_pos = len(tensor_names)

    op_info = {
        "op": "overwrite",
        "index_args": [index_pos],
        # Map index → output (the write-indirect target, not src)
        "index_value_pairs": [
            {"index_arg": index_pos, "value_arg": output_pos}
        ],
        "tensor_names": tensor_names,
        "is_scatter": True,
    }

    # Scatter is a frozen dataclass — use object.__setattr__ to bypass
    if not hasattr(op.data, 'op_info') or op.data.op_info is None:
        object.__setattr__(op.data, 'op_info', {})
    op.data.op_info.update(op_info)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Marked scatter operation {op.get_name()} with indirect access: "
            f"tensor_names={tensor_names}, index_arg={index_pos}, output_pos={output_pos}"
        )