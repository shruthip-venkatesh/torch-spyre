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

from typing import Any

from sympy import Symbol
from torch._inductor.virtualized import V

from .constants import SEGMENT_OFFSETS
from .errors import Unsupported
from .op_spec import OpSpec


def get_labeled_layout_metadata(
    tensor_name: str,
    dim_labels: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    from .pass_utils import concretize_expr

    buf = V.graph.get_buffer(tensor_name)
    layout = buf.get_layout()
    host_shape = [concretize_expr(s) for s in layout.size]
    host_stride = [concretize_expr(s) for s in layout.stride]
    labeled_shape = {label: int(size) for label, size in zip(dim_labels, host_shape)}
    labeled_stride = {
        label: int(stride) for label, stride in zip(dim_labels, host_stride)
    }
    return labeled_shape, labeled_stride


def enrich_indirect_index_value_pairs(
    op_info: dict[str, Any],
    dim_labels: list[str],
) -> None:
    tensor_names = op_info.get("tensor_names", [])
    enriched_pairs = []
    for pair in op_info.get("index_value_pairs", []):
        enriched_pair = dict(pair)
        index_arg = pair["index_arg"]
        value_arg = pair["value_arg"]

        if index_arg < len(tensor_names):
            index_shape, index_stride = get_labeled_layout_metadata(
                tensor_names[index_arg], dim_labels
            )
            enriched_pair["index_host_shape"] = index_shape
            enriched_pair["index_host_stride"] = index_stride

        if value_arg < len(tensor_names):
            value_shape, value_stride = get_labeled_layout_metadata(
                tensor_names[value_arg], dim_labels
            )
            enriched_pair["value_host_shape"] = value_shape
            enriched_pair["value_host_stride"] = value_stride

        enriched_pairs.append(enriched_pair)

    if enriched_pairs:
        op_info["index_value_pairs"] = enriched_pairs


def relabel_indirect_metadata_dims(
    metadata: dict[str, Any] | None,
    symbol_mapping: dict[Symbol, Symbol],
) -> dict[str, int] | None:
    if metadata is None:
        return None
    relabeled: dict[str, int] = {}
    for key, value in metadata.items():
        mapped_key = key
        if isinstance(key, str):
            for src_sym, dst_sym in symbol_mapping.items():
                if str(src_sym) == key:
                    mapped_key = str(dst_sym)
                    break
        relabeled[str(mapped_key)] = int(value)
    return relabeled


def get_indirect_pairs(op_spec: OpSpec) -> list[dict[str, Any]]:
    if not op_spec.op_info:
        return []
    return op_spec.op_info.get("index_value_pairs", [])


def find_indirect_pair_by_index_arg(
    op_spec: OpSpec,
    index_arg: int,
) -> dict[str, Any] | None:
    return next(
        (
            pair
            for pair in get_indirect_pairs(op_spec)
            if pair.get("index_arg") == index_arg
        ),
        None,
    )


def find_indirect_pair_by_value_arg(
    op_spec: OpSpec,
    value_arg: int,
) -> dict[str, Any] | None:
    return next(
        (
            pair
            for pair in get_indirect_pairs(op_spec)
            if pair.get("value_arg") == value_arg
        ),
        None,
    )


def get_relabeled_pair_metadata(
    pair: dict[str, Any] | None,
    symbol_mapping: dict[Symbol, Symbol],
) -> dict[str, dict[str, int] | None]:
    if pair is None:
        return {
            "value_host_shape": None,
            "value_host_stride": None,
            "index_host_shape": None,
            "index_host_stride": None,
        }
    return {
        "value_host_shape": relabel_indirect_metadata_dims(
            pair.get("value_host_shape"), symbol_mapping
        ),
        "value_host_stride": relabel_indirect_metadata_dims(
            pair.get("value_host_stride"), symbol_mapping
        ),
        "index_host_shape": relabel_indirect_metadata_dims(
            pair.get("index_host_shape"), symbol_mapping
        ),
        "index_host_stride": relabel_indirect_metadata_dims(
            pair.get("index_host_stride"), symbol_mapping
        ),
    }


def get_positive_indirect_dims(
    value_host_shape: dict[str, int] | None,
    index_host_shape: dict[str, int] | None,
) -> set[str]:
    if value_host_shape is None or index_host_shape is None:
        return set()
    return {
        dim_label
        for dim_label, value_dim_size in value_host_shape.items()
        if dim_label in index_host_shape
        and int(index_host_shape[dim_label]) < int(value_dim_size)
    }


def get_active_indirect_dims(
    all_dims: list[Symbol],
    value_host_shape: dict[str, int] | None,
    index_host_shape: dict[str, int] | None,
) -> set[Symbol]:
    positive_dims = get_positive_indirect_dims(value_host_shape, index_host_shape)
    if not positive_dims:
        return set(all_dims)
    return {dim for dim in all_dims if str(dim) in positive_dims}


def is_indirect_value_tensor(op_spec: OpSpec, arg_idx: int) -> bool:
    return arg_idx in [
        t.related_value_tensor_idx for t in op_spec.args if t.is_index_tensor
    ]


def get_indirect_tensor_address(
    op_spec: OpSpec,
    index_args: set[int],
    arg_idx: int,
):
    if arg_idx in index_args:
        return SEGMENT_OFFSETS[1]
    if is_indirect_value_tensor(op_spec, arg_idx):
        return SEGMENT_OFFSETS[0]
    if arg_idx == len(op_spec.args) - 1:
        return SEGMENT_OFFSETS[2]
    arg = op_spec.args[arg_idx]
    return None if arg.arg_index < 0 else SEGMENT_OFFSETS[arg.arg_index]


def get_index_related_positive_dims(
    op_spec: OpSpec,
    arg_idx: int,
    symbol_mapping: dict[Symbol, Symbol],
) -> set[str]:
    related_value_pair = find_indirect_pair_by_index_arg(op_spec, arg_idx)
    related_metadata = get_relabeled_pair_metadata(related_value_pair, symbol_mapping)
    return get_positive_indirect_dims(
        related_metadata["value_host_shape"],
        related_metadata["index_host_shape"],
    )


def get_value_pair_metadata(
    op_spec: OpSpec,
    arg_idx: int,
    symbol_mapping: dict[Symbol, Symbol],
) -> tuple[dict[str, Any] | None, dict[str, dict[str, int] | None]]:
    indirect_pair = find_indirect_pair_by_value_arg(op_spec, arg_idx)
    return indirect_pair, get_relabeled_pair_metadata(indirect_pair, symbol_mapping)


def get_value_tensor_max_dim_size(
    tensor_idx: int,
    dim: Symbol,
    stick_dim: Symbol | None,
    original_dev_dim_size: int,
    value_host_shape: dict[str, int] | None,
    index_host_shape: dict[str, int] | None,
) -> int:
    if dim == stick_dim:
        return -1

    host_dim_label = str(dim)
    value_dim_size = (
        int(value_host_shape[host_dim_label])
        if value_host_shape is not None and host_dim_label in value_host_shape
        else None
    )
    index_dim_size = (
        int(index_host_shape[host_dim_label])
        if index_host_shape is not None and host_dim_label in index_host_shape
        else None
    )

    if value_dim_size is not None and index_dim_size is not None:
        if value_dim_size == index_dim_size:
            return -1
        if index_dim_size < value_dim_size:
            return 1
        if value_dim_size < index_dim_size:
            raise Unsupported(
                f"Indirect access dimension mismatch for tensor {tensor_idx}, dim={dim}: "
                f"value_dim_size={value_dim_size} < index_dim_size={index_dim_size}. "
                f"value_host_shape={value_host_shape}, "
                f"index_host_shape={index_host_shape}"
            )

    # If dimension exists in value but NOT in index, it's a data dimension
    if value_dim_size is not None and index_dim_size is None:
        return -1  # Data dimension (not indexed)

    return original_dev_dim_size


def find_value_tensor_idx(args, index_args: set[int]) -> int | None:
    """Find the index of the value tensor in indirect access operations.

    Args:
        args: List of tensor arguments
        index_args: Set of indices that are index tensors

    Returns:
        Index of the value tensor, or None if not found
    """
    for j in range(len(args)):
        if j not in index_args and j != len(args) - 1:
            return j
    return None


def build_value_to_index_map(op_info: dict[str, Any] | None) -> dict[int, int]:
    """Build mapping of value_arg -> index_arg for indirect access operations.

    Args:
        op_info: Operation info dictionary containing index_value_pairs

    Returns:
        Dictionary mapping value tensor indices to their corresponding index tensor indices
    """
    value_to_index_map = {}
    if op_info and "index_value_pairs" in op_info:
        for pair in op_info["index_value_pairs"]:
            value_to_index_map[pair["value_arg"]] = pair["index_arg"]
    return value_to_index_map


def compute_index_tensor_layout(
    index_tensor_idx: int,
    op_spec: Any,
    symbol_mapping: dict,
    all_dims: list,
    index_args: set[int],
    logger: Any,
) -> tuple[list, Any]:
    """Compute the layout (dim_order, stick_dim) for an index tensor.

    Args:
        index_tensor_idx: Index of the index tensor
        op_spec: Operation specification
        symbol_mapping: Mapping from original symbols to SDSC symbols
        all_dims: All dimensions from iteration space
        index_args: Set of index tensor indices
        logger: Logger instance

    Returns:
        Tuple of (dim_order, stick_dim) for the index tensor
    """

    logger.debug(
        f"Index tensor {index_tensor_idx}: all_dims from iteration_space={all_dims}"
    )

    related_pair = find_indirect_pair_by_index_arg(op_spec, index_tensor_idx)
    related_metadata = get_relabeled_pair_metadata(related_pair, symbol_mapping)
    active_dims = get_active_indirect_dims(
        all_dims,
        related_metadata["value_host_shape"],
        related_metadata["index_host_shape"],
    )

    # Find value tensor to infer stick dimension
    value_tensor_idx = find_value_tensor_idx(op_spec.args, index_args)

    if value_tensor_idx is not None:
        # Import here to avoid circular dependency
        from torch_spyre._inductor.codegen.superdsc import _get_device_dim_order

        value_dim_order, value_stick_dim = _get_device_dim_order(
            op_spec.args[value_tensor_idx], symbol_mapping
        )
        logger.debug(
            f"Value tensor {value_tensor_idx}: dim_order={value_dim_order}, stick_dim={value_stick_dim}"
        )

        # Infer stick dimension if not found
        if value_stick_dim is None or value_stick_dim not in all_dims:
            value_stick_dim = all_dims[-1] if all_dims else None
            logger.debug(f"Inferred value_stick_dim={value_stick_dim}")

        # Index tensor should only keep truly indirect dimensions
        index_dim_order = [
            d for d in all_dims if d in active_dims and d != value_stick_dim
        ]

        # The index tensor's stick dimension should be the innermost (last) dimension
        index_stick_dim = index_dim_order[-1] if index_dim_order else None

        logger.debug(
            f"Index tensor {index_tensor_idx}: active_dims={sorted(map(str, active_dims))}, "
            f"final dim_order={index_dim_order}, stick_dim={index_stick_dim}"
        )
        return index_dim_order, index_stick_dim
    else:
        logger.warning(
            f"Could not find value tensor for index tensor {index_tensor_idx}"
        )
        return [], None


def compute_index_tensor_max_dim_size(
    dim: Any,
    related_value_positive_dims: set[str],
) -> int:
    """Compute maxDimSize for an index tensor dimension.

    For index tensors, only preserve dims corresponding to positive maxDimSize
    decisions on the related value tensor; all others stay dynamic.

    Args:
        dim: The dimension
        related_value_positive_dims: Set of dimension labels with positive maxDimSize on value tensor

    Returns:
        -1 if dimension should be dynamic, 0 otherwise
    """
    dim_label = str(dim)
    if dim_label in related_value_positive_dims:
        return -1
    else:
        return 0


def should_use_kernel_idx_layout(tensor_idx: int, index_args: set[int]) -> bool:
    """Determine if a tensor should use KERNEL_IDX layout.

    Args:
        tensor_idx: Index of the tensor
        index_args: Set of index tensor indices

    Returns:
        True if tensor should use KERNEL_IDX layout
    """
    return tensor_idx in index_args


def get_indirect_access_layout_label(
    tensor_idx: int,
    is_value_tensor: bool,
    is_output_tensor: bool,
    has_indirect_access: bool,
) -> str | None:
    """Get the layout label for indirect access tensors.

    Args:
        tensor_idx: Index of the tensor
        is_value_tensor: Whether this is a value tensor
        is_output_tensor: Whether this is the output tensor
        has_indirect_access: Whether the operation has indirect access

    Returns:
        Layout label string, or None if standard layout assignment should be used
    """
    if not has_indirect_access:
        return None

    if is_value_tensor:
        return "INPUT"
    elif is_output_tensor:
        return "OUTPUT"

    return None


def collect_index_tensor_layouts(
    op_spec: Any,
    symbol_mapping: dict,
    index_args: set[int],
    logger: Any,
) -> tuple[dict, dict]:
    """First pass: Collect index tensor layouts for indirect access operations.

    This function computes the layout (dim_order, stick_dim) for each index tensor
    and stores the active dimensions for later use in pass 2.

    Args:
        op_spec: Operation specification
        symbol_mapping: Mapping from original symbols to SDSC symbols
        index_args: Set of index tensor indices
        logger: Logger instance

    Returns:
        Tuple of (index_tensor_layouts, index_active_dims) where:
        - index_tensor_layouts: dict mapping tensor_idx -> (dim_order, stick_dim)
        - index_active_dims: dict mapping tensor_idx -> set of active dimensions
    """

    index_tensor_layouts = {}
    index_active_dims = {}

    for i, arg in enumerate(op_spec.args):
        if i in index_args:
            all_dims = [
                symbol_mapping.get(k, k) for k in op_spec.iteration_space.keys()
            ]

            # Use utility function to compute index tensor layout
            index_dim_order, index_stick_dim = compute_index_tensor_layout(
                i, op_spec, symbol_mapping, all_dims, index_args, logger
            )
            index_tensor_layouts[i] = (index_dim_order, index_stick_dim)

            # Store active dims for later use
            related_pair = find_indirect_pair_by_index_arg(op_spec, i)
            related_metadata = get_relabeled_pair_metadata(related_pair, symbol_mapping)
            active_dims = get_active_indirect_dims(
                all_dims,
                related_metadata["value_host_shape"],
                related_metadata["index_host_shape"],
            )
            index_active_dims[i] = active_dims

    logger.debug(f"index_tensor_layouts after first pass: {index_tensor_layouts}")
    return index_tensor_layouts, index_active_dims


def compute_indirect_max_dim_sizes(
    tensor_idx: int,
    dim: Any,
    stick_dim: Any,
    original_dev_dim_size: int,
    op_spec: Any,
    symbol_mapping: dict,
    index_args: set[int],
    index_active_dims: dict,
    logger: Any,
) -> int:
    """Compute max_dim_sizes for indirect access tensors.

    Args:
        tensor_idx: Index of the tensor
        dim: The dimension
        stick_dim: The stick dimension
        original_dev_dim_size: Original device dimension size
        op_spec: Operation specification
        symbol_mapping: Mapping from original symbols to SDSC symbols
        index_args: Set of index tensor indices
        index_active_dims: Mapping of index tensor idx to their active dimensions
        logger: Logger instance

    Returns:
        max_dim_size
    """
    is_value_tensor_for_indirect = is_indirect_value_tensor(op_spec, tensor_idx)

    if is_value_tensor_for_indirect:
        indirect_pair, indirect_metadata = get_value_pair_metadata(
            op_spec, tensor_idx, symbol_mapping
        )
        indirect_value_host_shape = indirect_metadata["value_host_shape"]
        indirect_index_host_shape = indirect_metadata["index_host_shape"]
        max_dim_size = get_value_tensor_max_dim_size(
            tensor_idx,
            dim,
            stick_dim,
            original_dev_dim_size,
            indirect_value_host_shape,
            indirect_index_host_shape,
        )
        return max_dim_size

    elif tensor_idx in index_args:
        related_value_positive_dims = get_index_related_positive_dims(
            op_spec, tensor_idx, symbol_mapping
        )
        max_dim_size = compute_index_tensor_max_dim_size(
            dim, related_value_positive_dims
        )
        return max_dim_size

    # Output tensors keep max_dim_sizes as -1 (dynamic)
    return -1


def get_indirect_layout_label(
    tensor_idx: int,
    op_spec: Any,
    index_args: set[int],
    has_indirect_access: bool,
    layouts: dict,
    dim_order: list,
    effective_stick: Any,
    stick_size: int,
    layout_labels: list[str],
    get_layout_label_func: Any,
    logger: Any,
) -> str:
    """Get layout label for indirect access tensors.

    Returns special labels for indirect access (KERNEL_IDX, INPUT, OUTPUT),
    otherwise falls back to normal layout assignment.

    Args:
        tensor_idx: Index of the tensor
        op_spec: Operation specification
        index_args: Set of index tensor indices
        has_indirect_access: Whether operation has indirect access
        layouts: Dictionary of layouts
        dim_order: Dimension order
        effective_stick: Effective stick dimension
        stick_size: Stick size
        layout_labels: List of layout labels to use
        get_layout_label_func: Function to get normal layout label
        logger: Logger instance

    Returns:
        Layout label string
    """

    # Helper to add layout to dict if not present
    def ensure_layout_exists(label: str) -> None:
        if label not in layouts:
            layouts[label] = {
                "dim_order": dim_order,
                "stick_dim_order": effective_stick,
                "stick_size": stick_size,
            }

    # Index tensors get KERNEL_IDX layout
    if should_use_kernel_idx_layout(tensor_idx, index_args):
        label = "KERNEL_IDX"
        logger.debug(f"Tensor {tensor_idx}: Assigned KERNEL_IDX layout (in index_args)")
        ensure_layout_exists(label)
        return label

    # Check for special indirect access labels (INPUT/OUTPUT)
    is_value_tensor = is_indirect_value_tensor(op_spec, tensor_idx)
    indirect_label = get_indirect_access_layout_label(
        tensor_idx,
        is_value_tensor,
        tensor_idx == len(op_spec.args) - 1,
        has_indirect_access,
    )

    if indirect_label:
        logger.debug(f"Tensor {tensor_idx}: {indirect_label} layout (indirect access)")
        ensure_layout_exists(indirect_label)
        return indirect_label

    # Fall back to normal layout assignment (same as direct access)
    return get_layout_label_func(
        layouts, dim_order, effective_stick, stick_size, layout_labels
    )


def is_indirect_access_operation(op) -> bool:
    """Check if an operation uses indirect access (e.g., gather, scatter, index_select).

    Indirect access operations have op_info with either 'index_args' or 'index_value_pairs'.

    Args:
        op: Operation to check (typically a ComputedBuffer or SchedulerNode)

    Returns:
        True if the operation uses indirect access, False otherwise
    """
    if not hasattr(op, "data"):
        return False
    if not hasattr(op.data, "op_info"):
        return False
    if not op.data.op_info:
        return False

    return "index_args" in op.data.op_info or "index_value_pairs" in op.data.op_info
