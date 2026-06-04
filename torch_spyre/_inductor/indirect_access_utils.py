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

from typing import Any, Callable

from sympy import Symbol
from torch._inductor.virtualized import V

from .constants import SEGMENT_OFFSETS
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
            raise ValueError(
                f"Indirect access dimension mismatch for tensor {tensor_idx}, dim={dim}: "
                f"value_dim_size={value_dim_size} < index_dim_size={index_dim_size}. "
                f"value_host_shape={value_host_shape}, "
                f"index_host_shape={index_host_shape}"
            )

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
            op_spec.args[value_tensor_idx], symbol_mapping, reverse_for_indirect=False
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


def should_zero_stride_offset(
    tensor_idx: int,
    dim: Any,
    stick_dim: Any,
    index_active_dims: dict[int, set],
    is_value_tensor: bool,
) -> bool:
    """Determine if stride and offset should be zeroed for a dimension.

    For value tensors in indirect access, dimensions that are not actively indexed
    (i.e., used for work division only) should have zero stride and offset to prevent
    work tile coordinates from corrupting address calculations.

    Args:
        tensor_idx: Index of the tensor
        dim: The dimension to check
        stick_dim: The stick dimension
        index_active_dims: Mapping of index tensor idx to their active dimensions
        is_value_tensor: Whether this is a value tensor

    Returns:
        True if stride and offset should be zeroed, False otherwise
    """
    if not is_value_tensor:
        return False

    if tensor_idx not in index_active_dims:
        return False

    active_dims = index_active_dims[tensor_idx]
    return dim not in active_dims and dim != stick_dim


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


def should_reverse_output_dims_for_indirect(
    tensor_idx: int,
    num_tensors: int,
    has_indirect_access: bool,
    dim_order_len: int,
) -> bool:
    """Determine if output tensor dimensions should be reversed for indirect access.

    For output tensors in indirect access (gather), reverse the non-stick dimensions
    to maintain correct dimension order from input tensor.

    Args:
        tensor_idx: Index of the tensor
        num_tensors: Total number of tensors
        has_indirect_access: Whether the operation has indirect access
        dim_order_len: Length of the dimension order

    Returns:
        True if dimensions should be reversed
    """
    return has_indirect_access and tensor_idx == num_tensors - 1 and dim_order_len > 2


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


def create_sdsc_arg_for_tensor(
    tensor_idx: int,
    arg: Any,
    op_spec: Any,
    symbol_mapping: dict,
    iteration_space: dict,
    dims: list,
    op_dim_order: list,
    op_stick_dim: Any,
    use_op_dims: bool,
    index_args: set[int],
    has_indirect_access: bool,
    index_tensor_layouts: dict,
    index_active_dims: dict,
    adjusted_output_size: list,
    sdsc_args: list,
    layouts: dict,
    layout_labels: list[str],
    get_device_dim_order_func: Callable,
    calculate_device_stride_func: Callable,
    get_layout_label_func: Callable,
    logger: Any,
) -> Any:
    """Second pass: Create SDSC arg for a single tensor with proper layout handling.

    This function handles all the complexity of creating an SDSCArgs object for a tensor,
    including layout determination, stride/offset calculation, and indirect access handling.

    Args:
        tensor_idx: Index of the tensor being processed
        arg: TensorArg from op_spec
        op_spec: Operation specification
        symbol_mapping: Mapping from original symbols to SDSC symbols
        iteration_space: SDSC iteration space
        dims: List of all dimensions
        op_dim_order: Operation dimension order
        op_stick_dim: Operation stick dimension
        use_op_dims: Whether to use operation dimensions
        index_args: Set of index tensor indices
        has_indirect_access: Whether operation has indirect access
        index_tensor_layouts: Pre-computed layouts from pass 1
        index_active_dims: Pre-computed active dims from pass 1
        adjusted_output_size: Adjusted output size for overwrite ops
        sdsc_args: List of SDSC args being built
        layouts: Dictionary of layouts
        layout_labels: List of layout labels to use
        get_device_dim_order_func: Function to get device dimension order
        calculate_device_stride_func: Function to calculate device stride
        get_layout_label_func: Function to get layout label
        logger: Logger instance

    Returns:
        SDSCArgs object for this tensor
    """
    import math
    from torch_spyre._C import DataFormats
    from torch_spyre._inductor.constants import (
        SEGMENT_OFFSETS,
        MATMUL_LAYOUT_LABELS,
        LAYOUT_LABELS,
    )

    # Import SDSCArgs here to avoid circular dependency
    from torch_spyre._inductor.codegen.superdsc import SDSCArgs

    # For indirect access: assign addresses based on tensor role
    addr = (
        get_indirect_tensor_address(op_spec, index_args, tensor_idx)
        if has_indirect_access
        else None
        if arg.arg_index < 0
        else SEGMENT_OFFSETS[arg.arg_index]
    )

    # Check if this is a value tensor for indirect access
    is_value_tensor = is_indirect_value_tensor(op_spec, tensor_idx)

    # For index tensors, use pre-computed layout from first pass
    if tensor_idx in index_tensor_layouts:
        dim_order, stick_dim = index_tensor_layouts[tensor_idx]
    else:
        # For value tensors in indirect access, use op_dim_order directly
        if is_value_tensor and has_indirect_access:
            dim_order = list(op_dim_order)
            stick_dim = op_dim_order[0] if op_dim_order else None
        else:
            dim_order, stick_dim = get_device_dim_order_func(
                arg, symbol_mapping, reverse_for_indirect=False
            )
            # For output tensors in indirect access (gather), reverse the non-stick dimensions
            if should_reverse_output_dims_for_indirect(
                tensor_idx, len(op_spec.args), has_indirect_access, len(dim_order)
            ):
                non_stick_dims = dim_order[:-1]
                non_stick_dims.reverse()
                dim_order = non_stick_dims + [dim_order[-1]]
                logger.debug(
                    f"Tensor {tensor_idx}: Reversed non-stick dims for indirect access output, new dim_order={dim_order}"
                )

    logger.debug(
        f"Tensor {tensor_idx}: Initial dim_order={dim_order}, stick_dim={stick_dim}, "
        f"device_size={arg.device_size}, device_coords={arg.device_coordinates}, "
        f"is_index_tensor={arg.is_index_tensor}, related_value_tensor_idx={arg.related_value_tensor_idx}"
    )

    scales = {}
    strides = {}
    offsets = {}
    backGap: dict[Symbol, int] = {}
    max_dim_sizes = {}
    reduced_dims = []
    use_adjusted_size = op_spec.op == "overwrite" and not arg.is_input

    # For index tensors, don't add reduced dimensions - keep the layout as determined in first pass
    if use_op_dims and dim_order != dims and tensor_idx not in index_args:
        reduced_dims = [d for d in op_dim_order if d not in dim_order]
        dim_order = dim_order + reduced_dims

    if op_stick_dim is None and tensor_idx not in index_args:
        # No stick dim found in op - add one (but not for index tensors)
        stick_dim = next(d for d in dims if d not in op_dim_order)
        dim_order = dim_order + [stick_dim]
        logger.debug(
            f"Tensor {tensor_idx}: Added stick_dim={stick_dim}, new dim_order={dim_order}"
        )

    if op_spec.op == "layernormscale" and len(sdsc_args) == 0:
        reduced_dims = [stick_dim]

    stride_dim_order = [d for d in dim_order if d not in reduced_dims] + reduced_dims

    # For indirect access: all scales should be 1 (no reduction)
    is_value_or_index_tensor = tensor_idx in index_args or tensor_idx in [
        t.related_value_tensor_idx for t in op_spec.args if t.is_index_tensor
    ]

    for dim in dim_order:
        stride_idx = stride_dim_order.index(dim)
        if is_value_or_index_tensor:
            scales[dim] = 1  # No reduction for indirect access tensors
        elif dim in reduced_dims and op_spec.op != "layernormscale":
            scales[dim] = -2 if (stick_dim is None and dim is op_stick_dim) else -1
        elif dim in reduced_dims and op_spec.op == "layernormscale":
            scales[dim] = -2 if (dim is stick_dim) else -1
        else:
            scales[dim] = 1

        strides[dim] = calculate_device_stride_func(
            stride_idx,
            arg.device_size if not use_adjusted_size else adjusted_output_size,
        )
        offsets[dim] = 0
        dim_device_stride = math.prod(arg.device_size[-stride_idx - 1 :])

        # Store the original device dimension size before any modifications
        original_dev_dim_size = arg.device_size[-stride_idx - 2]
        dev_dim_size = original_dev_dim_size
        it_dim_size = iteration_space[dim]
        if dim == stick_dim:
            stick_size = arg.device_dtype.elems_per_stick()
            dev_dim_size *= stick_size
            it_dim_size = ((it_dim_size - 1) // stick_size + 1) * stick_size

        if dev_dim_size > it_dim_size:
            dim_coord = arg.device_coordinates[-stride_idx - 2]
            dim_offset = int(dim_coord.as_coeff_Add()[0])
            offsets[dim] = dim_offset * dim_device_stride
            # backGap[dim] = dev_dim_size - it_dim_size # TODO: fix this
            strides[dim] = strides[dim] // dev_dim_size * it_dim_size

        # Set max_dim_sizes for indirect access
        is_value_tensor_for_indirect = is_indirect_value_tensor(op_spec, tensor_idx)
        indirect_pair, indirect_metadata = get_value_pair_metadata(
            op_spec, tensor_idx, symbol_mapping
        )
        indirect_value_host_shape = indirect_metadata["value_host_shape"]
        indirect_value_host_stride = indirect_metadata["value_host_stride"]
        indirect_index_host_shape = indirect_metadata["index_host_shape"]
        indirect_index_host_stride = indirect_metadata["index_host_stride"]
        related_value_positive_dims = get_index_related_positive_dims(
            op_spec, tensor_idx, symbol_mapping
        )

        if is_value_tensor_for_indirect:
            max_dim_sizes[dim] = get_value_tensor_max_dim_size(
                tensor_idx,
                dim,
                stick_dim,
                original_dev_dim_size,
                indirect_value_host_shape,
                indirect_index_host_shape,
            )

            # CRITICAL FIX: Zero out strides and offsets for non-indexed dimensions
            if should_zero_stride_offset(
                tensor_idx,
                dim,
                stick_dim,
                index_active_dims,
                is_value_tensor_for_indirect,
            ):
                strides[dim] = 0
                offsets[dim] = 0
                logger.debug(
                    f"Tensor {tensor_idx} (value), dim={dim}: ZEROED stride/offset "
                    f"(not in active_indirect_dims={sorted(map(str, index_active_dims.get(tensor_idx, set())))})"
                )

            logger.debug(
                f"Tensor {tensor_idx} (value), dim={dim}: "
                f"maxDimSize={max_dim_sizes[dim]}, stride_idx={stride_idx}, "
                f"value_host_shape={indirect_value_host_shape}, "
                f"value_host_stride={indirect_value_host_stride}, "
                f"index_host_shape={indirect_index_host_shape}, "
                f"index_host_stride={indirect_index_host_stride}, "
                f"device_size={arg.device_size}, strides={strides}, offsets={offsets}, backGap={backGap}"
            )
        else:
            # For index tensors, compute maxDimSize using utility function
            if tensor_idx in index_args:
                max_dim_sizes[dim] = compute_index_tensor_max_dim_size(
                    dim, related_value_positive_dims
                )
                logger.debug(
                    f"Tensor {tensor_idx} (index), dim={dim}: maxDimSize={max_dim_sizes[dim]}, "
                    f"stride_idx={stride_idx}, dim_label={str(dim)}, "
                    f"related_value_positive_dims={sorted(related_value_positive_dims)}, "
                    f"device_size={arg.device_size}, device_coords={arg.device_coordinates}, "
                    f"op_info_pairs={op_spec.op_info.get('index_value_pairs')}"
                )
            else:
                # Output tensors keep max_dim_sizes as -1 (dynamic)
                max_dim_sizes[dim] = -1

    effective_stick = op_stick_dim if stick_dim is None else stick_dim

    # Determine layout label using utility functions
    if should_use_kernel_idx_layout(tensor_idx, index_args):
        label = "KERNEL_IDX"
        logger.debug(f"Tensor {tensor_idx}: Assigned KERNEL_IDX layout (in index_args)")
        logger.debug(
            f"Tensor {tensor_idx}: About to create KERNEL_IDX layout with dim_order={dim_order}, effective_stick={effective_stick}, scales.keys()={list(scales.keys())}"
        )
        if "KERNEL_IDX" not in layouts:
            layouts["KERNEL_IDX"] = {
                "dim_order": dim_order,
                "stick_dim_order": effective_stick,
                "stick_size": arg.device_dtype.elems_per_stick(),
            }
    else:
        # Try to get indirect access layout label
        indirect_label = get_indirect_access_layout_label(
            tensor_idx,
            is_value_tensor,
            tensor_idx == len(op_spec.args) - 1,
            has_indirect_access,
        )

        if indirect_label:
            label = indirect_label
            logger.debug(f"Tensor {tensor_idx}: {label} layout (indirect access)")
            if label not in layouts:
                layouts[label] = {
                    "dim_order": dim_order,
                    "stick_dim_order": effective_stick,
                    "stick_size": arg.device_dtype.elems_per_stick(),
                }
        else:
            # Use normal layout assignment for other tensors
            label = get_layout_label_func(
                layouts,
                dim_order,
                effective_stick,
                arg.device_dtype.elems_per_stick(),
                MATMUL_LAYOUT_LABELS if not use_op_dims else LAYOUT_LABELS,
            )
        logger.debug(
            f"Tensor {tensor_idx}: Assigned {label} layout, dtype={arg.device_dtype.name}, stick_size={arg.device_dtype.elems_per_stick()}"
        )

    # Override dtype for index tensors: they use IEEE_FP32 in PyTorch but SENUINT32 in SDSC
    sdsc_dtype = DataFormats.SENUINT32 if arg.is_index_tensor else arg.device_dtype

    return SDSCArgs(
        layout=label,
        data_format=sdsc_dtype,
        dim_order=dim_order,
        scales=scales,
        strides=strides,
        offsets=offsets,
        max_dim_sizes=max_dim_sizes,
        allocation=arg.allocation,
        start_address=addr
        if (arg.is_index_tensor or has_indirect_access)
        else arg.allocation.get("pool")
        if "pool" in arg.allocation
        else arg.allocation.get("lx")
        if "lx" in arg.allocation
        else arg.allocation.get("hbm"),
        backGap=backGap,
        is_index_tensor=arg.is_index_tensor,
        related_value_tensor_idx=arg.related_value_tensor_idx,
    )
