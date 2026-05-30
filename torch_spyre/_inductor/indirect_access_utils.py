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
from .pass_utils import concretize_expr


def get_labeled_layout_metadata(
    tensor_name: str,
    dim_labels: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    buf = V.graph.get_buffer(tensor_name)
    layout = buf.get_layout()
    host_shape = [concretize_expr(s) for s in layout.size]
    host_stride = [concretize_expr(s) for s in layout.stride]
    labeled_shape = {
        label: int(size)
        for label, size in zip(dim_labels, host_shape)
    }
    labeled_stride = {
        label: int(stride)
        for label, stride in zip(dim_labels, host_stride)
    }
    return labeled_shape, labeled_stride


def enrich_indirect_index_value_pairs(
    op_info: dict[str, Any],
    dim_labels: list[str],
    extra_tensor_names: list[str] | None = None,
) -> None:
    """Enrich index_value_pairs with per-pair host shape/stride metadata.

    extra_tensor_names covers tensors beyond tensor_names (e.g. the implicit
    output buffer added by store()).  For scatter, value_arg == len(tensor_names)
    refers to the output, passed here as extra_tensor_names=[output_name].
    """
    tensor_names = op_info.get("tensor_names", [])
    all_names = list(tensor_names) + (extra_tensor_names or [])
    enriched_pairs = []
    for pair in op_info.get("index_value_pairs", []):
        enriched_pair = dict(pair)
        index_arg = pair["index_arg"]
        value_arg = pair["value_arg"]

        if index_arg < len(all_names):
            index_shape, index_stride = get_labeled_layout_metadata(
                all_names[index_arg], dim_labels
            )
            enriched_pair["index_host_shape"] = index_shape
            enriched_pair["index_host_stride"] = index_stride

        if value_arg < len(all_names):
            value_shape, value_stride = get_labeled_layout_metadata(
                all_names[value_arg], dim_labels
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
        (pair for pair in get_indirect_pairs(op_spec) if pair.get("index_arg") == index_arg),
        None,
    )


def find_indirect_pair_by_value_arg(
    op_spec: OpSpec,
    value_arg: int,
) -> dict[str, Any] | None:
    return next(
        (pair for pair in get_indirect_pairs(op_spec) if pair.get("value_arg") == value_arg),
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
    is_scatter: bool = False,
) -> set[str]:
    """
    Get dimensions that should have positive (non-zero) maxDimSize.
    
    For gather: dimensions where index_size < value_size (gathering subset)
    For scatter: dimensions where index_size <= value_size (scattering to positions)
    """
    if value_host_shape is None or index_host_shape is None:
        return set()
    
    if is_scatter:
        # For scatter, include dimensions where index_size <= value_size
        # This covers both equal sizes and subset cases
        return {
            dim_label
            for dim_label, value_dim_size in value_host_shape.items()
            if dim_label in index_host_shape
            and int(index_host_shape[dim_label]) <= int(value_dim_size)
        }
    else:
        # For gather, only include dimensions where index_size < value_size
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
    is_scatter: bool = False,
) -> set[Symbol]:
    positive_dims = get_positive_indirect_dims(value_host_shape, index_host_shape, is_scatter)
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
    is_scatter = op_spec.op_info.get("is_scatter", False) if op_spec.op_info else False
    related_value_pair = find_indirect_pair_by_index_arg(op_spec, arg_idx)
    related_metadata = get_relabeled_pair_metadata(related_value_pair, symbol_mapping)
    return get_positive_indirect_dims(
        related_metadata["value_host_shape"],
        related_metadata["index_host_shape"],
        is_scatter,
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
    is_scatter: bool = False,
) -> int:
    """
    Determine maxDimSize for value/output tensors in indirect access operations.
    
    For gather (read-indirect):
      - value_size == index_size: dimension traversed directly, return -1 (dynamic)
      - index_size < value_size: dimension indirectly indexed, return 1 (use actual size)
    
    For scatter (write-indirect):
      - value_size == index_size: dimension indexed for writes, return 1 (use actual size)
      - index_size < value_size: would be invalid for scatter
    """
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
            # For scatter: equal sizes mean this dimension is indexed, return 1
            # For gather: equal sizes mean direct traversal, return -1
            return 1 if is_scatter else -1
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
