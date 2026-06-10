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

from torch._inductor.virtualized import V


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
