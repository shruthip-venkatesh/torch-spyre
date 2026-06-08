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

"""Index-to-address transformation pass for indirect access operations."""

import torch
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from .logging_utils import _get_env_bool, get_inductor_logger

aten = torch.ops.aten

logger = get_inductor_logger("index_to_address")

add_index_to_address_pass = PatternMatcherPass(pass_name="add_index_to_address")


@register_graph_pattern(
    CallFunction(aten.gather.default, Arg(), Arg(), Arg()),
    pass_dict=add_index_to_address_pass,
)
def _add_index_to_address_gather(
    match: Match,
    input_node: torch.fx.Node,
    dim: int,
    index_node: torch.fx.Node,
) -> None:
    """Transform gather operation to use address tensors for indirect access.

    Pattern: torch.gather(input, dim, index)
    Transforms index tensor to address tensor for efficient indirect memory access.
    """
    _ADD_INDEX_TO_ADDRESS_ENABLED = _get_env_bool(
        "SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS", False
    )

    node = match.nodes[-1]

    if not isinstance(input_node, torch.fx.Node) or not isinstance(
        index_node, torch.fx.Node
    ):
        return

    # Extract tensor metadata
    input_meta = input_node.meta.get("val") if hasattr(input_node, "meta") else None
    index_meta = index_node.meta.get("val") if hasattr(index_node, "meta") else None

    if input_meta is None or index_meta is None:
        return
    if not isinstance(input_meta, torch.Tensor) or not isinstance(
        index_meta, torch.Tensor
    ):
        return
    if input_meta.device.type != "spyre":
        return
    if not _ADD_INDEX_TO_ADDRESS_ENABLED:
        return

    logger.debug(" >>>> Enabled Inductor Index to Address Translation for gather <<<< ")

    # Create address tensor node
    with node.graph.inserting_before(node):
        address_node = node.graph.call_function(
            torch.ops.spyre.indices_to_address.default,
            args=(index_node, input_node, dim, 0),
        )
        address_node.meta["val"] = torch.empty(
            index_meta.shape,
            dtype=torch.int64,
            device="meta",
        )

    # Replace index with address in gather arguments: (input, dim, index) -> (input, dim, address)
    node.args = (input_node, dim, address_node)


@register_graph_pattern(
    CallFunction(aten.embedding.default, Arg(), Arg()),
    pass_dict=add_index_to_address_pass,
)
def _add_index_to_address_embedding(
    match: Match,
    weight_node: torch.fx.Node,
    indices_node: torch.fx.Node,
) -> None:
    """Transform embedding operation to use address tensors for indirect access.

    Pattern: torch.embedding(weight, indices)
    Transforms indices tensor to address tensor for efficient indirect memory access.
    Embedding always indexes along dim=0 (rows of the weight matrix).
    """
    _ADD_INDEX_TO_ADDRESS_ENABLED = _get_env_bool(
        "SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS", False
    )

    node = match.nodes[-1]

    if not isinstance(weight_node, torch.fx.Node) or not isinstance(
        indices_node, torch.fx.Node
    ):
        return

    # Extract tensor metadata
    weight_meta = weight_node.meta.get("val") if hasattr(weight_node, "meta") else None
    indices_meta = (
        indices_node.meta.get("val") if hasattr(indices_node, "meta") else None
    )

    if weight_meta is None or indices_meta is None:
        return
    if not isinstance(weight_meta, torch.Tensor) or not isinstance(
        indices_meta, torch.Tensor
    ):
        return
    if weight_meta.device.type != "spyre":
        return
    if not _ADD_INDEX_TO_ADDRESS_ENABLED:
        return

    logger.debug(
        " >>>> Enabled Inductor Index to Address Translation for embedding <<<< "
    )

    # Create address tensor node (embedding always uses dim=0)
    with node.graph.inserting_before(node):
        address_node = node.graph.call_function(
            torch.ops.spyre.indices_to_address.default,
            args=(indices_node, weight_node, 0, 0),
        )
        address_node.meta["val"] = torch.empty(
            indices_meta.shape,
            dtype=torch.int64,
            device="meta",
        )

    # Replace indices with address in embedding arguments: (weight, indices) -> (weight, address)
    node.args = (weight_node, address_node)


@register_graph_pattern(
    CallFunction(aten.index.Tensor, Arg(), Arg()),
    pass_dict=add_index_to_address_pass,
)
def _add_index_to_address_index(
    match: Match,
    input_node: torch.fx.Node,
    indices_list,
) -> None:
    """Transform aten.index.Tensor operation to use address tensors for indirect access.

    Pattern: torch.ops.aten.index.Tensor(input, [index])
    This handles the decomposed form of index_select where:
    - input_node: the tensor being indexed
    - indices_list: a list containing the index tensor (e.g., [index_tensor])

    The index operation selects elements along the first non-None dimension in indices_list.
    """
    _ADD_INDEX_TO_ADDRESS_ENABLED = _get_env_bool(
        "SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS", False
    )

    node = match.nodes[-1]

    # The indices_list should be a Python list
    if not isinstance(indices_list, (list, tuple)):
        logger.debug(
            f"indices_list is not a list/tuple, got {type(indices_list)}: {indices_list}"
        )
        return

    # Find the first non-None index in the list
    index_node = None
    dim = 0
    for i, idx in enumerate(indices_list):
        if idx is not None:
            index_node = idx
            dim = i
            break

    if index_node is None:
        logger.debug("No non-None index found in indices_list")
        return

    if not isinstance(index_node, torch.fx.Node):
        logger.debug(f"index_node is not a torch.fx.Node, got {type(index_node)}")
        return

    # Check device and enabled flag
    if not isinstance(input_node, torch.fx.Node):
        return

    # Extract tensor metadata
    input_meta = input_node.meta.get("val") if hasattr(input_node, "meta") else None
    index_meta = index_node.meta.get("val") if hasattr(index_node, "meta") else None

    if input_meta is None or index_meta is None:
        return
    if not isinstance(input_meta, torch.Tensor) or not isinstance(
        index_meta, torch.Tensor
    ):
        return
    if input_meta.device.type != "spyre":
        return
    if not _ADD_INDEX_TO_ADDRESS_ENABLED:
        return

    logger.debug(" >>>> Enabled Inductor Index to Address Translation for index <<<< ")

    # Create the address tensor
    with node.graph.inserting_before(node):
        address_node = node.graph.call_function(
            torch.ops.spyre.indices_to_address.default,
            args=(index_node, input_node, dim, 0),
        )
        address_node.meta["val"] = torch.empty(
            index_meta.shape,
            dtype=torch.int64,
            device="meta",
        )

    # For aten.index.Tensor, we need to replace the index in the list
    # while keeping the list structure intact
    new_indices_list = list(indices_list)
    new_indices_list[dim] = address_node

    # Update the node's arguments with the new indices list
    node.args = (input_node, new_indices_list)


# Made with Bob
