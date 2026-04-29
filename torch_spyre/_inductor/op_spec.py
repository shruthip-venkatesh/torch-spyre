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


import dataclasses
from typing import Any, Sequence

from sympy import Symbol, Expr
from torch_spyre._C import DataFormats


@dataclasses.dataclass
class TensorArg:
    """
    A class representing a Tensor argument to an OpSpec

    Attributes:
        is_input: Is the Tensor used as an input to the operation?
        arg_index: The index of the Tensor in the argument array of the Kernel.
        device_dtype: The device dtype of the tensor elements.
        device_size: The device size (as per SpyreTensorLayout) of the Tensor
        device_coordinates: The sympy Exprs that describe how elements in the Tensor are accessed.
                Free variables in device_coordinates refer to entries in the OpSpec's iteration_space.
        allocation: If present, the offset in scratchpad memory assigned to the Tensor.
        is_index_tensor: If True, this tensor contains indices for indirect access (KERNEL_IDX type).
        related_value_tensor_idx: If this is an index tensor, the index of the value tensor it accesses.
    """

    is_input: bool
    arg_index: int
    device_dtype: DataFormats
    device_size: list[int]
    device_coordinates: list[Expr]
    allocation: Any
    is_index_tensor: bool = False
    related_value_tensor_idx: int = -1

@dataclasses.dataclass
class OpSpec:
    """
    A class representing a single operation to perform on the device

    Attributes:
        op: The name of the operation.
        is_reduction: Is the operation a reduction?
        iteration_space: The iteration space of the operation. The values are tuples of (range, core_division).
        args: The input and output arguments to the operation.
        op_info: A dictionary of auxiliary information whose content is operation-specific.
    """

    op: str
    is_reduction: bool
    iteration_space: dict[Symbol, tuple[Expr, int]]
    args: Sequence[TensorArg]
    op_info: dict[str, Any]


@dataclasses.dataclass
class UnimplementedOp:
    op: str
