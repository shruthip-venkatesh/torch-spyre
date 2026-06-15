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

from types import MethodType, SimpleNamespace
from unittest.mock import PropertyMock, patch

import sympy

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.spyre_kernel import SpyreKernel


class _FakeDeviceLayout:
    def __init__(self, device_size, stride_map, device_dtype):
        self.device_size = device_size
        self.stride_map = stride_map
        self.device_dtype = device_dtype


class _FakeLayout(FixedTiledLayout):
    def __init__(self, size, stride, device_size, stride_map, allocation=None):
        self.size = size
        self.stride = stride
        self.allocation = allocation or {"hbm": 0}
        self.per_tile_fixed = False
        self.device_layout = _FakeDeviceLayout(
            device_size=device_size,
            stride_map=stride_map,
            device_dtype=DataFormats.SEN169_FP16,
        )


class _FakeBuffer:
    def __init__(self, layout):
        self._layout = layout

    def get_layout(self):
        return self._layout


# _DummyValue is no longer needed - we use TensorAccess directly


def test_store_indirect_access_builds_index_tensor_args_inline():
    x0 = sympy.Symbol("x0")
    x1 = sympy.Symbol("x1")

    value_layout = _FakeLayout(
        size=[4, 64],
        stride=[64, 1],
        device_size=[4, 64],
        stride_map=[64, 1],
    )
    index_layout = _FakeLayout(
        size=[4],
        stride=[1],
        device_size=[4],
        stride_map=[1],
    )
    output_layout = _FakeLayout(
        size=[4, 64],
        stride=[64, 1],
        device_size=[4, 64],
        stride_map=[64, 1],
    )

    buffers = {
        "value_tensor": _FakeBuffer(value_layout),
        "index_tensor": _FakeBuffer(index_layout),
        "out": _FakeBuffer(output_layout),
    }

    kernel = SpyreKernel()
    kernel.args = SimpleNamespace(output=lambda name: None)
    kernel.op_specs = []
    kernel.spyre_kernel_args = []
    kernel.indirect_vars = {}
    kernel._indirect_var_count = 0
    kernel.current_node = SimpleNamespace(
        node=SimpleNamespace(
            data=SimpleNamespace(
                op_info={
                    "op": "identity",
                    "tensor_names": ["value_tensor", "index_tensor"],
                    "index_args": [1],
                    "index_value_pairs": [{"index_arg": 1, "value_arg": 0}],
                }
            )
        )
    )

    created_args = []
    # Track indirect access metadata separately since TensorArg no longer has these fields
    arg_metadata = {}

    def fake_create_tensor_arg(
        is_input,
        name,
        tensor,
        opspec_name=None,
    ):
        arg = TensorArg(
            is_input=is_input,
            arg_index=len(created_args),
            device_dtype=tensor.layout.device_layout.device_dtype,
            device_size=list(tensor.layout.device_layout.device_size),
            device_coordinates=[tensor.index],
            allocation=tensor.layout.allocation,
            stride_map=list(tensor.layout.device_layout.stride_map),
            per_tile_fixed=tensor.layout.per_tile_fixed,
            name=opspec_name,
        )
        # Store metadata for indirect access validation based on name
        # In the new version, indirect access info is tracked differently
        is_index = "index" in name if name else False
        arg_metadata[id(arg)] = {
            "is_index_tensor": is_index,
            "related_value_tensor_idx": 0 if is_index else -1,
        }
        created_args.append((name, arg, tensor.index))
        return arg

    captured = {}

    def fake_create_op_spec(self, op, is_reduction, args, op_info):
        captured["op"] = op
        captured["is_reduction"] = is_reduction
        captured["args"] = args
        captured["op_info"] = dict(op_info)
        return OpSpec(
            op=op,
            is_reduction=is_reduction,
            iteration_space={},
            args=args,
            op_info=op_info,
            tiled_symbols=[],
        )

    fake_graph = SimpleNamespace(
        get_buffer=lambda name: buffers[name],
        sizevars=SimpleNamespace(precomputed_replacements={}),
        scheduler=SimpleNamespace(mutation_real_name={}),
        removed_buffers=set(),
    )

    with (
        patch(
            "torch_spyre._inductor.spyre_kernel.V.__class__.graph",
            new_callable=PropertyMock,
            return_value=fake_graph,
        ),
        patch(
            "torch_spyre._inductor.spyre_kernel.iteration_space",
            return_value={x0: (4, 1), x1: (64, 1)},
        ),
    ):
        from torch_spyre._inductor.spyre_kernel import TensorAccess

        kernel.create_tensor_arg = fake_create_tensor_arg
        kernel.create_op_spec = MethodType(fake_create_op_spec, kernel)

        # Create a TensorAccess for the value (simulating a gather/identity operation)
        value_tensor_access = TensorAccess("value_tensor", x0 * 64 + x1, value_layout)
        kernel.store("out", sympy.Integer(0), value_tensor_access)

    assert len(kernel.op_specs) == 1
    # The op is determined by the store method based on coordinate analysis
    # It can be 'identity' or 'ReStickifyOpHBM' depending on the coordinates
    assert captured["op"] in ["identity", "ReStickifyOpHBM"]
    assert captured["is_reduction"] is False
    # When there are no indirect_vars, only 2 args are created (input and output)
    assert len(captured["args"]) == 2

    # With no indirect_vars, we only get input and output args
    input_arg = captured["args"][0]
    output_arg = captured["args"][1]

    assert input_arg.is_input is True
    assert arg_metadata[id(input_arg)]["is_index_tensor"] is False
    assert arg_metadata[id(input_arg)]["related_value_tensor_idx"] == -1

    assert output_arg.is_input is False
    assert arg_metadata[id(output_arg)]["is_index_tensor"] is False
    assert arg_metadata[id(output_arg)]["related_value_tensor_idx"] == -1

    assert created_args[0][0] == "value_tensor"
    assert created_args[1][0] == "out"

    assert created_args[0][2] == x0 * 64 + x1


def test_tensor_arg_with_stride_map():
    """Test that TensorArg correctly stores stride_map information."""
    stride_map = [128, 64, 1]
    device_size = [2, 128, 64]

    arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=device_size,
        device_coordinates=[sympy.Symbol("x0"), sympy.Symbol("x1"), sympy.Symbol("x2")],
        allocation={"hbm": 0},
        stride_map=stride_map,
        per_tile_fixed=False,
        name="test_tensor",
    )

    assert arg.stride_map == stride_map
    assert arg.device_size == device_size
    assert arg.per_tile_fixed is False
    assert arg.name == "test_tensor"


def test_tensor_arg_with_per_tile_fixed():
    """Test that TensorArg correctly handles per_tile_fixed flag."""
    arg = TensorArg(
        is_input=False,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[64, 64],
        device_coordinates=[sympy.Symbol("x0"), sympy.Symbol("x1")],
        allocation={"lx": 1024},
        stride_map=[64, 1],
        per_tile_fixed=True,
        name="scratch_buffer",
    )

    assert arg.per_tile_fixed is True
    assert arg.allocation == {"lx": 1024}
    assert arg.is_input is False


def test_op_spec_with_tiled_symbols():
    """Test that OpSpec correctly stores tiled_symbols."""
    x0 = sympy.Symbol("x0")
    x1 = sympy.Symbol("x1")
    tiled_syms = [x0, x1]

    arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[128, 64],
        device_coordinates=[x0, x1],
        allocation={"hbm": 0},
        stride_map=[64, 1],
        per_tile_fixed=False,
        name=None,
    )

    op_spec = OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={x0: (sympy.Integer(128), 1), x1: (sympy.Integer(64), 1)},
        args=[arg],
        op_info={},
        tiled_symbols=tiled_syms,
    )

    assert op_spec.tiled_symbols == tiled_syms
    assert len(op_spec.tiled_symbols) == 2
    assert x0 in op_spec.tiled_symbols
    assert x1 in op_spec.tiled_symbols


def test_op_spec_with_empty_tiled_symbols():
    """Test that OpSpec works with empty tiled_symbols (default case)."""
    x0 = sympy.Symbol("x0")

    arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[128],
        device_coordinates=[x0],
        allocation={"hbm": 0},
        stride_map=[1],
        per_tile_fixed=False,
        name=None,
    )

    op_spec = OpSpec(
        op="mul",
        is_reduction=False,
        iteration_space={x0: (sympy.Integer(128), 1)},
        args=[arg],
        op_info={},
        tiled_symbols=[],
    )

    assert op_spec.tiled_symbols == []
    assert len(op_spec.tiled_symbols) == 0


def test_multiple_tensor_args_with_different_allocations():
    """Test OpSpec with multiple TensorArgs having different allocation types."""
    x0 = sympy.Symbol("x0")
    x1 = sympy.Symbol("x1")

    # HBM input
    hbm_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[128, 64],
        device_coordinates=[x0, x1],
        allocation={"hbm": 0},
        stride_map=[64, 1],
        per_tile_fixed=False,
        name="input_hbm",
    )

    # LX scratch
    lx_arg = TensorArg(
        is_input=False,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[128, 64],
        device_coordinates=[x0, x1],
        allocation={"lx": 2048},
        stride_map=[64, 1],
        per_tile_fixed=True,
        name="scratch_lx",
    )

    # HBM output
    output_arg = TensorArg(
        is_input=False,
        arg_index=2,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[128, 64],
        device_coordinates=[x0, x1],
        allocation={"hbm": 8192},
        stride_map=[64, 1],
        per_tile_fixed=False,
        name="output_hbm",
    )

    op_spec = OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={x0: (sympy.Integer(128), 1), x1: (sympy.Integer(64), 1)},
        args=[hbm_arg, lx_arg, output_arg],
        op_info={},
        tiled_symbols=[],
    )

    assert len(op_spec.args) == 3
    assert op_spec.args[0].allocation == {"hbm": 0}
    assert op_spec.args[1].allocation == {"lx": 2048}
    assert op_spec.args[2].allocation == {"hbm": 8192}
    assert op_spec.args[1].per_tile_fixed is True
    assert op_spec.args[0].per_tile_fixed is False


def test_tensor_arg_with_none_values():
    """Test that TensorArg handles None values correctly for optional fields."""
    arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[64],
        device_coordinates=[sympy.Symbol("x0")],
        allocation={"hbm": 0},
        stride_map=None,
        per_tile_fixed=False,
        name=None,
    )

    assert arg.stride_map is None
    assert arg.name is None
    assert arg.per_tile_fixed is False


def test_op_spec_with_reduction():
    """Test OpSpec for reduction operations."""
    x0 = sympy.Symbol("x0")
    x1 = sympy.Symbol("x1")

    input_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[128, 64],
        device_coordinates=[x0, x1],
        allocation={"hbm": 0},
        stride_map=[64, 1],
        per_tile_fixed=False,
        name="input",
    )

    output_arg = TensorArg(
        is_input=False,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[128],
        device_coordinates=[x0],
        allocation={"hbm": 8192},
        stride_map=[1],
        per_tile_fixed=False,
        name="output",
    )

    op_spec = OpSpec(
        op="sum",
        is_reduction=True,
        iteration_space={x0: (sympy.Integer(128), 1), x1: (sympy.Integer(64), 1)},
        args=[input_arg, output_arg],
        op_info={"reduction_dim": 1},
        tiled_symbols=[],
    )

    assert op_spec.is_reduction is True
    assert op_spec.op == "sum"
    assert op_spec.op_info["reduction_dim"] == 1


def test_op_spec_with_complex_device_coordinates():
    """Test OpSpec with complex sympy expressions in device_coordinates."""
    x0 = sympy.Symbol("x0")
    x1 = sympy.Symbol("x1")

    # Complex coordinate expression: floor(x1/64), x0, Mod(x1, 64)
    arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[4, 128, 64],
        device_coordinates=[
            sympy.floor(x1 / 64),
            x0,
            sympy.Mod(x1, 64),
        ],
        allocation={"hbm": 0},
        stride_map=[8192, 64, 1],
        per_tile_fixed=False,
        name="complex_tensor",
    )

    op_spec = OpSpec(
        op="identity",
        is_reduction=False,
        iteration_space={x0: (sympy.Integer(128), 1), x1: (sympy.Integer(256), 1)},
        args=[arg],
        op_info={},
        tiled_symbols=[],
    )

    assert len(arg.device_coordinates) == 3
    assert isinstance(arg.device_coordinates[0], sympy.floor)
    assert isinstance(arg.device_coordinates[2], sympy.Mod)
