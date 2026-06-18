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

"""Tests for OpSpec structure in indirect access operations.

These tests verify the OpSpec and TensorArg objects the compiler generates:
- Operation names
- Argument ordering
- Data types and sizes
- Where IndirectAccess coordinates appear

Current status:
- Gather (indirect load): Works, generates OpSpecs
- Scatter (indirect store): Works, generates OpSpecs
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    IndirectAccessTestCase,
    capture_op_specs,
    indirect_access_target_names,
    op_spec_has_indirect_input,
    op_spec_has_indirect_output,
)

from torch_spyre._C import DataFormats  # noqa: E402
from torch_spyre._inductor.constants import IDENTITY_OP, RESTICKIFY_OP  # noqa: E402


class TestGatherOpSpec(IndirectAccessTestCase):
    """Tests for gather (indirect load) OpSpec structure."""

    def test_gather_exp(self):
        """Test x[i].exp() generates correct OpSpec.

        Should produce 'exp' op with IndirectAccess on input, and the named
        index arg should match the IndirectAccess target buffer.
        """
        M, N, P, Q = 128, 256, 3, 192
        x = torch.rand(M, N, dtype=torch.float16)
        i = torch.randint(0, M, (P, Q), dtype=torch.int32)

        def kernel(x, i):
            return x[i].exp()

        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with capture_op_specs() as captured:
            torch.compile(kernel)(x_dev, i_dev)

        op_specs = self.assert_gather_op_spec(captured)
        self.assertIn("exp", [s.op for s in op_specs])

        # The named index arg is one of the IndirectAccess targets.
        named_inputs = [
            a for s in op_specs for a in s.args if a.is_input and a.name is not None
        ]
        self.assertTrue(named_inputs, "no named index arg found")
        targets = indirect_access_target_names(op_specs)
        self.assertTrue(
            any(a.name in targets for a in named_inputs),
            f"named index args {[a.name for a in named_inputs]} not in "
            f"IndirectAccess targets {targets}",
        )

    def test_gather_tanh(self):
        """Test that gather pattern works with other unaries like tanh."""
        M, N, P = 64, 128, 32
        x = torch.rand(M, N, dtype=torch.float16)
        i = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(x, i):
            return x[i].tanh()

        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P})

        with capture_op_specs() as captured:
            torch.compile(kernel)(x_dev, i_dev)

        op_specs = self.assert_gather_op_spec(captured)
        self.assertIn("tanh", [s.op for s in op_specs])

    def test_gather_bare_index_is_copy_with_arg_ordering(self):
        """Test x[i] without unary generates identity/restickify op.

        Arg ordering should be: index(es) first, then value, then output.
        """
        M, N, P, Q = 128, 256, 3, 192
        x = torch.rand(M, N, dtype=torch.float16)
        i = torch.randint(0, M, (P, Q), dtype=torch.int32)

        def kernel(x, i):
            return x[i]

        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with capture_op_specs() as captured:
            torch.compile(kernel)(x_dev, i_dev)

        op_specs = self.assert_reaches_op_spec(captured)
        gather_specs = [
            s
            for s in op_specs
            if s.op in (IDENTITY_OP, RESTICKIFY_OP) and op_spec_has_indirect_input(s)
        ]
        self.assertTrue(
            gather_specs,
            f"expected an indirect copy op ({IDENTITY_OP}/{RESTICKIFY_OP}); "
            f"got ops={[s.op for s in op_specs]}",
        )
        spec = gather_specs[0]
        self.assertIsNotNone(spec.args[0].name, "first arg should be the named index")
        self.assertTrue(spec.args[0].is_input, "index arg should be an input")
        self.assertFalse(spec.args[-1].is_input, "last arg should be the output")
        self.assertEqual(
            len([a for a in spec.args if not a.is_input]), 1, "one output expected"
        )

    def test_gather_indirect_is_on_input_not_output(self):
        """Test gather signature: IndirectAccess on input, not output.

        Also verifies OpSpecs are well-formed (one output, at least one input,
        non-empty iteration space).
        """
        M, N, P, Q = 128, 256, 3, 192
        x = torch.rand(M, N, dtype=torch.float16)
        i = torch.randint(0, M, (P, Q), dtype=torch.int32)

        def kernel(x, i):
            return x[i].exp()

        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with capture_op_specs() as captured:
            torch.compile(kernel)(x_dev, i_dev)

        op_specs = self.assert_reaches_op_spec(captured)
        for s in op_specs:
            outputs = [a for a in s.args if not a.is_input]
            inputs = [a for a in s.args if a.is_input]
            self.assertEqual(len(outputs), 1, f"op {s.op} should have one output")
            self.assertGreaterEqual(len(inputs), 1, f"op {s.op} needs an input")
            self.assertTrue(s.iteration_space, f"op {s.op} has empty iteration space")

        self.assertTrue(
            any(op_spec_has_indirect_input(s) for s in op_specs),
            "gather should carry IndirectAccess on an input arg",
        )
        self.assertFalse(
            any(op_spec_has_indirect_output(s) for s in op_specs),
            "gather should not carry IndirectAccess on an output arg",
        )

    def test_gather_tensor_arg_metadata(self):
        """Test that TensorArgs have correct metadata.

        Should have: DataFormats dtype, non-empty device_size, populated stride_map.
        """
        M, N, P, Q = 128, 256, 3, 192
        x = torch.rand(M, N, dtype=torch.float16)
        i = torch.randint(0, M, (P, Q), dtype=torch.int32)

        def kernel(x, i):
            return x[i].exp()

        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with capture_op_specs() as captured:
            torch.compile(kernel)(x_dev, i_dev)

        op_specs = self.assert_gather_op_spec(captured)
        for s in op_specs:
            for a in s.args:
                self.assertIsInstance(a.device_dtype, DataFormats)
                self.assertTrue(a.device_size, "device_size should be non-empty")
                self.assertIsNotNone(a.stride_map, "stride_map should be populated")
                # Coordinates are one expr per device dimension.
                self.assertEqual(len(a.device_coordinates), len(a.device_size))

    def test_gather_supported_unaries_matrix(self):
        """Test gather works with all supported unary operations.

        Each should produce a gather OpSpec (IndirectAccess on input).
        We check the signature, not the exact op name (some ops decompose).
        """
        M, N, P = 128, 256, 32
        unaries = {
            "abs": torch.abs,
            "neg": torch.neg,
            "exp": torch.exp,
            "tanh": torch.tanh,
            "sqrt": torch.sqrt,
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
        }
        for label, fn in unaries.items():
            with self.subTest(unary=label):
                torch._dynamo.reset()
                x = torch.rand(M, N, dtype=torch.float16)
                idx = torch.randint(0, M, (P,), dtype=torch.int32)

                def kernel(x, idx, _fn=fn):
                    return _fn(x[idx])

                x_dev, idx_dev = x.to("spyre"), idx.to("spyre")
                self.name_dims(x_dev, {"M": M, "N": N})
                self.name_dims(idx_dev, {"P": P})

                with capture_op_specs() as captured:
                    torch.compile(kernel)(x_dev, idx_dev)

                op_specs = self.assert_reaches_op_spec(captured)
                self.assertTrue(
                    any(op_spec_has_indirect_input(s) for s in op_specs),
                    f"{label}: gather signature lost",
                )

    def test_gather_chained_unaries(self):
        """Test gather with multiple chained unaries: x[i].exp().tanh()

        Both unaries should stay on Spyre and gather should keep IndirectAccess.
        """
        M, N, P = 128, 256, 32
        x = torch.rand(M, N, dtype=torch.float16)
        i = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(x, i):
            return x[i].exp().tanh()

        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P})

        with capture_op_specs() as captured:
            torch.compile(kernel)(x_dev, i_dev)

        op_specs = self.assert_reaches_op_spec(captured)
        ops = [s.op for s in op_specs]
        self.assertIn("exp", ops)
        self.assertIn("tanh", ops)
        self.assertTrue(any(op_spec_has_indirect_input(s) for s in op_specs))

    # -- additional gather structure / coverage --------------------------
    def _compile_gather(self, fn, P=32, two_d=False):
        M, N, Q = 128, 256, 192
        x = torch.rand(M, N, dtype=torch.float16)
        shape = (P, Q) if two_d else (P,)
        i = torch.randint(0, M, shape, dtype=torch.int32)
        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q} if two_d else {"P": P})
        with capture_op_specs() as captured:
            torch.compile(fn)(x_dev, i_dev)
        return self.assert_gather_op_spec(captured)

    def test_gather_exp_iteration_space_well_formed(self):
        """Test that gather OpSpec has a well-formed iteration space."""
        op_specs = self._compile_gather(lambda x, i: x[i].exp())
        for s in op_specs:
            self.assertIsInstance(s.iteration_space, dict)
            self.assertTrue(s.iteration_space)
            for _range, work in s.iteration_space.values():
                self.assertGreaterEqual(int(work), 1)

    def test_gather_has_value_and_index_inputs(self):
        """Test that gather OpSpec has both value and index input args."""
        op_specs = self._compile_gather(lambda x, i: x[i].exp())
        spec = next(s for s in op_specs if op_spec_has_indirect_input(s))
        named = [a for a in spec.args if a.is_input and a.name is not None]
        unnamed_in = [a for a in spec.args if a.is_input and a.name is None]
        self.assertTrue(named, "expected a named index input arg")
        self.assertTrue(unnamed_in, "expected a value (non-index) input arg")

    def test_gather_index_arg_name_is_nonempty_str(self):
        """Test that gather index arg has a non-empty string name."""
        op_specs = self._compile_gather(lambda x, i: x[i].exp())
        for s in op_specs:
            for a in s.args:
                if a.name is not None:
                    self.assertIsInstance(a.name, str)
                    self.assertTrue(a.name)

    def test_gather_2d_index_op_spec(self):
        """Test that gather with 2-D index generates correct OpSpec."""
        op_specs = self._compile_gather(lambda x, i: x[i].exp(), P=3, two_d=True)
        self.assertIn("exp", [s.op for s in op_specs])

    def test_gather_abs_op_name(self):
        """Test that x[i].abs() generates 'abs' op name."""
        op_specs = self._compile_gather(lambda x, i: x[i].abs())
        self.assertIn("abs", [s.op for s in op_specs])

    def test_gather_neg_keeps_indirect_input(self):
        """Test that x[i].neg() maintains IndirectAccess on input."""
        self._compile_gather(lambda x, i: x[i].neg())

    def test_gather_sqrt_keeps_indirect_input(self):
        """Test that x[i].sqrt() maintains IndirectAccess on input."""
        self._compile_gather(lambda x, i: x[i].sqrt())

    def test_gather_sigmoid_keeps_indirect_input(self):
        """Test that x[i].sigmoid() maintains IndirectAccess on input."""
        self._compile_gather(lambda x, i: x[i].sigmoid())

    def test_gather_relu_keeps_indirect_input(self):
        """Test that relu(x[i]) maintains IndirectAccess on input."""
        self._compile_gather(lambda x, i: torch.relu(x[i]))


class TestScatterOpSpec(IndirectAccessTestCase):
    """Tests for scatter (indirect store) OpSpec structure.

    Scatter now compiles to a real op spec (work-division + OUTPUT-layout fixes),
    so each test asserts the actual scatter signature via
    ``assert_scatter_op_spec``: the op spec carries ``IndirectAccess`` on an
    *output* arg.  They stop at the sdsc handoff (capture_op_specs patches sdsc),
    so they never reach the deeptools backend.
    """

    def _scatter_op_specs(self, kernel, *dev_args):
        with capture_op_specs() as captured:
            torch.compile(kernel)(*dev_args)
        return self.assert_scatter_op_spec(captured)

    def test_index_put_with_exp(self):
        """y[i] = src.exp() -> op spec with IndirectAccess on the output arg."""
        M, N, P = 128, 256, 3
        y = torch.zeros(M, N, dtype=torch.float16)
        src = torch.rand(P, N, dtype=torch.float16)
        i = torch.randint(0, M, (P,), dtype=torch.int32)

        def kernel(y, src, i):
            y[i] = src.exp()
            return y

        y_dev, src_dev, i_dev = y.to("spyre"), src.to("spyre"), i.to("spyre")
        self.name_dims(y_dev, {"M": M, "N": N})
        self.name_dims(src_dev, {"P": P, "N": N})
        self.name_dims(i_dev, {"P": P})
        self._scatter_op_specs(kernel, y_dev, src_dev, i_dev)

    def test_scatter_method_with_exp(self):
        """y.scatter_(0, index, src.exp()) -> op spec, output IndirectAccess."""
        M, N, P = 128, 256, 3
        y = torch.zeros(M, N, dtype=torch.float16)
        src = torch.rand(P, N, dtype=torch.float16)
        index = torch.randint(0, M, (P, N), dtype=torch.int32)

        def kernel(y, src, index):
            return y.scatter_(0, index, src.exp())

        y_dev, src_dev, index_dev = y.to("spyre"), src.to("spyre"), index.to("spyre")
        self.name_dims(y_dev, {"M": M, "N": N})
        self.name_dims(src_dev, {"P": P, "N": N})
        self.name_dims(index_dev, {"P": P, "N": N})
        self._scatter_op_specs(kernel, y_dev, src_dev, index_dev)

    def test_scatter_method_without_unary(self):
        """y.scatter_(0, index, src) (no fused unary) -> output IndirectAccess."""
        M, N, P = 128, 256, 3
        y = torch.zeros(M, N, dtype=torch.float16)
        src = torch.rand(P, N, dtype=torch.float16)
        index = torch.randint(0, M, (P, N), dtype=torch.int32)

        def kernel(y, src, index):
            return y.scatter_(0, index, src)

        y_dev, src_dev, index_dev = y.to("spyre"), src.to("spyre"), index.to("spyre")
        self.name_dims(y_dev, {"M": M, "N": N})
        self.name_dims(src_dev, {"P": P, "N": N})
        self.name_dims(index_dev, {"P": P, "N": N})
        self._scatter_op_specs(kernel, y_dev, src_dev, index_dev)

    def test_scatter_add(self):
        """y.scatter_add_(0, index, src) (accumulating) -> output IndirectAccess."""
        M, N, P = 128, 256, 3
        y = torch.zeros(M, N, dtype=torch.float16)
        src = torch.rand(P, N, dtype=torch.float16)
        index = torch.randint(0, M, (P, N), dtype=torch.int32)

        def kernel(y, src, index):
            return y.scatter_add_(0, index, src)

        y_dev, src_dev, index_dev = y.to("spyre"), src.to("spyre"), index.to("spyre")
        self.name_dims(y_dev, {"M": M, "N": N})
        self.name_dims(src_dev, {"P": P, "N": N})
        self.name_dims(index_dev, {"P": P, "N": N})
        self._scatter_op_specs(kernel, y_dev, src_dev, index_dev)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
