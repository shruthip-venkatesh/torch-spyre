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

"""Tests for detecting indirect-access patterns (gather and scatter).

Three test layers:

  1. IndirectAccess sympy atom - Basic building block (no device/compile needed)
  2. Detection on synthetic OpSpecs - Distinguish gather vs scatter patterns
  3. Compile-time detection - Test that indirect_index_dep_names correctly
     identifies index buffers during layout propagation (before work-division)
"""

import contextlib
import os
import sys
from unittest.mock import patch

import sympy
import torch

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    IndirectAccessTestCase,
    arg_has_indirect_access,
    capture_op_specs,
    coord_atoms,
    flatten_entries,
    indirect_access_target_names,
    op_spec_has_indirect_access,
    op_spec_has_indirect_input,
    op_spec_has_indirect_output,
)

import torch_spyre._inductor.propagate_layouts as _pl  # noqa: E402
from torch_spyre._C import DataFormats  # noqa: E402
from torch_spyre._inductor.op_spec import (  # noqa: E402
    IndirectAccess,
    OpSpec,
    TensorArg,
)
from torch_spyre._inductor import config  # noqa: E402


def _tensor_arg(is_input, coords, name=None):
    return TensorArg(
        is_input=is_input,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[1, 128, 64],
        device_coordinates=coords,
        allocation={},
        stride_map=[1, 1, 1],
        name=name,
    )


@contextlib.contextmanager
def spy_indirect_detection():
    """Capture all calls to indirect_index_dep_names during compilation.

    This patches the function in propagate_layouts to record what index buffers
    are detected during layout propagation (before work-division).
    """
    real = _pl.indirect_index_dep_names
    seen: list[set] = []

    def wrapper(op):
        names = real(op)
        seen.append(set(names))
        return names

    with patch.object(_pl, "indirect_index_dep_names", wrapper):
        yield seen


# ===========================================================================
# Layer 1: IndirectAccess atom tests (pure sympy, no device)
# ===========================================================================
class TestIndirectAccessAtom(IndirectAccessTestCase):
    def test_atom_is_opaque_and_carries_name(self):
        """Test that IndirectAccess atom stores and returns the buffer name."""
        ia = IndirectAccess(sympy.Symbol("arg1_1"))
        self.assertIsInstance(ia, IndirectAccess)
        self.assertEqual(str(ia.args[0]), "arg1_1")

    def test_atom_found_inside_compound_expression(self):
        """Test that IndirectAccess can be found within complex sympy expressions."""
        c1 = sympy.Symbol("c1")
        ia = IndirectAccess(sympy.Symbol("arg1_1"))
        expr = sympy.floor(c1 / 64) + ia * 128
        found = expr.atoms(IndirectAccess)
        self.assertEqual(len(found), 1)
        self.assertIn(ia, found)

    def test_direct_expression_has_no_indirect_atom(self):
        """Test that regular expressions without IndirectAccess return empty set."""
        c0, c1 = sympy.symbols("c0 c1")
        expr = sympy.floor(c1 / 64) + c0
        self.assertEqual(expr.atoms(IndirectAccess), set())

    def test_xreplace_substitutes_indirect_symbol(self):
        """Test that we can replace a symbol with IndirectAccess in an expression.

        This mimics how compute_coordinates works.
        """
        c1, tmp0 = sympy.symbols("c1 tmp0")
        coord = sympy.floor(c1 / 64) + tmp0 * 128
        out = coord.xreplace({tmp0: IndirectAccess(sympy.Symbol("arg1_1"))})
        self.assertNotIn(tmp0, out.free_symbols)
        self.assertEqual(len(out.atoms(IndirectAccess)), 1)

    def test_distinct_names_are_distinct_atoms(self):
        """IndirectAccess atoms with different names are distinct atoms."""
        a = IndirectAccess(sympy.Symbol("buf_a"))
        b = IndirectAccess(sympy.Symbol("buf_b"))
        self.assertNotEqual(a, b)
        self.assertEqual(len((a + b).atoms(IndirectAccess)), 2)

    def test_indirect_access_with_constant_offset(self):
        ia = IndirectAccess(sympy.Symbol("arg1_1"))
        self.assertEqual(len((ia + 5).atoms(IndirectAccess)), 1)

    def test_indirect_access_inside_mod(self):
        c1 = sympy.Symbol("c1")
        ia = IndirectAccess(sympy.Symbol("arg1_1"))
        expr = sympy.Mod(ia + c1, 64)
        self.assertEqual(len(expr.atoms(IndirectAccess)), 1)

    def test_indirect_access_equality_same_name(self):
        self.assertEqual(
            IndirectAccess(sympy.Symbol("b")), IndirectAccess(sympy.Symbol("b"))
        )

    def test_indirect_access_usable_in_set(self):
        s = {IndirectAccess(sympy.Symbol("b")), IndirectAccess(sympy.Symbol("b"))}
        self.assertEqual(len(s), 1)

    def test_two_indirect_symbols_substituted(self):
        t0, t1, c0 = sympy.symbols("t0 t1 c0")
        expr = t0 * 256 + t1 + c0
        out = expr.xreplace(
            {
                t0: IndirectAccess(sympy.Symbol("a")),
                t1: IndirectAccess(sympy.Symbol("b")),
            }
        )
        self.assertEqual(len(out.atoms(IndirectAccess)), 2)
        self.assertNotIn(t0, out.free_symbols)
        self.assertNotIn(t1, out.free_symbols)


# ===========================================================================
# Layer 2: Detection helpers on synthetic OpSpecs (no compile)
# ===========================================================================
class TestDetectionHelpers(IndirectAccessTestCase):
    def _op(self, args):
        """Helper to create a synthetic OpSpec for testing."""
        return OpSpec(
            op="identity",
            is_reduction=False,
            iteration_space={},
            args=args,
            op_info={},
        )

    def test_gather_signature_detected(self):
        """Gather pattern (IndirectAccess on an input) is identified."""
        ia = IndirectAccess(sympy.Symbol("idx_buf"))
        gathered_in = _tensor_arg(True, [ia, sympy.Symbol("c0"), sympy.Symbol("c1")])
        out = _tensor_arg(False, list(sympy.symbols("c0 c1 c2")))
        op = self._op([gathered_in, out])

        self.assertTrue(arg_has_indirect_access(gathered_in))
        self.assertFalse(arg_has_indirect_access(out))
        self.assertTrue(op_spec_has_indirect_access(op))
        self.assertTrue(op_spec_has_indirect_input(op))
        self.assertFalse(op_spec_has_indirect_output(op))

    def test_scatter_signature_detected(self):
        """Scatter pattern (IndirectAccess on an output) is identified."""
        ia = IndirectAccess(sympy.Symbol("idx_buf"))
        src_in = _tensor_arg(True, list(sympy.symbols("c0 c1 c2")))
        scattered_out = _tensor_arg(False, [ia, sympy.Symbol("c0"), sympy.Symbol("c1")])
        op = self._op([src_in, scattered_out])

        self.assertTrue(op_spec_has_indirect_access(op))
        self.assertFalse(op_spec_has_indirect_input(op))
        self.assertTrue(op_spec_has_indirect_output(op))

    def test_direct_op_no_false_positive(self):
        """Direct operations don't falsely trigger indirect detection."""
        a = _tensor_arg(True, list(sympy.symbols("c0 c1 c2")))
        b = _tensor_arg(False, list(sympy.symbols("c0 c1 c2")))
        op = self._op([a, b])
        self.assertFalse(op_spec_has_indirect_access(op))
        self.assertFalse(op_spec_has_indirect_input(op))
        self.assertFalse(op_spec_has_indirect_output(op))

    def test_coord_atoms_extracts_target_name(self):
        """Test that coord_atoms correctly extracts buffer names from IndirectAccess."""
        ia = IndirectAccess(sympy.Symbol("idx_buf"))
        arg = _tensor_arg(True, [ia, sympy.Symbol("c0")])
        atoms = coord_atoms(arg, IndirectAccess)
        self.assertEqual({str(a.args[0]) for a in atoms}, {"idx_buf"})

    def test_coord_atoms_tolerates_non_sympy_coordinate(self):
        """Test that coord_atoms handles non-sympy coordinates (like bare 0).

        Scalar/broadcast coordinates don't have .atoms() but shouldn't cause errors.
        """
        ia = IndirectAccess(sympy.Symbol("idx_buf"))
        arg = _tensor_arg(True, [0, ia, sympy.Symbol("c0")])
        self.assertTrue(arg_has_indirect_access(arg))

    def test_multiple_index_tensors_all_detected(self):
        """Test that we detect all index buffers in multi-dimensional indexing.

        A 2-D advanced index uses two index buffers - both should be found.
        """
        ia0 = IndirectAccess(sympy.Symbol("idx0"))
        ia1 = IndirectAccess(sympy.Symbol("idx1"))
        arg = _tensor_arg(True, [ia0, ia1, sympy.Symbol("c0")])
        names = {str(a.args[0]) for a in coord_atoms(arg, IndirectAccess)}
        self.assertEqual(names, {"idx0", "idx1"})

    def test_empty_args_has_no_indirect(self):
        op = self._op([])
        self.assertFalse(op_spec_has_indirect_access(op))
        self.assertFalse(op_spec_has_indirect_input(op))
        self.assertFalse(op_spec_has_indirect_output(op))

    def test_multiple_inputs_single_indirect(self):
        ia = IndirectAccess(sympy.Symbol("idx"))
        plain = _tensor_arg(True, list(sympy.symbols("c0 c1 c2")))
        gathered = _tensor_arg(True, [ia, sympy.Symbol("c0"), sympy.Symbol("c1")])
        out = _tensor_arg(False, list(sympy.symbols("c0 c1 c2")))
        op = self._op([plain, gathered, out])
        self.assertTrue(op_spec_has_indirect_input(op))
        self.assertFalse(op_spec_has_indirect_output(op))

    def test_indirect_access_target_names_extracted(self):
        ia = IndirectAccess(sympy.Symbol("idx_buf"))
        gathered = _tensor_arg(True, [ia, sympy.Symbol("c0")])
        out = _tensor_arg(False, list(sympy.symbols("c0 c1")))
        names = indirect_access_target_names([self._op([gathered, out])])
        self.assertEqual(names, {"idx_buf"})

    def test_zero_coordinate_arg_no_false_positive(self):
        arg = _tensor_arg(True, [0, 0, 0])
        self.assertFalse(arg_has_indirect_access(arg))

    def test_gather_and_scatter_signatures_mutually_exclusive(self):
        ia = IndirectAccess(sympy.Symbol("idx"))
        gather = self._op(
            [
                _tensor_arg(True, [ia, sympy.Symbol("c0")]),
                _tensor_arg(False, list(sympy.symbols("c0 c1"))),
            ]
        )
        scatter = self._op(
            [
                _tensor_arg(True, list(sympy.symbols("c0 c1"))),
                _tensor_arg(False, [ia, sympy.Symbol("c0")]),
            ]
        )
        self.assertTrue(op_spec_has_indirect_input(gather))
        self.assertFalse(op_spec_has_indirect_output(gather))
        self.assertTrue(op_spec_has_indirect_output(scatter))
        self.assertFalse(op_spec_has_indirect_input(scatter))


# ===========================================================================
# Layer 3: Compile-time detection tests
# ===========================================================================
class TestCompileTimeDetection(IndirectAccessTestCase):
    def test_gather_index_is_detected(self):
        """Test that gather index buffers are detected during layout propagation.

        We capture sdsc to avoid hitting the backend (which doesn't support
        indirect gather yet).
        """
        M, N, P, Q = 128, 256, 3, 192
        x = torch.rand(M, N, dtype=torch.float16)
        i = torch.randint(0, M, (P, Q), dtype=torch.int32)

        def kernel(x, i):
            return x[i].exp()

        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with capture_op_specs(), spy_indirect_detection() as seen:
            torch.compile(kernel)(x_dev, i_dev)

        self.assertTrue(
            any(names for names in seen),
            "gather index buffer was never detected during layout propagation",
        )

    @config.patch({"sencores": 1})
    def test_scatter_index_not_flagged_by_layout_detection(self):
        """``indirect_index_dep_names`` flags gather *loads* but returns empty
        for scatter *stores*.

        A property of that (reads-only) helper, unchanged by the scatter fixes:
        a scatter's output is recognized later in superdsc (via is_output_tensor),
        not by this layout-propagation pass. Captured so we stop before the
        backend.
        """
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

        with capture_op_specs(), spy_indirect_detection() as seen:
            torch.compile(kernel)(y_dev, src_dev, index_dev)

        self.assertFalse(
            any(names for names in seen),
            "scatter store index unexpectedly flagged by layout detection",
        )

    def test_direct_op_not_detected_as_indirect(self):
        """Test that direct operations don't trigger indirect index detection."""
        M, N = 128, 256
        x = torch.rand(M, N, dtype=torch.float16)

        def kernel(x):
            return x.exp()

        x_dev = x.to("spyre")

        with capture_op_specs(), spy_indirect_detection() as seen:
            torch.compile(kernel)(x_dev)

        self.assertFalse(
            any(names for names in seen),
            "direct op wrongly flagged as having an indirect index",
        )


# ===========================================================================
# Compile-level control tests
# ===========================================================================
class TestDirectVsIndirect(IndirectAccessTestCase):
    def test_gather_produces_indirect_op_spec(self):
        """Test that gather operations generate OpSpecs with IndirectAccess."""
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
        self.assertTrue(any(op_spec_has_indirect_access(s) for s in op_specs))

    def test_direct_access_has_no_indirect(self):
        """Test that direct operations don't trigger indirect access detection.

        A plain x.exp() should have no IndirectAccess atoms (no false positives).
        """
        M, N = 128, 256
        x = torch.rand(M, N, dtype=torch.float16)

        def kernel(x):
            return x.exp()

        x_dev = x.to("spyre")

        with capture_op_specs() as captured:
            torch.compile(kernel)(x_dev)

        op_specs = self.assert_reaches_op_spec(captured)
        self.assertIn("exp", [s.op for s in op_specs])
        self.assertFalse(any(op_spec_has_indirect_access(s) for s in op_specs))
        self.assertFalse(any(a.name is not None for s in op_specs for a in s.args))

    def test_unsupported_unary_after_gather_falls_back_to_cpu(self):
        """Test that unsupported ops after gather fall back to CPU.

        x[i].sin() - sin isn't supported on Spyre, so it runs on CPU.
        The gather still compiles to Spyre though.
        """
        M, N, P, Q = 128, 256, 3, 192
        x = torch.rand(M, N, dtype=torch.float16)
        i = torch.randint(0, M, (P, Q), dtype=torch.int32)

        def kernel(x, i):
            return x[i].sin()

        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with capture_op_specs() as captured:
            torch.compile(kernel)(x_dev, i_dev)

        op_specs = self.assert_reaches_op_spec(captured)
        self.assertTrue(
            any(op_spec_has_indirect_access(s) for s in op_specs),
            "gather should still produce an IndirectAccess op spec",
        )
        self.assertFalse(
            any(getattr(e, "op", None) == "sin" for e in flatten_entries(captured)),
            "unsupported 'sin' must not appear as a Spyre op (it falls back to CPU)",
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
