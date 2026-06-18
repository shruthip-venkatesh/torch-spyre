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

"""Shared test utilities for indirect-access (gather/scatter) tests.

This file is NOT named `test_*` so pytest won't run it directly. The five test
files import helpers from here:

  * test_indirect_access_op_spec.py   -- OpSpec structure tests
  * test_indirect_access_sdsc.py      -- SDSC boundary tests
  * test_indirect_access_detection.py -- Pattern detection tests
  * test_torch_gather_ops.py          -- PyTorch gather operations
  * test_torch_scatter_ops.py         -- PyTorch scatter operations

How it works: We intercept the compiler before it reaches SuperDSC, capture
what OpSpecs were generated, then return a no-op runner. This lets us test
what the compiler produces without needing actual hardware or numeric results.
"""

import contextlib
from unittest.mock import patch

import torch
from torch._inductor.test_case import TestCase as InductorTestCase

import torch_spyre._inductor.propagate_named_dims as _pnd
from torch_spyre._inductor.op_spec import (
    IndirectAccess,
    LoopSpec,
    OpSpec,
    TensorArg,
    UnimplementedOp,
)
from torch_spyre.execution.async_compile import SpyreAsyncCompile

declare_tensor_dim = _pnd.declare_tensor_dim
name_tensor_dims = _pnd.name_tensor_dims


# ---------------------------------------------------------------------------
# Capture machinery
# ---------------------------------------------------------------------------
class _NoopRunner:
    """Dummy kernel runner that does nothing when called."""

    def run(self, *args, **kwargs):
        return None


@contextlib.contextmanager
def capture_op_specs():
    """Capture OpSpecs generated during compilation without running the backend.

    Returns a list where each entry is the OpSpec list for one kernel.
    This skips SuperDSC generation, so we can test what the compiler produces
    even for operations that would crash in the backend.
    """
    captured: list[list] = []

    def _spy(self, kernel_name, specs):  # noqa: ARG001
        captured.append(list(specs))
        return _NoopRunner()

    with patch.object(SpyreAsyncCompile, "sdsc", _spy):
        yield captured


@contextlib.contextmanager
def capture_sdsc_calls():
    """Like capture_op_specs but also captures kernel names.

    Returns (kernel_name, specs) tuples for tests that need the kernel name.
    """
    calls: list[tuple[str, list]] = []

    def _spy(self, kernel_name, specs):  # noqa: ARG001
        calls.append((kernel_name, list(specs)))
        return _NoopRunner()

    with patch.object(SpyreAsyncCompile, "sdsc", _spy):
        yield calls


# ---------------------------------------------------------------------------
# Spec-tree flattening
# ---------------------------------------------------------------------------
def flatten_op_specs(spec_lists) -> list[OpSpec]:
    """Extract all OpSpecs from nested LoopSpecs, ignoring UnimplementedOps."""
    out: list[OpSpec] = []

    def _rec(specs):
        for s in specs:
            if isinstance(s, LoopSpec):
                _rec(s.body)
            elif isinstance(s, OpSpec):
                out.append(s)

    for lst in spec_lists:
        _rec(lst)
    return out


def flatten_entries(spec_lists) -> list:
    """Extract all entries (OpSpecs AND UnimplementedOps) from nested LoopSpecs."""
    out: list = []

    def _rec(specs):
        for s in specs:
            if isinstance(s, LoopSpec):
                _rec(s.body)
            else:
                out.append(s)

    for lst in spec_lists:
        _rec(lst)
    return out


# ---------------------------------------------------------------------------
# IndirectAccess helpers
# ---------------------------------------------------------------------------
def coord_atoms(arg: TensorArg, kind):
    """Find all sympy atoms of a given type in a tensor arg's coordinates."""
    atoms = set()
    for coord in arg.device_coordinates:
        if hasattr(coord, "atoms"):
            atoms |= coord.atoms(kind)
    return atoms


def arg_has_indirect_access(arg: TensorArg) -> bool:
    return bool(coord_atoms(arg, IndirectAccess))


def op_spec_has_indirect_access(op_spec: OpSpec) -> bool:
    return any(arg_has_indirect_access(a) for a in op_spec.args)


def op_spec_has_indirect_input(op_spec: OpSpec) -> bool:
    """Check if OpSpec has IndirectAccess on an input (gather pattern)."""
    return any(a.is_input and arg_has_indirect_access(a) for a in op_spec.args)


def op_spec_has_indirect_output(op_spec: OpSpec) -> bool:
    """Check if OpSpec has IndirectAccess on an output (scatter pattern)."""
    return any((not a.is_input) and arg_has_indirect_access(a) for a in op_spec.args)


def indirect_access_target_names(op_specs) -> set:
    """Get all buffer names used in IndirectAccess operations."""
    return {
        str(sym)
        for s in op_specs
        for a in s.args
        for ia in coord_atoms(a, IndirectAccess)
        for sym in ia.args
    }


# ---------------------------------------------------------------------------
# Compilation outcome classification
# ---------------------------------------------------------------------------
# Possible outcomes when compiling an operation:
CRASHED = "crashed_before_op_spec"  # Crashed before generating OpSpec
GATHER_OP_SPEC = "op_spec_indirect_input"  # Generated gather OpSpec (indirect read)
SCATTER_OP_SPEC = "op_spec_indirect_output"  # Generated scatter OpSpec (indirect write)
DIRECT_OP_SPEC = "op_spec_no_indirect"  # Generated OpSpec without indirect access
UNIMPLEMENTED = "unimplemented_op"  # Hit UnimplementedOp
NO_SPYRE_OP = "no_spyre_op_spec"  # Fell back to CPU


def classify_compile(kernel, *dev_args):
    """Compile a kernel and classify what happened.

    Returns (label, detail) where:
    - label is one of the outcome constants above
    - detail is the exception if crashed, otherwise the OpSpec list

    Catches all exceptions to classify any compilation failure as CRASHED.
    """
    with capture_op_specs() as captured:
        try:
            torch.compile(kernel)(*dev_args)
        except Exception as exc:  # noqa: BLE001 - characterizing the failure mode
            return CRASHED, exc

    op_specs = flatten_op_specs(captured)
    entries = flatten_entries(captured)

    if any(op_spec_has_indirect_output(s) for s in op_specs):
        return SCATTER_OP_SPEC, op_specs
    if any(op_spec_has_indirect_input(s) for s in op_specs):
        return GATHER_OP_SPEC, op_specs
    if any(isinstance(e, UnimplementedOp) for e in entries):
        return UNIMPLEMENTED, entries
    if op_specs:
        return DIRECT_OP_SPEC, op_specs
    return NO_SPYRE_OP, op_specs


# ---------------------------------------------------------------------------
# Real SDSC bundle generation (the Python SuperDSC path, no backend)
# ---------------------------------------------------------------------------
def generate_sdsc_jsons(kernel, *dev_args) -> dict:
    """Compile ``kernel``, capture the op specs handed to sdsc, then run the
    REAL ``generate_bundle`` on them and return ``{relpath: parsed_json}`` for
    every ``sdsc_*.json`` produced.

    This exercises actual SDSC JSON generation (the Python SuperDSC path),
    independent of the deeptools backend compiler.  Each captured kernel is
    bundled into its own subdir so per-kernel ``sdsc_N.json`` files never
    collide.  Raises if compilation raises before reaching sdsc (e.g. scatter,
    which crashes in work division first)."""
    import glob
    import json
    import os
    import tempfile

    from torch_spyre._inductor.codegen.bundle import generate_bundle

    with capture_op_specs() as captured:
        torch.compile(kernel)(*dev_args)

    jsons: dict = {}
    out = tempfile.mkdtemp(prefix="sdsc_dump_")
    for ci, specs in enumerate(captured):
        sub = os.path.join(out, f"kernel{ci}")
        os.makedirs(sub, exist_ok=True)
        generate_bundle(f"kernel{ci}", sub, specs)
        for path in sorted(glob.glob(os.path.join(sub, "sdsc_*.json"))):
            with open(path) as f:
                jsons[os.path.relpath(path, out)] = json.load(f)
    return jsons


def iter_sdsc_op_bodies(jsons):
    """Yield ``(opfunc, body)`` for each compiled op across all sdsc json files.

    SDSC layout: ``{f"{idx}_{opfunc}": {... "dscs_": [{opfunc: body}] ...}}``,
    where ``body`` holds ``N_`` / ``scheduleTree_`` / ``labeledDs_`` /
    ``computeOp_``.
    """
    for top in jsons.values():
        for outer in top.values():
            for dsc in outer.get("dscs_", []):
                for opfunc, body in dsc.items():
                    yield opfunc, body


def iter_schedule_tree_nodes(jsons):
    """Yield every ``scheduleTree_`` allocate node across all sdsc json files."""
    for _, body in iter_sdsc_op_bodies(jsons):
        yield from body.get("scheduleTree_", [])


# ---------------------------------------------------------------------------
# Base test case
# ---------------------------------------------------------------------------
class IndirectAccessTestCase(InductorTestCase):
    """Base class for indirect access tests with common setup and assertions."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(3)
        # Recompile from scratch so our sdsc spy always sees the op specs.
        torch._dynamo.reset()

    # -- Dimension naming helpers ----------------------------------------
    def name_dims(self, tensor, dims: dict):
        """Declare and attach named dimensions to a tensor.

        Args:
            tensor: The tensor to name
            dims: Dict mapping dimension name -> size (in order)
        """
        for nm, size in dims.items():
            declare_tensor_dim(nm, size)
        name_tensor_dims(tensor, list(dims.keys()))

    # -- Assertion helpers -----------------------------------------------
    def assert_reaches_op_spec(self, captured):
        """Verify that compilation reached OpSpec generation."""
        self.assertTrue(captured, "sdsc was never called; no op spec produced")
        op_specs = flatten_op_specs(captured)
        self.assertTrue(op_specs, "no OpSpec found in captured sdsc calls")
        return op_specs

    def assert_gather_op_spec(self, captured):
        """Verify that a gather OpSpec was generated (IndirectAccess on input)."""
        op_specs = self.assert_reaches_op_spec(captured)
        self.assertTrue(
            any(op_spec_has_indirect_input(s) for s in op_specs),
            "no op spec had an IndirectAccess on an input arg (gather not encoded)",
        )
        self.assertFalse(
            any(op_spec_has_indirect_output(s) for s in op_specs),
            "gather should not put IndirectAccess on an output arg",
        )
        return op_specs

    def assert_scatter_op_spec(self, captured):
        """Verify that a scatter OpSpec was generated (IndirectAccess on output)."""
        op_specs = self.assert_reaches_op_spec(captured)
        self.assertTrue(
            any(op_spec_has_indirect_output(s) for s in op_specs),
            "no op spec had an IndirectAccess on an output arg (scatter not encoded)",
        )
        return op_specs
