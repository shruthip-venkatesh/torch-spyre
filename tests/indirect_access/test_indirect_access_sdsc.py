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

"""Tests for the SDSC boundary in indirect access operations.

This file tests the handoff to SuperDSC bundle generation:

  * Gather: Reaches sdsc with well-formed OpSpecs
  * Scatter: Reaches sdsc with well-formed OpSpecs
  * SuperDSC generation: Python side works, but backend doesn't support it yet
  * Capture harness: Properly patches and restores sdsc
"""

import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    CRASHED,
    DIRECT_OP_SPEC,
    GATHER_OP_SPEC,
    NO_SPYRE_OP,
    SCATTER_OP_SPEC,
    UNIMPLEMENTED,
    IndirectAccessTestCase,
    capture_op_specs,
    capture_sdsc_calls,
    flatten_entries,
    flatten_op_specs,
    generate_sdsc_jsons,
    iter_schedule_tree_nodes,
    iter_sdsc_op_bodies,
)

from torch._inductor.exc import InductorError  # noqa: E402
from torch_spyre._inductor.op_spec import (  # noqa: E402
    LoopSpec,
    OpSpec,
    UnimplementedOp,
    find_unimplemented,
)
from torch_spyre.execution.async_compile import SpyreAsyncCompile  # noqa: E402

# Mock target to disable actual kernel execution on device
_LAUNCH_KERNEL = "torch_spyre.execution.kernel_runner.launch_kernel"


def _gather_exp():
    M, N, P, Q = 128, 256, 3, 192
    x = torch.rand(M, N, dtype=torch.float16)
    i = torch.randint(0, M, (P, Q), dtype=torch.int32)

    def kernel(x, i):
        return x[i].exp()

    return kernel, (M, N, P, Q), x, i


class TestSdscHandoffGather(IndirectAccessTestCase):
    """Tests that gather operations reach sdsc with valid OpSpecs."""

    def test_gather_invokes_sdsc_with_specs(self):
        """Test that gather calls sdsc with valid kernel name and OpSpecs."""
        kernel, (M, N, P, Q), x, i = _gather_exp()
        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with capture_sdsc_calls() as calls:
            torch.compile(kernel)(x_dev, i_dev)

        self.assertTrue(calls, "sdsc was never called")
        for kernel_name, specs in calls:
            self.assertIsInstance(kernel_name, str)
            self.assertTrue(kernel_name, "kernel name is empty")
            self.assertTrue(specs, "spec list is empty")
        # Verify at least one real OpSpec was generated
        all_specs = [s for _, specs in calls for s in specs]
        self.assertTrue(flatten_op_specs([all_specs]))

    def test_gather_kernel_name_is_sdsc_named(self):
        """Test that Spyre kernels are named with 'sdsc_' prefix."""
        kernel, (M, N, P, Q), x, i = _gather_exp()
        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with capture_sdsc_calls() as calls:
            torch.compile(kernel)(x_dev, i_dev)

        self.assertTrue(
            any("sdsc" in name.lower() for name, _ in calls),
            f"expected an 'sdsc'-named kernel; got {[n for n, _ in calls]}",
        )

    def test_gather_specs_have_no_unimplemented(self):
        """Test that supported gather operations don't contain UnimplementedOps.

        This means sdsc won't short-circuit to SpyreUnimplementedRunner.
        """
        kernel, (M, N, P, Q), x, i = _gather_exp()
        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with capture_op_specs() as captured:
            torch.compile(kernel)(x_dev, i_dev)

        self.assertTrue(captured)
        for specs in captured:
            self.assertIsNone(
                find_unimplemented(specs),
                "supported gather should not contain an UnimplementedOp",
            )


class TestSuperDscGenerationGather(IndirectAccessTestCase):
    """Tests for SuperDSC bundle generation with gather operations.

    Current status:
    - Python SuperDSC generation: WORKS
    - Backend (dxp_standalone): FAILS with "Expect LX in labeledDs memOrg"

    When subprocess.run is mocked, gather compiles successfully through
    generate_bundle. The gap is in the backend, not Python code.
    """

    def test_python_bundle_generation_succeeds(self):
        """Test that Python SuperDSC generation works for gather.

        With backend subprocess mocked, bundle generation completes successfully.
        """
        kernel, (M, N, P, Q), x, i = _gather_exp()
        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with patch("subprocess.run"), patch(_LAUNCH_KERNEL):
            result = torch.compile(kernel)(x_dev, i_dev)

        self.assertIsNotNone(result, "gather compile returned nothing")

    @unittest.skip(
        "MANUAL TEST ONLY: Runs real backend which crashes and corrupts memory. "
        "Run in isolation: pytest -k test_real_backend_compile_for_gather_aborts. "
        "Remove skip once backend supports indirect gather."
    )
    def test_real_backend_compile_for_gather_aborts(self):
        """Test that backend currently crashes on indirect gather.

        SKIPPED: Backend aborts with SIGABRT and corrupts memory.
        Kernel launch is mocked to avoid touching hardware.
        """
        kernel, (M, N, P, Q), x, i = _gather_exp()
        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})

        with patch(_LAUNCH_KERNEL):
            with self.assertRaises(InductorError):
                torch.compile(kernel)(x_dev, i_dev)


class TestSdscBoundaryScatter(IndirectAccessTestCase):
    """Scatter now reaches the SDSC layer."""

    def test_scatter_method_reaches_sdsc(self):
        """Scatter is handed to sdsc with a non-empty spec list."""
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

        with capture_sdsc_calls() as calls:
            torch.compile(kernel)(y_dev, src_dev, index_dev)
        self.assertTrue(calls, "scatter never reached the sdsc layer")
        for _kernel_name, specs in calls:
            self.assertTrue(specs, "empty spec list handed to sdsc")


class TestCaptureHarness(IndirectAccessTestCase):
    """Tests that the capture harness properly patches and restores sdsc."""

    def test_capture_restores_sdsc(self):
        """Test that capture_op_specs properly patches and restores sdsc."""
        original = SpyreAsyncCompile.sdsc
        with capture_op_specs() as captured:
            self.assertIsNot(SpyreAsyncCompile.sdsc, original, "should be patched")
            self.assertEqual(captured, [], "nothing compiled yet")
        self.assertIs(SpyreAsyncCompile.sdsc, original, "must be restored")

    def test_capture_sdsc_calls_restores_on_exception(self):
        """Test that sdsc is restored even when an exception occurs."""
        original = SpyreAsyncCompile.sdsc
        with self.assertRaises(ValueError):
            with capture_sdsc_calls():
                raise ValueError("boom")
        self.assertIs(SpyreAsyncCompile.sdsc, original, "must be restored after error")


def _op(name):
    return OpSpec(op=name, is_reduction=False, iteration_space={}, args=[], op_info={})


class TestSpecTreeUtils(IndirectAccessTestCase):
    """Unit tests for spec-tree flattening utilities (no device needed)."""

    def test_flatten_descends_into_loopspec(self):
        """Test that flatten_op_specs extracts ops from nested LoopSpecs."""
        inner = LoopSpec(count=2, body=[_op("exp")])
        loop = LoopSpec(count=4, body=[_op("identity"), inner])
        specs = flatten_op_specs([[loop, _op("tanh")]])
        self.assertEqual([s.op for s in specs], ["identity", "exp", "tanh"])

    def test_flatten_op_specs_drops_unimplemented(self):
        """Test that flatten_op_specs ignores UnimplementedOps."""
        specs = flatten_op_specs([[_op("exp"), UnimplementedOp("sin")]])
        self.assertEqual([s.op for s in specs], ["exp"])

    def test_flatten_entries_keeps_unimplemented(self):
        """Test that flatten_entries includes UnimplementedOps."""
        entries = flatten_entries([[_op("exp"), UnimplementedOp("sin")]])
        kinds = [type(e).__name__ for e in entries]
        self.assertIn("UnimplementedOp", kinds)
        self.assertIn("OpSpec", kinds)

    def test_flatten_multiple_sdsc_calls(self):
        """Test that multiple sdsc calls (multiple kernels) flatten correctly."""
        specs = flatten_op_specs([[_op("identity")], [_op("exp")]])
        self.assertEqual([s.op for s in specs], ["identity", "exp"])

    def test_classification_labels_are_distinct(self):
        """Test that all classification outcome labels are unique."""
        labels = {
            CRASHED,
            GATHER_OP_SPEC,
            SCATTER_OP_SPEC,
            DIRECT_OP_SPEC,
            UNIMPLEMENTED,
            NO_SPYRE_OP,
        }
        self.assertEqual(len(labels), 6, "classification labels must be unique")

    def test_flatten_empty(self):
        self.assertEqual(flatten_op_specs([]), [])
        self.assertEqual(flatten_op_specs([[]]), [])

    def test_flatten_entries_empty(self):
        self.assertEqual(flatten_entries([]), [])

    def test_flatten_deeply_nested_loopspec(self):
        deep = LoopSpec(count=2, body=[LoopSpec(count=2, body=[_op("exp")])])
        outer = LoopSpec(count=4, body=[_op("identity"), deep])
        specs = flatten_op_specs([[outer]])
        self.assertEqual([s.op for s in specs], ["identity", "exp"])

    def test_flatten_entries_mixed(self):
        entries = flatten_entries(
            [[_op("a"), LoopSpec(count=2, body=[UnimplementedOp("sin"), _op("b")])]]
        )
        self.assertEqual([getattr(e, "op", None) for e in entries], ["a", "sin", "b"])

    def test_iter_sdsc_op_bodies_synthetic(self):
        fake = {
            "sdsc_0.json": {
                "0_exp": {"dscs_": [{"exp": {"scheduleTree_": [{"ldsIdx_": 0}]}}]}
            }
        }
        bodies = list(iter_sdsc_op_bodies(fake))
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0][0], "exp")

    def test_iter_schedule_tree_nodes_synthetic(self):
        fake = {
            "sdsc_0.json": {
                "0_exp": {
                    "dscs_": [
                        {"exp": {"scheduleTree_": [{"ldsIdx_": 0}, {"ldsIdx_": 1}]}}
                    ]
                }
            }
        }
        self.assertEqual(len(list(iter_schedule_tree_nodes(fake))), 2)


class TestSdscGenerationGatherFields(IndirectAccessTestCase):
    """Tests for SDSC JSON generation and field verification.

    These tests run real bundle generation (without backend) and verify:
    - JSON structure is correct
    - Indirect access is properly encoded (index_tensor / value_tensor)
    - Index tensors are HBM-only (not in LX memory)
    """

    def _gen(self):
        kernel, (M, N, P, Q), x, i = _gather_exp()
        x_dev, i_dev = x.to("spyre"), i.to("spyre")
        self.name_dims(x_dev, {"M": M, "N": N})
        self.name_dims(i_dev, {"P": P, "Q": Q})
        return generate_sdsc_jsons(kernel, x_dev, i_dev)

    def test_gather_emits_sdsc_json(self):
        """Test that gather generates SDSC JSON files with correct structure."""
        jsons = self._gen()
        self.assertTrue(jsons, "no SDSC JSON files generated")
        for fname, top in jsons.items():
            self.assertTrue(fname.endswith(".json"))
            self.assertEqual(len(top), 1, "expected one top-level operation key")
            self.assertRegex(next(iter(top)), r"^\d+_.+")

    def test_gather_sdsc_core_structural_fields(self):
        """Test that SDSC JSON contains all required structural fields."""
        bodies = list(iter_sdsc_op_bodies(self._gen()))
        self.assertTrue(bodies, "no operation bodies found in SDSC JSON")
        for opfunc, body in bodies:
            # Check all required fields are present
            for field in (
                "N_",
                "scheduleTree_",
                "labeledDs_",
                "computeOp_",
                "numCoresUsed_",
            ):
                self.assertIn(field, body, f"{opfunc} missing field: {field}")
            # Verify field consistency
            self.assertEqual(len(body["labeledDs_"]), len(body["scheduleTree_"]))
            self.assertGreaterEqual(body["numCoresUsed_"], 1)
            self.assertTrue(body["computeOp_"][0]["opFuncName"])

    def test_gather_sdsc_encodes_indirect_access(self):
        """Test that indirect access is properly encoded in SDSC JSON.

        The index/value relationship should be in scheduleTree_ with
        index_tensor and value_tensor nodes.
        """
        index_nodes, value_nodes = [], []
        for _, body in iter_sdsc_op_bodies(self._gen()):
            for node in body["scheduleTree_"]:
                kind = node.get("indirectAllocType_")
                if kind == "index_tensor":
                    index_nodes.append(node)
                elif kind == "value_tensor":
                    value_nodes.append(node)
        self.assertTrue(index_nodes, "no index_tensor nodes found")
        self.assertTrue(value_nodes, "no value_tensor nodes found")
        for n in index_nodes:
            self.assertEqual(n.get("indexTensorType_"), "index")
            self.assertIn("relatedIndirectAccessAlloc_", n)

    def test_gather_sdsc_index_tensor_is_hbm_only(self):
        """Test that index tensors are in HBM memory, not LX.

        Spyre can't do indirect addressing through LX memory. The backend
        currently rejects this with 'Expect LX in labeledDs memOrg'.
        """
        checked = False
        for _, body in iter_sdsc_op_bodies(self._gen()):
            # Find all index tensor nodes
            idx_ldsidx = {
                node["ldsIdx_"]
                for node in body["scheduleTree_"]
                if node.get("indirectAllocType_") == "index_tensor"
            }
            # Verify they're in HBM, not LX
            for lds in body["labeledDs_"]:
                if lds["ldsIdx_"] in idx_ldsidx:
                    checked = True
                    self.assertIn("hbm", lds["memOrg_"])
                    self.assertNotIn("lx", lds["memOrg_"], "index must be in HBM only")
        self.assertTrue(checked, "no index tensor entries found")

    # -- more SDSC field checks -----------------------------------------
    def test_gather_sdsc_single_top_level_key(self):
        for top in self._gen().values():
            self.assertEqual(len(top), 1)

    def test_gather_sdsc_labeled_ds_field_shape(self):
        for _, body in iter_sdsc_op_bodies(self._gen()):
            for lds in body["labeledDs_"]:
                for key in (
                    "ldsIdx_",
                    "dsName_",
                    "dsType_",
                    "dataFormat_",
                    "wordLength",
                    "memOrg_",
                ):
                    self.assertIn(key, lds)
                self.assertIsInstance(lds["dsName_"], str)
                self.assertIsInstance(lds["dataFormat_"], str)

    def test_gather_sdsc_wordlength_correct_for_dtype(self):
        """Test that wordLength matches data type: 2 for FP16, 4 for int32."""
        for _, body in iter_sdsc_op_bodies(self._gen()):
            # Get index tensor ldsIdx values
            index_ldsidx = {
                node["ldsIdx_"]
                for node in body["scheduleTree_"]
                if node.get("indirectAllocType_") == "index_tensor"
            }

            for lds in body["labeledDs_"]:
                if lds["ldsIdx_"] in index_ldsidx:
                    # Index tensors are int32, wordLength should be 4
                    self.assertEqual(
                        lds["wordLength"],
                        4,
                        "Index tensor should have wordLength 4 (int32)",
                    )
                else:
                    # Value tensors are FP16, wordLength should be 2
                    self.assertEqual(
                        lds["wordLength"],
                        2,
                        "Value tensor should have wordLength 2 (FP16)",
                    )

    def test_gather_sdsc_dsname_matches_ldsidx(self):
        for _, body in iter_sdsc_op_bodies(self._gen()):
            for lds in body["labeledDs_"]:
                self.assertEqual(lds["dsName_"], f"Tensor{lds['ldsIdx_']}")

    def test_gather_sdsc_schedule_node_fields(self):
        for node in iter_schedule_tree_nodes(self._gen()):
            self.assertEqual(node["nodeType_"], "allocate")
            self.assertTrue(node["name_"].startswith("allocate-Tensor"))
            self.assertIn(node["component_"], ("hbm", "lx"))
            self.assertIn(
                node["indirectAllocType_"],
                ("index_tensor", "value_tensor", "no_indirection"),
            )

    def test_gather_sdsc_computeop_io_labels(self):
        for _, body in iter_sdsc_op_bodies(self._gen()):
            cop = body["computeOp_"][0]
            self.assertTrue(cop["opFuncName"])
            self.assertIsInstance(cop["inputLabeledDs"], list)
            self.assertTrue(cop["outputLabeledDs"], "output label missing")

    def test_gather_sdsc_numcores_in_range(self):
        for _, body in iter_sdsc_op_bodies(self._gen()):
            self.assertGreaterEqual(body["numCoresUsed_"], 1)
            self.assertLessEqual(body["numCoresUsed_"], 32)

    def test_gather_sdsc_indirect_cross_link(self):
        """index_tensor nodes point at an 'allocate-Tensor{n}_hbm' value alloc."""
        for _, body in iter_sdsc_op_bodies(self._gen()):
            for node in body["scheduleTree_"]:
                if node.get("indirectAllocType_") == "index_tensor":
                    related = node.get("relatedIndirectAccessAlloc_")
                    if related is not None:
                        self.assertRegex(related, r"^allocate-Tensor\d+_hbm$")


class TestSdscGenerationScatter(IndirectAccessTestCase):
    """Scatter now generates SDSC bundles (Python generate_bundle succeeds).

    A scatter lowers to two ops (copy self -> output, then scatter src into it),
    so it emits two SDSC json files. The deeptools backend still rejects them,
    but generate_sdsc_jsons stops before the backend.
    """

    def _gen(self):
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
        return generate_sdsc_jsons(kernel, y_dev, src_dev, index_dev)

    def test_scatter_produces_sdsc(self):
        self.assertTrue(self._gen(), "scatter produced no SDSC JSON")

    def test_scatter_sdsc_core_structural_fields(self):
        bodies = list(iter_sdsc_op_bodies(self._gen()))
        self.assertTrue(bodies, "no compiled op bodies in scatter SDSC")
        for opfunc, body in bodies:
            for field in ("N_", "scheduleTree_", "labeledDs_", "computeOp_"):
                self.assertIn(field, body, f"{opfunc} body missing {field}")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
