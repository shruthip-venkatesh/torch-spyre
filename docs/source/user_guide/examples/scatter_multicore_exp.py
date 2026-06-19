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

"""Multicore scatter scenarios for `out[i] = x`

All scenarios share the SAME shapes; only the value-table *contents* differ, so
each scenario isolates one addressing behavior. With these shapes the planner
wants to split K (the largest dim). Proposed implementation forbids splitting a
value-data dim, so it instead splits the INDEX dim (M sticks -> cores) and
writes full rows. Without proposed implementation the planner splits K and the
shared output tensor is mis-addressed.

These shapes follow the known-good single-core example: the source first dim,
the index length, and the output first dim are all equal (M), and the index is
a permutation of the M rows (every row written exactly once, like
`out = torch.zeros_like(x); out[i] = x`).

    source  x = [M, K, N]
    index   i = [M]            (a permutation of the M output rows)
    output  out[i] = x -> [M, K, N]

Expected: With proposed implementation, every scenario PASSES. Without proposed
implementation, the `column_addressing` and `realistic_random` scenarios FAIL
(the shared output tensor is written wrongly when K is split); `row_addressing`
passes either way.
"""

import gc
import torch
import torch._dynamo

# Fixed shapes for every scenario -- matched to the working single-core example:
# source rows == index length == output rows, so out = zeros_like(x).
M = 256  # rows: scatter axis == source rows == output rows == index length
K = 64  # value-data dim; the planner wants to split this (largest), implementation forbids it
N = 512  # value-data (stick) dim

torch.manual_seed(0)

# Deterministic permutation index: each of the M rows is written exactly once
# (no duplicate writes), mirroring `out[i] = x` in the working example.
INDEX = torch.randperm(M, dtype=torch.int32)

_results: list[tuple[str, bool, str]] = []


def _scatter(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (cpu_reference, spyre_result) for out[INDEX] = x."""
    out_ref = torch.zeros(M, K, N, dtype=torch.float16)
    out_ref[INDEX] = x

    def kernel(out, x, idx):
        out[idx] = x
        return out

    # Clear any cached guards/graphs so a failure in one scenario cannot leave
    # stale compiler state that poisons the next one.
    torch._dynamo.reset()

    try:
        compiled = torch.compile(kernel)
        out = torch.zeros(M, K, N, dtype=torch.float16, device="spyre")
        x_dev = x.to("spyre")
        idx_dev = INDEX.to("spyre")
        result = compiled(out, x_dev, idx_dev).cpu()

        # Explicitly clean up device tensors to prevent memory corruption
        del out, x_dev, idx_dev, compiled
        gc.collect()

        return out_ref, result
    except Exception as e:
        msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
        print(f"  Compilation/execution failed: {type(e).__name__}: {msg}")
        # Clean up on error path as well
        gc.collect()
        return out_ref, None


def _record(name: str, ok: bool, note: str) -> None:
    _results.append((name, ok, note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {note}")


# -------------------------------------------------------------------------------------------
# Scenario 1 -- realistic scatter: random data that varies along every dim.
# This is what real scatter operations (embedding updates, gradient accumulation) look like.
# Correct only if BOTH the row (scatter) and column (K/N) addressing are right.
#   Proposed Implementation: splits the index dim -> correct.
#   Without Implementation: splits K -> shared output mis-addressed -> Data written wrong.
# -------------------------------------------------------------------------------------------
def scenario_realistic_random() -> None:
    x = torch.rand(M, K, N, dtype=torch.float16)
    ref, out = _scatter(x)
    if out is None:
        _record("realistic_random", False, "compilation/execution failed")
        del x, ref
        gc.collect()
        return
    # A scatter is a pure copy -- it does no arithmetic, so a CORRECT result can
    # still differ from the CPU reference by ~1 ULP because the device stores
    # fp16 in SEN169_FP16, not IEEE fp16.
    close = torch.isclose(ref, out, atol=1e-2, rtol=1e-2)
    ok = bool(close.all())
    out_of_tol = 100.0 * (~close).float().mean().item()
    max_diff = (ref.float() - out.float()).abs().max().item()
    _record(
        "realistic_random",
        ok,
        f"random data; {out_of_tol:.2f}% out of tolerance, max|diff|={max_diff:.2e} "
        f"(tiny diffs are the device fp16 format, not addressing)",
    )
    del x, ref, out, close
    gc.collect()


# -------------------------------------------------------------------------------------------
# Scenario 2 -- column addressing: source[m, k, n] = k (varies only along K).
# The number written at column k IS the column actually stored.
#   Proposed Implementation: K not split -> every column correct.
#   Without Implementation: K split -> every core writes to column 0 -> wrong.
# -------------------------------------------------------------------------------------------
def scenario_column_addressing() -> None:
    x = torch.arange(K, dtype=torch.float16).view(1, K, 1).expand(M, K, N).contiguous()
    _, out = _scatter(x)
    if out is None:
        _record("column_addressing", False, "compilation/execution failed")
        del x
        gc.collect()
        return
    # Every row carries the same column pattern; check the row written by INDEX[0].
    check_row = int(INDEX[0])
    got = out[check_row, :, 0].to(torch.int32)
    expected = torch.arange(K, dtype=torch.int32)
    ok = torch.equal(got, expected)
    bad = (got != expected).nonzero().flatten().tolist()
    note = (
        "all K columns written correctly"
        if ok
        else f"{len(bad)} columns mis-addressed (e.g. col {bad[0]} has {int(got[bad[0]])})"
    )
    _record("column_addressing", ok, note)
    del x, out, got, expected
    gc.collect()


# -------------------------------------------------------------------------------------------
# Scenario 3 -- row addressing: source[m, k, n] = m (varies only along the scatter dim).
# Isolates the scatter axis. The index selects the row; this is never split, so
# it is correct with OR without proposed implementation. A sanity check on index selection.
# -------------------------------------------------------------------------------------------
def scenario_row_addressing() -> None:
    x = torch.arange(M, dtype=torch.float16).view(M, 1, 1).expand(M, K, N).contiguous()
    _, out = _scatter(x)
    if out is None:
        _record("row_addressing", False, "compilation/execution failed")
        del x
        gc.collect()
        return
    # Source row q (value q) is written to output row INDEX[q]; check the round-trip.
    for q in range(M):
        row_idx = int(INDEX[q])
        got_val = int(out[row_idx, 0, 0])
        if got_val != q:
            _record(
                "row_addressing", False, f"row {row_idx} has {got_val}, expected {q}"
            )
            del x, out
            gc.collect()
            return
    _record(
        "row_addressing", True, "each output row holds the source row that wrote it"
    )
    del x, out
    gc.collect()


if __name__ == "__main__":
    print(f"shapes: x=[{M},{K},{N}]  i=[{M}]  ->  out[i]=x -> [{M},{K},{N}]\n")
    scenario_realistic_random()
    scenario_column_addressing()
    scenario_row_addressing()

    print("\nsummary:")
    for name, ok, _ in _results:
        print(f"  {'PASS' if ok else 'FAIL':4}  {name}")

    # Force garbage collection to clean up any remaining references before exit
    import gc

    gc.collect()
