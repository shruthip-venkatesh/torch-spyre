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

"""Multicore gather scenarios for `out = x[i]`

All scenarios share the SAME shapes; only the value-table *contents* differ, so
each scenario isolates one addressing behavior. With these shapes the planner
wants to split K (the largest dim). Proposed implementation forbids splitting a
value-data dim, so it instead splits the INDEX dim (256 -> 8 sticks -> 8 cores)
and reads full rows.
Without proposed implementation the planner splits K (32 cores) and the shared value
tensor is mis-addressed.

    table   x = [M, K, N]
    index   i = [Q]            (Q gathers into an M-row table)
    output  x[i] = [Q, K, N]

Expected: With proposed implementation, every scenario PASSES. Without proposed
implementation, the `column_addressing` and `realistic_random` scenarios FAIL
(the shared value tensor is read wrongly when K is split); `row_addressing` pass either way
"""

import torch

# Fixed shapes for every scenario.
M = 128  # value-table rows (gather axis) -- tiny, so no 256 MB span involved
K = 64  # value-data dim; the planner wants to split this (largest), proposed implementation forbids it
N = 512  # value-data (stick) dim
Q = 256  # number of gathers -> 8 index sticks -> proposed implementation parallelises here

torch.manual_seed(0)

# Deterministic index: 256 gathers covering all 128 rows twice.
INDEX = torch.arange(Q, dtype=torch.int32) % M

_results: list[tuple[str, bool, str]] = []


def _gather(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cpu_reference, spyre_result) for out = x[INDEX]."""
    ref = x[INDEX]
    compiled = torch.compile(lambda t, idx: t[idx])
    out = compiled(x.to("spyre"), INDEX.to("spyre")).cpu()
    return ref, out


def _record(name: str, ok: bool, note: str) -> None:
    _results.append((name, ok, note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {note}")


# -------------------------------------------------------------------------------------------
# Scenario 1 -- realistic gather: a random table that varies along every dim.
# This is what real value tables (embeddings, weights, KV cache) look like.
# Correct only if BOTH the row (gather) and column (K/N) addressing are right.
#   Proposed Implementation: splits the index dim (8 cores) -> correct.
#   Without Implementation: splits K -> shared value mis-addressed -> Data fetched wrong.
# -------------------------------------------------------------------------------------------
def scenario_realistic_random() -> None:
    x = torch.rand(M, K, N, dtype=torch.float16)
    ref, out = _gather(x)
    # A gather is a pure copy -- it does no arithmetic, so a CORRECT result can
    # still differ from the CPU reference by ~1 ULP because the device stores
    # fp16 in SEN169_FP16, not IEEE fp16.
    close = torch.isclose(ref, out, atol=1e-2, rtol=1e-2)
    ok = bool(close.all())
    out_of_tol = 100.0 * (~close).float().mean().item()
    max_diff = (ref.float() - out.float()).abs().max().item()
    _record(
        "realistic_random",
        ok,
        f"random table; {out_of_tol:.2f}% out of tolerance, max|diff|={max_diff:.2e} "
        f"(tiny diffs are the device fp16 format, not addressing)",
    )


# -------------------------------------------------------------------------------------------
# Scenario 2 -- column addressing: value[r, k, n] = k (varies only along K).
# The number read back at column k IS the column actually fetched.
#   Proposed Implementation: K not split -> every column correct.
#   Without Implementation:K split -> every core reads column 0 -> wrong.
# -------------------------------------------------------------------------------------------
def scenario_column_addressing() -> None:
    x = torch.arange(K, dtype=torch.float16).view(1, K, 1).expand(M, K, N).contiguous()
    _, out = _gather(x)
    got = out[0, :, 0].to(torch.int32)
    expected = torch.arange(K, dtype=torch.int32)
    ok = torch.equal(got, expected)
    bad = (got != expected).nonzero().flatten().tolist()
    note = (
        "all K columns read correctly"
        if ok
        else f"{len(bad)} columns mis-addressed (e.g. col {bad[0]} returned {int(got[bad[0]])})"
    )
    _record("column_addressing", ok, note)


# -------------------------------------------------------------------------------------------
# Scenario 3 -- row addressing: value[r, k, n] = r (varies only along the rows).
# Isolates the gather axis. The index selects the row; this is never split, so
# it is correct with OR without proposed implementation. A sanity check on index selection.
# -------------------------------------------------------------------------------------------
def scenario_row_addressing() -> None:
    x = torch.arange(M, dtype=torch.float16).view(M, 1, 1).expand(M, K, N).contiguous()
    _, out = _gather(x)
    got = out[:, 0, 0].to(torch.int32)
    ok = torch.equal(got, INDEX.to(torch.int32))
    _record("row_addressing", ok, "each output row holds its requested table row")


if __name__ == "__main__":
    print(f"shapes: x=[{M},{K},{N}]  i=[{Q}]  ->  out=[{Q},{K},{N}]\n")
    scenario_realistic_random()
    scenario_column_addressing()
    scenario_row_addressing()

    print("\nsummary:")
    for name, ok, _ in _results:
        print(f"  {'PASS' if ok else 'FAIL':4}  {name}")
