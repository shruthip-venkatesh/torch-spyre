# Multi-Core Indirect Access Approach

## Summary

A gather (`out = x[i]`) reads rows from a value table `x` at positions chosen
at runtime by an index tensor `i`. This page describes **proposed implementation**
of running gathers across multiple Spyre cores: making them **correct** under
work division.

The single load-bearing change is: **the work-division planner must never split
a gather along one of the value table's data dimensions.** Without that
restriction, a gather whose value table is divided across cores along a data
dimension (e.g. the hidden dim of an embedding, the head dim of a KV cache)
silently returns wrong results — every core reads column 0 of the shared table.

Proposed implementation makes such gathers correct. It does **not** attempt to
parallelise the value-data dimensions themselves; that is future work (see
[Limitations](#limitations-and-future-work)).

## Background: how a gather is compiled

A gather is lowered to a **Pointwise `identity` op** with three tensor
arguments:

| Arg | Role | Device coordinates |
|---|---|---|
| index | the positions to read, tagged `KERNEL_IDX` | regular, statically known |
| value | the table being read from (shared) | carries an `IndexLoad(name)` node at the gathered dimension |
| output | the gathered result | regular, statically known |

The `IndexLoad(name)` node (see `op_spec.IndexLoad`) marks the value tensor's
**gather axis** — the dimension whose row is selected at runtime by the index.
That axis has **no iteration-space symbol**; its address is resolved on the
device via `SEGMENT_OFFSETS`, with `maxDimSizes == 1` for that dimension.

Work division splits the op's iteration space across cores. For a gather over
`x : [M, K, N]` with `i : [Q]`, the iteration space is `{d0 = Q, d1 = K,
d2 = N}` and the output is `[Q, K, N]`. Note that the value's gather axis `M`
is **not** an iteration variable — the iteration dims are the index-entry
dim (`Q`) and the value's data dims (`K`, `N`).

### The gather already parallelises along the index — by construction

`get_mem_deps_from_rw` ([pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py))
filters indirect reads out of work division's inputs, so the planner only ever
sees the **index read** and the **output write** — both regular tensors. It
naturally distributes cores over their dimensions, which include the index-entry
dim. So index-driven parallelism works without special handling. The problem is
purely that nothing stops the planner from *also* splitting a value-data dim.

## The defect

The planner ranks dimensions by size and splits the largest. When a value-data
dim (`K` or `N`) is the largest — the common case for **wide rows** (embedding
`hidden`, expert weight matrices, attention `head_dim`) — it splits that
dimension. That is wrong, because the value table is **shared**: every core
must be able to read any row, so it cannot be sliced per core.

Concretely, under a K-split the output advances its per-core base across `K`
(core `c` writes columns `[2c, 2c+2)`), but the shared value tensor's per-core
base does **not** advance. So every core reads value columns `[0, 2)` and writes
them into its own output columns:

```text
out[:, 2c : 2c+2, :] = value[:, 0 : 2, :]    for every core c
```

Core 0 is correct by coincidence; the other 31 write core-0's columns into their
slices.

### Why it was not caught earlier

The defect is **silent** — wrong numbers, no crash — and is **masked by
constant data**: if every column of a value row holds the same number, reading
the wrong column is indistinguishable from reading the right one. It only
surfaces when the value varies along the split dimension, which is true of all
real tables but not of a constant-filled test.

Validated on hardware (`x : [128, 64, 512]`, `i : [256]`, `SENCORES=32`):

## The proposed fix

Four changes, in order of importance.

### 1. Forbid splitting the value-data dims — the correctness fix

`indirect_value_data_syms` ([work_division.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/work_division.py))
returns the iteration symbols that index the value tensor's non-gather (data)
dimensions. `_default_split` removes them from the output and reduction priority
lists, so the core budget is distributed only over the index-entry dims —
**divide by the index tensor, not the value tensor.**

When the index dim cannot absorb all the cores, the gather falls back to fewer
cores rather than splitting a value-data dim. **Correct-but-serial is the
intended trade-off; silent corruption is not.**

### 2. Shared-base value addressing

`SDSCArgs.shared_base` ([codegen/superdsc.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/codegen/superdsc.py))
marks the value tensor. In `core_idx_to_slice_offset`
([codegen/compute_ops.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/codegen/compute_ops.py))
a `shared_base` tensor keeps its per-core base at the static offset (address 0)
instead of advancing with the work-division slice. The gather row comes from the
runtime `IndexLoad`; the index and output tensors keep the normal per-core base
advance. This makes the shared-table addressing correct by construction rather
than relying on the backend to tolerate a stale per-core base.

### 3. Value-table span guard (defensive)

`collect_indirect_value_tds` rebuilds the value `TensorDep` with
`IndexLoad`-aware coordinates so the value table is visible to the per-core
span check, and `get_per_core_span` treats an `IndexLoad` coordinate as
contributing its full device extent (any core may touch any row) and never
splits it. `warn_if_per_core_overflow` then logs a critical message if the
value table would exceed the documented 256 MB per-core span.

This is conservative: a 512 MB value table read correctly in testing, so the
limit's applicability to indirect value tensors is not firmly established. The
guard surfaces a potential hardware-limit violation as a compile-time warning
rather than failing.

### 4. Deterministic split round-trip

The split plan is encoded with the coefficients of the read/write index
expressions. An indirect read carries data-dependent symbols whose coefficients
are not a stable identity key, so the encode side (`apply_splits`) and both
decode sites (`work_distribution_pass`, `create_op_spec`) now prefer the first
**non-indirect** read as the reduction-split reference index, via
`_first_non_indirect_read_index`.

## Worked example

`out = x[i]` with `x : [128, 64, 512]`, `i : [256]`, `SENCORES=32`. Iteration
space `{d0 = Q = 256, d1 = K = 64, d2 = N = 512}`; the value's gather axis
`M = 128` is addressed by `IndexLoad` and is not an iteration symbol. The index
stickifies to `256 / 32 = 8` sticks on `d0`.

| | Without the fix | With the proposed fix |
|---|---|---|
| Largest dim | `K = 64` | (forbidden) |
| Split | `K`, 32 cores | `d0` (index), 8 cores |
| Value tensor | per-core column base diverges | shared at base 0 |
| Result | wrong (every core reads column 0) | correct |

Parallelism with the fix is set by the index size in sticks:
`cores = min(Q / 32, 32)` for a 1-D index (`Q = 256 → 8`, `Q = 1024 → 32`), or a
spatial (non-stick) index dimension splits directly.

## Validation

`examples/gather_multicore_exp.py` runs three scenarios at the shapes above, each
isolating one behavior by changing only the value-table contents:

| Scenario | Table contents | Isolates | With proposed fix | Without fix |
|---|---|---|---|---|
| `realistic_random` | `rand()` | real-world gather | pass | fail (~97% wrong) |
| `column_addressing` | `x[r,k,n] = k` | K-column addressing | pass (exact) | fail (all read col 0) |
| `row_addressing` | `x[r,k,n] = r` | gather-axis selection | pass (exact) | pass (row never split) |

`row_addressing` pass either way and are included to show, respectively,
that the gather axis is always correct and hides the defect.
 `column_addressing` and `realistic_random` are the correctness signal.

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 SENCORES=32 python examples/gather_multicore_exp.py
```

## Limitations and future work

- **Parallelism is capped by the index size.** Current proposal parallelises only over
  the index dimension. Workloads with a small, fixed index but wide rows — MoE
  expert gathers (top-k routing), paged-attention KV reads (short block tables),
  wide embedding lookups at small batch — get few cores (often 1–2). They are
  **correct**, but not maximally parallel.

- **Next phase (parallelising the value-data dims) is deferred.** A prototype
  let the planner split a value-data dim and advanced the value tensor's per-core
  base along it. On hardware the per-core base of a `value_tensor` allocation is
  interpreted as an absolute offset into the table *where the gather happens*, so
  advancing it shifts the gathered **row**, not the **column** — there is no
  per-core base value that expresses "different columns per core" for a value
  tensor. Making the data-dim split correct requires carrying the per-core column
  offset in the coordinate folds and/or a backend change to how `value_tensor`
  bases are interpreted. This needs the DeepTools/backend team and is out of
  scope for current proposal.

- **Scatter is not addressed.** For a scatter (`out[i] = src`), the `IndexLoad`
  is on the output, so splitting index-entry dims gives data-dependent
  destinations: two cores can write the same output row. Multi-core scatter is
  only safe under an index-uniqueness guarantee and is left for future work.

## Implementation

| File | Change |
|---|---|
| `_inductor/work_division.py` | `indirect_value_data_syms`, `collect_indirect_value_tds`, `IndexLoad` span guard, `forbidden_split_syms` in `_default_split`, `_first_non_indirect_read_index` |
| `_inductor/codegen/superdsc.py` | `SDSCArgs.shared_base`, set for the value tensor |
| `_inductor/codegen/compute_ops.py` | `core_idx_to_slice_offset` honours `shared_base` |
| `_inductor/spyre_kernel.py` | non-indirect read index in `create_op_spec` |
| `examples/gather_multicore_exp.py` | three-scenario validation example |

## See also

- [Work Division Planning](work_division_planning.md) — the three-pass planner
  this builds on.
