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

import torch
import math

# Simplified parameters for 1D indexing test
total_elements = 32  # Total number of elements to process
cache_size = 64  # Total cache slots
H = 4  # Number of heads
D = 8  # Head dimension

print("=" * 60)
print("1D INDEX TENSOR EXAMPLE - Pure 1D Indirect Access")
print("WITH SPYRE DEVICE COMPARISON")
print("=" * 60)

# Create sample data
torch.manual_seed(42)
queries = torch.randn(total_elements, H, D, dtype=torch.float16)
keys = torch.randn(cache_size, H, D, dtype=torch.float16)
values = torch.randn(cache_size, H, D, dtype=torch.float16)

print("\nInput Shapes:")
print(f"  queries: {queries.shape} = (total_elements, heads, dim)")
print(f"  keys:    {keys.shape} = (cache_size, heads, dim)")
print(f"  values:  {values.shape} = (cache_size, heads, dim)")

# Create 1D slot indices - random indices into the cache
slot_idxs = torch.randint(0, cache_size, (total_elements,), dtype=torch.int64)

print("\n1D Slot Indices:")
print(f"  Shape: {slot_idxs.shape} (pure 1D tensor)")
print(f"  Values: {slot_idxs.tolist()}")
print(f"  Min index: {slot_idxs.min().item()}, Max index: {slot_idxs.max().item()}")

# ============================================================================
# SIMPLIFIED COMPUTATION WITH 1D INDEXING
# ============================================================================

print("\n" + "=" * 60)
print("SIMPLIFIED COMPUTATION WITH 1D INDIRECT ACCESS")
print("=" * 60)


def gather_kv(table, idx):
    """Pure 1D gather operation"""
    return table[idx]


def simple_computation_1d(queries, keys, values, slot_idxs, gather_kv_fn=gather_kv):
    """
    Simplified computation using pure 1D slot indices

    Args:
        queries: (N, H, D) where N = total_elements
        keys: (cache_size, H, D)
        values: (cache_size, H, D)
        slot_idxs: (N,) - 1D indices into cache
    """
    # Gather keys and values using 1D indices
    # This is the KEY operation being tested!
    gathered_keys = gather_kv_fn(keys, slot_idxs)  # (N, H, D)
    gathered_values = gather_kv_fn(values, slot_idxs)  # (N, H, D)

    print(f"  Gathered keys shape: {gathered_keys.shape}")
    print(f"  Gathered values shape: {gathered_values.shape}")

    # Simple element-wise computation
    # Compute dot product between queries and gathered keys
    scale = 1.0 / math.sqrt(D)
    scores = (queries * gathered_keys).sum(dim=-1) * scale  # (N, H)
    weights = torch.softmax(scores, dim=-1)  # (N, H)

    # Apply weights to gathered values
    output = gathered_values * weights.unsqueeze(-1)  # (N, H, D)

    return output


# ============================================================================
# CPU REFERENCE EXECUTION
# ============================================================================

print("\n" + "=" * 60)
print("CPU REFERENCE EXECUTION")
print("=" * 60)

cpu_output = simple_computation_1d(queries, keys, values, slot_idxs)
print(f"\nCPU Output shape: {cpu_output.shape}")
print(f"CPU Sample values (element 0, head 0): {cpu_output[0, 0, :4].tolist()}")
print(f"CPU Sample values (element 1, head 0): {cpu_output[1, 0, :4].tolist()}")

# ============================================================================
# SPYRE DEVICE EXECUTION
# ============================================================================

print("\n" + "=" * 60)
print("SPYRE DEVICE EXECUTION WITH TORCH.COMPILE")
print("=" * 60)

try:
    if not torch.spyre.is_available():
        print("ERROR: Spyre device is not available!")
    else:
        print("Spyre device is available!")

        # Move tensors to Spyre device
        print("\nMoving tensors to Spyre device...")
        queries_spyre = queries.to("spyre")
        keys_spyre = keys.to("spyre")
        values_spyre = values.to("spyre")
        slot_idxs_spyre = slot_idxs.to("spyre")

        print(f"  queries device: {queries_spyre.device}")
        print(f"  keys device: {keys_spyre.device}")
        print(f"  values device: {values_spyre.device}")
        print(f"  slot_idxs device: {slot_idxs_spyre.device}")
        print(f"  slot_idxs is 1D: {slot_idxs_spyre.dim() == 1}")

        # Compile the gather function
        print("\nCompiling gather function with torch.compile...")
        compiled_gather_kv = torch.compile(gather_kv, backend="inductor")

        # Run on Spyre device
        print("Executing on Spyre device...")
        spyre_output = simple_computation_1d(
            queries_spyre, keys_spyre, values_spyre, slot_idxs_spyre, compiled_gather_kv
        )

        print(f"\nSpyre Output shape: {spyre_output.shape}")
        print(f"Spyre Output device: {spyre_output.device}")

        # Move back to CPU for comparison
        spyre_output_cpu = spyre_output.cpu()
        print(
            f"Spyre Sample values (element 0, head 0): {spyre_output_cpu[0, 0, :4].tolist()}"
        )
        print(
            f"Spyre Sample values (element 1, head 0): {spyre_output_cpu[1, 0, :4].tolist()}"
        )

        # ============================================================================
        # VALIDATION: Compare CPU vs Spyre
        # ============================================================================

        print("\n" + "=" * 60)
        print("VALIDATION: CPU vs SPYRE")
        print("=" * 60)

        # Compute differences
        abs_diff = torch.abs(cpu_output - spyre_output_cpu)
        rel_diff = abs_diff / (torch.abs(cpu_output) + 1e-8)

        max_abs_diff = abs_diff.amax().item()
        mean_abs_diff = abs_diff.mean().item()
        max_rel_diff = rel_diff.amax().item()
        mean_rel_diff = rel_diff.mean().item()

        print("\nAbsolute Differences:")
        print(f"  Max:  {max_abs_diff:.6e}")
        print(f"  Mean: {mean_abs_diff:.6e}")

        print("\nRelative Differences:")
        print(f"  Max:  {max_rel_diff:.6e}")
        print(f"  Mean: {mean_rel_diff:.6e}")

        # Tolerance check
        atol = 1e-2  # Absolute tolerance for float16
        rtol = 1e-2  # Relative tolerance

        passed = torch.allclose(cpu_output, spyre_output_cpu, atol=atol, rtol=rtol)

        print("\nValidation Result:")
        print(f"  Tolerance: atol={atol}, rtol={rtol}")
        print(f"  Test PASSED: {passed}")

        if not passed:
            print("\n  WARNING: Outputs do not match within tolerance!")
            max_diff_idx = torch.argmax(abs_diff.flatten())
            max_diff_coords = torch.unravel_index(max_diff_idx, abs_diff.shape)
            print(f"\n  Largest difference at coordinates: {max_diff_coords}")
            print(f"    CPU value:   {cpu_output[max_diff_coords].item():.6f}")
            print(f"    Spyre value: {spyre_output_cpu[max_diff_coords].item():.6f}")
        else:
            print("\n  ✓ CPU and Spyre outputs match!")
            print("  ✓ 1D indirect access working correctly!")

except Exception as e:
    print(f"\nERROR during Spyre execution: {e}")
    import traceback

    traceback.print_exc()
