import torch
import math

# Paged Attention Parameters
B = 2  # Batch size
num_heads = 4  # Number of attention heads
seq_len = 16  # Sequence length
head_dim = 8  # Dimension per head
cache_size = 64  # Total KV cache slots
block_size = 4  # Slots per block

print("=" * 80)
print("COMPLETE PAGED ATTENTION WITH GATHER AND SCATTER")
print("=" * 80)

torch.manual_seed(42)

# Input tensors
queries = torch.randn(B, seq_len, num_heads, head_dim, dtype=torch.float16)
new_keys = torch.randn(B, seq_len, num_heads, head_dim, dtype=torch.float16)
new_values = torch.randn(B, seq_len, num_heads, head_dim, dtype=torch.float16)

# KV cache (shared across sequences)
kv_cache_keys = torch.randn(cache_size, num_heads, head_dim, dtype=torch.float16)
kv_cache_values = torch.randn(cache_size, num_heads, head_dim, dtype=torch.float16)

# Block table: maps each sequence to cache blocks
num_blocks_per_seq = seq_len // block_size
block_table = torch.randint(
    0, cache_size // block_size, (B, num_blocks_per_seq), dtype=torch.int64
)

print("\nInput Shapes:")
print(f"  queries:     {queries.shape}")
print(f"  new_keys:    {new_keys.shape}")
print(f"  new_values:  {new_values.shape}")
print(f"  kv_cache:    {kv_cache_keys.shape}")
print(f"  block_table: {block_table.shape}")

# Convert block table to slot indices
offsets = torch.arange(0, block_size, dtype=torch.int64)
slot_idxs = (block_table.unsqueeze(-1) * block_size + offsets).reshape(B, seq_len)

print(f"\nSlot Indices (shape {slot_idxs.shape}):")
for b in range(B):
    print(f"  Batch {b}: {slot_idxs[b].tolist()}")

# ============================================================================
# COMPLETE PAGED ATTENTION KERNEL
# ============================================================================


def gather_from_cache(cache, slot_idxs):
    """Gather operation: Read from cache using slot indices"""
    return cache[slot_idxs]


def scatter_to_cache(cache, slot_idxs, values):
    """Scatter operation: Write to cache using slot indices"""
    # Create a copy to avoid in-place modification issues
    updated_cache = cache.clone()
    # Use index_put_ for scatter operation
    updated_cache.index_put_(
        (slot_idxs.flatten(),), values.reshape(-1, *values.shape[2:])
    )
    return updated_cache


def paged_attention_kernel(
    queries,
    new_keys,
    new_values,
    kv_cache_keys,
    kv_cache_values,
    slot_idxs,
    gather_fn=gather_from_cache,
    scatter_fn=scatter_to_cache,
):
    """
    Complete paged attention with gather and scatter

    Steps:
    1. SCATTER: Write new K/V to cache at specified slots
    2. GATHER: Read K/V from cache for attention computation
    3. ATTENTION: Compute attention scores and output

    Args:
        queries: (B, L, H, D)
        new_keys: (B, L, H, D) - new keys to write to cache
        new_values: (B, L, H, D) - new values to write to cache
        kv_cache_keys: (cache_size, H, D) - key cache
        kv_cache_values: (cache_size, H, D) - value cache
        slot_idxs: (B, L) - cache slot indices

    Returns:
        output: (B, L, H, D) - attention output
        updated_cache_keys: (cache_size, H, D) - updated key cache
        updated_cache_values: (cache_size, H, D) - updated value cache
    """
    B, L, H, D = queries.shape
    scale = 1.0 / math.sqrt(D)

    print("\n  === PAGED ATTENTION KERNEL ===")

    # STEP 1: SCATTER - Write new K/V to cache
    print("  [SCATTER] Writing new K/V to cache slots...")
    updated_cache_keys = scatter_fn(kv_cache_keys, slot_idxs, new_keys)
    updated_cache_values = scatter_fn(kv_cache_values, slot_idxs, new_values)
    print(f"    Cache updated at slots: {slot_idxs.flatten().unique().tolist()}")

    # STEP 2: GATHER - Read K/V from cache
    print("  [GATHER] Reading K/V from cache...")
    gathered_keys = gather_fn(updated_cache_keys, slot_idxs)  # (B, L, H, D)
    gathered_values = gather_fn(updated_cache_values, slot_idxs)  # (B, L, H, D)
    print(f"    Gathered keys shape: {gathered_keys.shape}")
    print(f"    Gathered values shape: {gathered_values.shape}")

    # STEP 3: ATTENTION - Compute attention
    print("  [ATTENTION] Computing attention scores...")

    # Reshape for attention: (B, H, L, D)
    q = queries.transpose(1, 2)
    k = gathered_keys.transpose(1, 2)
    v = gathered_values.transpose(1, 2)

    # Compute attention scores: (B, H, L, L)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention to values: (B, H, L, D)
    output = torch.matmul(attn_weights, v)
    output = output.transpose(1, 2)  # (B, L, H, D)

    print(f"    Output shape: {output.shape}")

    return output, updated_cache_keys, updated_cache_values


# ============================================================================
# CPU REFERENCE EXECUTION
# ============================================================================

print("\n" + "=" * 80)
print("CPU REFERENCE EXECUTION")
print("=" * 80)

cpu_output, cpu_cache_k, cpu_cache_v = paged_attention_kernel(
    queries, new_keys, new_values, kv_cache_keys, kv_cache_values, slot_idxs
)

print("\nCPU Results:")
print(f"  Output shape: {cpu_output.shape}")
print(f"  Updated cache keys shape: {cpu_cache_k.shape}")
print(f"  Updated cache values shape: {cpu_cache_v.shape}")
print(f"  Sample output (batch 0, pos 0, head 0): {cpu_output[0, 0, 0, :4].tolist()}")

# Verify scatter worked
print("\nVerifying SCATTER operation:")
for b in range(min(2, B)):
    slot = slot_idxs[b, 0].item()
    print(f"  Batch {b}, slot {slot}:")
    print(f"    Original cache: {kv_cache_keys[slot, 0, :3].tolist()}")
    print(f"    New key:        {new_keys[b, 0, 0, :3].tolist()}")
    print(f"    Updated cache:  {cpu_cache_k[slot, 0, :3].tolist()}")
    match = torch.allclose(cpu_cache_k[slot], new_keys[b, 0], atol=1e-5)
    print(f"    Match: {match}")

# ============================================================================
# SPYRE DEVICE EXECUTION WITH TORCH.COMPILE
# ============================================================================

print("\n" + "=" * 80)
print("SPYRE DEVICE EXECUTION WITH TORCH.COMPILE")
print("=" * 80)

try:
    if not torch.spyre.is_available():
        print("ERROR: Spyre device is not available!")
    else:
        print("Spyre device is available!")

        # Move tensors to Spyre
        print("\nMoving tensors to Spyre device...")
        queries_spyre = queries.to("spyre")
        new_keys_spyre = new_keys.to("spyre")
        new_values_spyre = new_values.to("spyre")
        kv_cache_keys_spyre = kv_cache_keys.to("spyre")
        kv_cache_values_spyre = kv_cache_values.to("spyre")
        slot_idxs_spyre = slot_idxs.to("spyre")

        print(f"  All tensors moved to: {queries_spyre.device}")

        # Compile gather and scatter functions
        print("\nCompiling gather and scatter functions...")
        compiled_gather = torch.compile(gather_from_cache, backend="inductor")
        compiled_scatter = torch.compile(scatter_to_cache, backend="inductor")

        # Run on Spyre
        print("\nExecuting paged attention on Spyre...")
        spyre_output, spyre_cache_k, spyre_cache_v = paged_attention_kernel(
            queries_spyre,
            new_keys_spyre,
            new_values_spyre,
            kv_cache_keys_spyre,
            kv_cache_values_spyre,
            slot_idxs_spyre,
            compiled_gather,
            compiled_scatter,
        )

        print("\nSpyre Results:")
        print(f"  Output device: {spyre_output.device}")
        print(f"  Output shape: {spyre_output.shape}")

        # Move back to CPU for comparison
        spyre_output_cpu = spyre_output.cpu()
        spyre_cache_k_cpu = spyre_cache_k.cpu()
        spyre_cache_v_cpu = spyre_cache_v.cpu()

        print(
            f"  Sample output (batch 0, pos 0, head 0): {spyre_output_cpu[0, 0, 0, :4].tolist()}"
        )

        # ============================================================================
        # VALIDATION
        # ============================================================================

        print("\n" + "=" * 80)
        print("VALIDATION: CPU vs SPYRE")
        print("=" * 80)

        # Validate output
        output_diff = torch.abs(cpu_output - spyre_output_cpu)
        output_max_diff = output_diff.amax().item()
        output_mean_diff = output_diff.mean().item()

        # Validate cache updates
        cache_k_diff = torch.abs(cpu_cache_k - spyre_cache_k_cpu)
        cache_k_max_diff = cache_k_diff.amax().item()

        cache_v_diff = torch.abs(cpu_cache_v - spyre_cache_v_cpu)
        cache_v_max_diff = cache_v_diff.amax().item()

        print("\nOutput Differences:")
        print(f"  Max:  {output_max_diff:.6e}")
        print(f"  Mean: {output_mean_diff:.6e}")

        print("\nCache Keys Differences:")
        print(f"  Max:  {cache_k_max_diff:.6e}")

        print("\nCache Values Differences:")
        print(f"  Max:  {cache_v_max_diff:.6e}")

        # Tolerance check
        atol, rtol = 1e-2, 1e-2

        output_passed = torch.allclose(
            cpu_output, spyre_output_cpu, atol=atol, rtol=rtol
        )
        cache_k_passed = torch.allclose(
            cpu_cache_k, spyre_cache_k_cpu, atol=atol, rtol=rtol
        )
        cache_v_passed = torch.allclose(
            cpu_cache_v, spyre_cache_v_cpu, atol=atol, rtol=rtol
        )

        all_passed = output_passed and cache_k_passed and cache_v_passed

        print(f"\nValidation Results (atol={atol}, rtol={rtol}):")
        print(f"  ✓ Output:       {'PASS' if output_passed else 'FAIL'}")
        print(f"  ✓ Cache Keys:   {'PASS' if cache_k_passed else 'FAIL'}")
        print(f"  ✓ Cache Values: {'PASS' if cache_v_passed else 'FAIL'}")
        print(
            f"\n  Overall: {'✓ ALL TESTS PASSED!' if all_passed else '✗ SOME TESTS FAILED'}"
        )

        if all_passed:
            print("\n  SUCCESS: Paged attention with gather/scatter working correctly!")
            print("  - GATHER operations validated")
            print("  - SCATTER operations validated")
            print("  - Attention computation validated")

except Exception as e:
    print(f"\nERROR during Spyre execution: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nThis complete paged attention kernel demonstrates:")
print("  1. SCATTER: Writing new K/V to cache using slot indices")
print("     - Operation: cache[slot_idxs] = new_kv")
print("     - Tests indirect write/update to memory")
print()
print("  2. GATHER: Reading K/V from cache using slot indices")
print("     - Operation: kv = cache[slot_idxs]")
print("     - Tests indirect read from memory")
print()
print("  3. ATTENTION: Standard attention computation")
print("     - Scores, softmax, weighted sum")
print()
print("  4. VALIDATION: CPU vs Spyre comparison")
print("     - Validates both gather and scatter operations")
print("     - Ensures numerical correctness")

print("\nKey Operations Tested:")
print("  ✓ Indirect write (scatter): cache[indices] = values")
print("  ✓ Indirect read (gather):  values = cache[indices]")
print("  ✓ Matrix multiplication:   torch.matmul")
print("  ✓ Softmax:                 torch.softmax")
print("  ✓ Transpose:               tensor.transpose")

print("\n" + "=" * 80)
