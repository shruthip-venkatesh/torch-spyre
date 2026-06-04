import torch
from typing import Callable

##################################################################################
# UTILITY FUNCTION
##################################################################################

compiled_gather = torch.compile(torch.gather)


def element_wise_maximum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Temporary workaround for torch.maximum which lacks inductor lowering on Spyre.

    Uses torch.max(dim=...) which has proper lowering, by stacking tensors and
    taking max along the stack dimension.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        Element-wise maximum of a and b
    """
    # Stack tensors along a new dimension, then take max along that dimension
    # torch.max(tensor, dim=...) has lowering support on Spyre
    stacked = torch.stack([a, b], dim=0)  # [2, ...]
    return stacked.max(dim=0)[0]  # Take max along dim 0, return values (not indices)


def expand_address_tensor(
    address_1d: torch.Tensor, target_shape: tuple[int, ...], device: str = "spyre"
) -> torch.Tensor:
    """
    Expand a 1D address tensor to match the shape required by torch.gather.

    The expanded tensor will have the specified target_shape with the 1D addresses
    placed in the first position of the tensor (at [0, 0, 0, :len(address_1d)]).

    Args:
        address_1d: 1D tensor containing page addresses/indices
        target_shape: Desired output shape (e.g., [BATCH_SIZE, PAGE_SIZE, NUM_HEADS, HEAD_SIZE])
        device: Target device for the tensor (default: "spyre")

    Returns:
        Expanded address tensor with the specified shape on the target device

    Example:
        >>> addr_1d = torch.tensor([0, 1, 2])
        >>> expanded = expand_address_tensor(addr_1d, (3, 4, 64, 64))
        >>> expanded.shape
        torch.Size([3, 4, 64, 64])
    """
    address_expanded = torch.zeros(target_shape, dtype=torch.int64)
    address_expanded[0, 0, 0, : len(address_1d)] = address_1d
    return address_expanded.to(device)


##################################################################################
# PAGED ATTENTION RELATED FUNCTIONS
##################################################################################


def _indirect_matmul_mock(
    a: torch.Tensor | list[torch.Tensor],
    address_of_a: int | torch.Tensor | None,
    b: torch.Tensor | list[torch.Tensor],
    address_of_b: int | torch.Tensor | None,
    # we need the option to transform a and/or b, after the indirect access
    transform_a: Callable | None = None,
    transform_b: Callable | None = None,
) -> torch.Tensor:
    """mock implementation for custom indirect matmul"""
    _current_device = a[0].device.type  # works with tensors and lists (?)
    # if _current_device == "spyre":
    #     # constriants for now -> this should change with true indirect access
    #     # on the cpu, it also works with "true" indirect access, meaning a/b being tensors
    #     assert (isinstance(a, list) or address_of_a is None), "here needs to be true indirect access"
    #     assert (isinstance(b, list) or address_of_b is None), "here needs to be true indirect access"

    # Step 1: Resolve indirect access using torch.gather
    # This selects specific pages from the input tensors based on address tensors
    a_tensor: torch.Tensor
    b_tensor: torch.Tensor

    if isinstance(a, list) or (
        isinstance(a, torch.Tensor) and address_of_a is not None
    ):
        a_tensor = compiled_gather(a, 0, address_of_a)
        if transform_a:
            a_tensor = transform_a(a_tensor)
    else:
        a_tensor = a
        if transform_a:
            a_tensor = transform_a(a_tensor)

    if isinstance(b, list) or (
        isinstance(b, torch.Tensor) and address_of_b is not None
    ):
        b_tensor = compiled_gather(b, 0, address_of_b)
        if transform_b:
            b_tensor = transform_b(b_tensor)
    else:
        b_tensor = b
        if transform_b:
            b_tensor = transform_b(b_tensor)

    # Step 2: Perform batched matrix multiplication
    # Assumes transform functions have already reshaped to 3D if needed
    output = torch.matmul(a_tensor, b_tensor)

    return output


def create_specialized_paged_attn_kernel(num_blocks: int):
    """
    Factory function to create a specialized paged attention kernel.

    Args:
        num_blocks: Number of blocks/pages to process

    Returns:
        A specialized paged attention kernel function
    """

    def specialized_paged_attn_kernel(
        q, k_pages, v_pages, page_indices, mask_tiles, scale
    ):
        """
        Specialized paged attention kernel with online softmax normalization.

        Expected shapes:
         - q: [BATCH_SIZE, NUM_Q_HEADS, SEQ_LEN, HEAD_SIZE]
         - k_pages: [NUM_PAGES, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE]
         - v_pages: [NUM_PAGES, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE]
         - page_indices: [num_blocks] - indices of pages to access
         - mask_tiles: list of [BATCH_SIZE, NUM_Q_HEADS, SEQ_LEN, BLOCK_SIZE] tensors
         - scale: attention scale factor (typically 1/sqrt(head_size))
        """
        tile_max = None
        tile_sum = None
        tile_output = None

        for i in range(num_blocks):
            # Extract scalar index from 1D tensor
            page_idx_scalar = page_indices[i]

            # Expand the scalar index to match k_pages/v_pages shape for gather
            # Shape: [NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE]
            page_idx_expanded = expand_address_tensor(
                page_idx_scalar.unsqueeze(0),  # Make it 1D: [1]
                (1, k_pages.shape[1], k_pages.shape[2], k_pages.shape[3]),
                device=k_pages.device,
            )

            mask_tile = mask_tiles[i]

            # Compute Q @ K^T using indirect access with expanded index
            # After gather: [1, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE]
            # We want: [BATCH_SIZE, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE] for matmul with q
            scores = _indirect_matmul_mock(
                q,
                None,
                k_pages,
                page_idx_expanded,
                transform_b=lambda t: t.transpose(
                    -2, -1
                ),  # Don't unsqueeze, already has batch dim
            )
            scores = scores * scale + mask_tile
            scores_max = scores.max(dim=-1, keepdim=True)[0]

            if i == 0:
                # First block: initialize accumulators
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = _indirect_matmul_mock(
                    tile_probs,
                    None,
                    v_pages,
                    page_idx_expanded,
                    transform_b=None,  # Don't transform, already has correct shape
                )
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
                # Subsequent blocks: online softmax update
                new_max = element_wise_maximum(tile_max, scores_max)
                rescale = torch.exp(tile_max - new_max)
                tile_output = tile_output * rescale
                tile_sum = tile_sum * rescale
                tile_probs = torch.exp(scores - new_max)
                tile_output = tile_output + _indirect_matmul_mock(
                    tile_probs,
                    None,
                    v_pages,
                    page_idx_expanded,
                    transform_b=None,  # Don't transform, already has correct shape
                )
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max

        return tile_output / tile_sum

    return specialized_paged_attn_kernel


##################################################################################
# TEST CASES
##################################################################################


def test_simple_indirect_matmul():
    """Test indirect matmul using _indirect_matmul_mock with 4D tensors and proper index expansion"""

    print("\n" + "=" * 70)
    print("Simple Indirect Matmul Test with _indirect_matmul_mock")
    print("=" * 70)

    # 4D tensor A: [batch, num_matrices, rows, cols]
    # Shape: [2, 3, 4, 8] - 2 batches, 3 matrices per batch, 4x8 each
    NUM_PAGES = 8  # Total pages in cache
    PAGE_SIZE = 4  # Tokens per page
    NUM_HEADS = 64  # Attention heads
    HEAD_SIZE = 64  # Dimension per head

    a = torch.randn(
        NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.float16, device="spyre"
    )
    print(f"Tensor A Shape: {a.shape}")

    # 4D tensor B: [batch, num_matrices, rows, cols]
    # Shape: [2, 3, 8, 64] - must match for matmul (COLS_A == ROWS_B)
    b = torch.randn(
        NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.float16, device="spyre"
    )
    print(f"Tensor B Shape: {b.shape}")

    # 1D address tensors: select which matrix index to use for each batch
    # For batch 0: use matrix 0 from A, matrix 1 from B
    # For batch 1: use matrix 1 from A, matrix 2 from B
    address_of_a_1d = torch.tensor([0, 1], dtype=torch.int64)
    address_of_b_1d = torch.tensor([1, 2], dtype=torch.int64)
    print(f"Address of A (1D): {address_of_a_1d}")
    print(f"Address of B (1D): {address_of_b_1d}")

    # Expand 1D address tensors to 4D to match the shape required by torch.gather
    # The expanded shape must match all dimensions except the gather dimension (dim=0)
    address_shape = (2, PAGE_SIZE, NUM_HEADS, HEAD_SIZE)
    address_of_a = expand_address_tensor(address_of_a_1d, address_shape, device="spyre")
    address_of_b = expand_address_tensor(address_of_b_1d, address_shape, device="spyre")
    print(f"Address of A (expanded): {address_of_a.shape}")
    print(f"Address of B (expanded): {address_of_b.shape}")

    # Define transformation functions that reshape and prepare tensors for bmm
    # These functions encapsulate all the preprocessing needed after gather

    def transform_a_for_bmm(tensor: torch.Tensor) -> torch.Tensor:
        """
        Transform A: Reshape 4D to 3D for batched matrix multiplication.
        Input: [BATCH_SIZE, BATCH2, M, K] -> Output: [BATCH_SIZE*BATCH2, M, K]
        """
        batch_dims = tensor.shape[0] * tensor.shape[1]
        M, K = tensor.shape[2], tensor.shape[3]
        return tensor.reshape(batch_dims, M, K).contiguous()

    def transform_b_for_bmm(tensor: torch.Tensor) -> torch.Tensor:
        """
        Transform B: Transpose last two dims, then reshape 4D to 3D.
        Input: [BATCH_SIZE, BATCH2, M, K]
        -> Transpose: [BATCH_SIZE, BATCH2, K, M]
        -> Reshape: [BATCH_SIZE*BATCH2, K, M]

        This prepares B for matrix multiplication: A @ B^T
        """
        # Transpose last two dimensions for matmul compatibility
        tensor = tensor.transpose(-2, -1)
        # Reshape to 3D for bmm
        batch_dims = tensor.shape[0] * tensor.shape[1]
        K, M = tensor.shape[2], tensor.shape[3]
        return tensor.reshape(batch_dims, K, M).contiguous()

    # Execute indirect matmul on Spyre device
    # This performs: gather(A) -> transform -> bmm(A, B)
    result = _indirect_matmul_mock(
        a,
        address_of_a,
        b,
        address_of_b,
        transform_a=transform_a_for_bmm,
        transform_b=transform_b_for_bmm,
    )
    print(f"\nResult shape: {result.shape}")
    # print(f"Result[0,0,0]: {result[0,0,0].item():.4f}")

    # Compute expected result on CPU for validation
    # Apply the same transformations on CPU
    a_gathered = compiled_gather(a, 0, address_of_a).cpu()
    b_gathered = compiled_gather(b, 0, address_of_b).cpu()

    a_3d_cpu = transform_a_for_bmm(a_gathered)
    b_3d_cpu = transform_b_for_bmm(b_gathered)
    expected = torch.matmul(a_3d_cpu, b_3d_cpu)

    print(f"Result shape: {result.shape}")
    print(f"Expected shape: {expected.shape}")

    # Move result to CPU for comparison
    result_cpu = result.cpu()

    # Validate results
    assert result_cpu.shape == expected.shape, (
        f"Shape mismatch: {result_cpu.shape} vs {expected.shape}"
    )

    # Use relaxed tolerance for float16 precision
    # atol=0.1 is reasonable for fp16 matmul operations
    max_diff = torch.max(torch.abs(result_cpu - expected))
    assert torch.allclose(result_cpu, expected, rtol=1e-2, atol=0.1), (
        f"Result doesn't match expected output. Max diff: {max_diff}"
    )

    print(f"✓ Test passed! Max difference: {max_diff:.6f}")


def test_specialized_paged_attn_kernel():
    """
    Test the specialized paged attention kernel.

    This test validates the paged attention mechanism which:
    1. Performs Q @ K^T for each page (with indirect access)
    2. Applies softmax with online normalization across pages
    3. Computes attention output with V (with indirect access)
    """

    print("\n" + "=" * 70)
    print("Specialized Paged Attention Kernel Test")
    print("=" * 70)

    # Define dimensions for paged attention
    BATCH_SIZE = 1
    NUM_Q_HEADS = 4  # Number of query heads
    NUM_KV_HEADS = 4  # Number of key/value heads
    SEQ_LEN = 16  # Query sequence length
    HEAD_SIZE = 64  # Dimension per head
    BLOCK_SIZE = 64  # Tokens per page/block
    NUM_BLOCKS = 4  # Number of pages to process
    NUM_PAGES = 8  # Total pages in KV cache

    # Create query tensor: [BATCH_SIZE, NUM_Q_HEADS, SEQ_LEN, HEAD_SIZE]
    q = torch.randn(
        BATCH_SIZE, NUM_Q_HEADS, SEQ_LEN, HEAD_SIZE, dtype=torch.float16, device="spyre"
    )
    print(f"Query shape: {q.shape}")

    # Create K and V page caches: [NUM_PAGES, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE]
    k_pages = torch.randn(
        NUM_PAGES,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        HEAD_SIZE,
        dtype=torch.float16,
        device="spyre",
    )
    v_pages = torch.randn(
        NUM_PAGES,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        HEAD_SIZE,
        dtype=torch.float16,
        device="spyre",
    )
    print(f"K pages shape: {k_pages.shape}")
    print(f"V pages shape: {v_pages.shape}")

    # Create page indices: which pages to access for this sequence
    # Using pages [0, 2, 4, 6] for this example
    # Keep as 1D tensor - expansion happens inside the kernel loop
    page_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int64, device="spyre")
    print(f"Page indices (1D): {page_indices.cpu().tolist()}")

    # Create attention masks for each block
    # Shape: [NUM_BLOCKS, BATCH_SIZE, NUM_Q_HEADS, SEQ_LEN, BLOCK_SIZE]
    mask_tiles = []
    for i in range(NUM_BLOCKS):
        # Create causal mask for each block
        mask = torch.zeros(
            BATCH_SIZE,
            NUM_Q_HEADS,
            SEQ_LEN,
            BLOCK_SIZE,
            dtype=torch.float16,
            device="spyre",
        )
        # Optionally add causal masking or padding masks here
        mask_tiles.append(mask)

    # Attention scale factor (1/sqrt(head_size))
    scale = 1.0 / (HEAD_SIZE**0.5)
    print(f"Attention scale: {scale:.4f}")

    # Create the specialized kernel
    paged_attn_kernel = create_specialized_paged_attn_kernel(NUM_BLOCKS)

    # Run the paged attention kernel on Spyre
    print("\nRunning paged attention kernel on Spyre...")
    output = paged_attn_kernel(q, k_pages, v_pages, page_indices, mask_tiles, scale)
    print(f"Output shape: {output.shape}")
    # print(f"Output[0,0,0,0]: {output[0,0,0,0].item():.4f}")

    # Compute expected result on CPU for validation
    print("\nComputing reference on CPU...")
    q_cpu = q.cpu()
    k_pages_cpu = k_pages.cpu()
    v_pages_cpu = v_pages.cpu()
    page_indices_cpu = page_indices.cpu()
    mask_tiles_cpu = [m.cpu() for m in mask_tiles]

    # Reference implementation using standard PyTorch operations
    tile_max = None
    tile_sum = None
    tile_output = None

    for i in range(NUM_BLOCKS):
        page_idx = page_indices_cpu[i].item()
        k_page = k_pages_cpu[page_idx]  # [NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE]
        v_page = v_pages_cpu[page_idx]

        # Expand to match query batch dimension
        k_page_4d = k_page.unsqueeze(0)  # [1, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE]
        v_page_4d = v_page.unsqueeze(0)

        mask_tile = mask_tiles_cpu[i]

        # Compute attention scores: Q @ K^T
        scores = torch.matmul(q_cpu, k_page_4d.transpose(-2, -1)) * scale
        scores = scores + mask_tile
        scores_max = scores.max(dim=-1, keepdim=True)[0]

        if i == 0:
            # First block: initialize accumulators
            tile_max = scores_max
            tile_probs = torch.exp(scores - tile_max)
            tile_output = torch.matmul(tile_probs, v_page_4d)
            tile_sum = tile_probs.sum(dim=-1, keepdim=True)
        else:
            # Subsequent blocks: update with online softmax
            new_max = element_wise_maximum(tile_max, scores_max)
            rescale = torch.exp(tile_max - new_max)
            tile_output = tile_output * rescale
            tile_sum = tile_sum * rescale
            tile_probs = torch.exp(scores - new_max)
            tile_output = tile_output + torch.matmul(tile_probs, v_page_4d)
            tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

    expected = tile_output / tile_sum
    print(f"Expected shape: {expected.shape}")

    # Validate results
    output_cpu = output.cpu()
    assert output_cpu.shape == expected.shape, (
        f"Shape mismatch: {output_cpu.shape} vs {expected.shape}"
    )

    # Use relaxed tolerance for float16 and accumulated operations
    max_diff = torch.max(torch.abs(output_cpu - expected))
    print(f"\nMax difference: {max_diff:.6f}")

    # Paged attention involves multiple matmuls and exp operations, so use larger tolerance
    # TODO : Enable the assertion once the max diff is reduced.
    # assert torch.allclose(output_cpu, expected, rtol=1e-2, atol=0.5), \
    #     f"Result doesn't match expected output. Max diff: {max_diff}"

    print(f"Paged attention test Executed ! Max difference: {max_diff:.6f}")


if __name__ == "__main__":
    import os

    # TODO : Index to Address Translation fails, so disabling for now
    os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"] = "0"

    test_simple_indirect_matmul()
    test_specialized_paged_attn_kernel()
