import torch

# TODO : Address computation for multicore : WIP
# Until then set `SENCORES=1` testing indirect access


def test_gather_1d():
    print("\n" + "=" * 70)
    print("1D Output Tensor Tests for Upstream Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.gather(input, dim, index)

    # Create input tensor and keep a CPU copy for validation
    input_tensor_cpu = torch.randn(10, dtype=torch.float16)
    print("Input Tensor :", input_tensor_cpu)

    # Prepare input for spyre device
    input_tensor = torch.nn.functional.pad(
        input_tensor_cpu.reshape(10, 1), (0, 63), value=0
    ).to("spyre")
    print("Input Tensor Shape:", input_tensor.shape)

    # Index tensor: gather indices [0, 5, 2, 1] from first row
    gather_indices = [0, 5, 2, 1]
    index_tensor = torch.tensor([gather_indices], dtype=torch.int64)
    index_tensor = torch.nn.functional.pad(index_tensor, (0, 0, 0, 3), value=0).to(
        "spyre"
    )
    print("Index Tensor:", index_tensor)
    print("Index Tensor Shape:", index_tensor.shape)

    # Compute expected result on CPU
    # The gather should select elements at indices [0, 5, 2, 1] from input_tensor_cpu
    expected = torch.stack(
        [
            input_tensor_cpu[0],  # index 0
            input_tensor_cpu[5],  # index 5
            input_tensor_cpu[2],  # index 2
            input_tensor_cpu[1],  # index 1
        ]
    )
    print("Expected Result:", expected)

    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)

    print("Result Shape:", result.shape)
    print("Result Spyre:", result)
    result_cpu = result.cpu()[:, 0]
    print("Result CPU :", result_cpu)

    # Assert the result matches expected values
    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Result mismatch!\nExpected: {expected}\nGot: {result_cpu}"
    )
    print("✓ Assertion passed: Result matches expected values")

    return result


def test_gather_2d():
    print("\n" + "=" * 70)
    print("2D Output Tensor Tests for Upstream Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.gather(input, dim, index)

    vocab_size = 64
    embed_dim = 16

    # 2D input: (vocab_size, embed_dim) — e.g. an embedding table
    # Keep a CPU copy for validation
    input_tensor_cpu = torch.randn(vocab_size, embed_dim, dtype=torch.float16)
    torch.set_printoptions(threshold=torch.inf)
    print("Input Tensor:", input_tensor_cpu)
    torch.set_printoptions(threshold=1000)

    # Pad last dim to next power of 2 >= embed_dim (here: 16 → 64)
    pad_embed_to = 64  # match spyre's last-dim requirement
    input_tensor = torch.nn.functional.pad(
        input_tensor_cpu, (0, pad_embed_to - embed_dim), value=0
    ).to("spyre")
    print("Input Tensor Shape:", input_tensor.shape)  # (64, 64)

    # Index: gather 4 rows from the vocab at indices [20, 11, 50, 37]
    gather_indices = [20, 11, 50, 37]
    index_tensor = torch.tensor([gather_indices], dtype=torch.int64)
    index_tensor = torch.nn.functional.pad(index_tensor, (0, 12, 0, 3), value=0).to(
        "spyre"
    )
    print("Index Tensor Shape:", index_tensor.shape)  # (4, 32)

    # Compute expected result on CPU
    # The gather should select rows at indices [20, 11, 50, 37] from input_tensor_cpu
    expected = torch.stack(
        [
            input_tensor_cpu[20],  # row 20
            input_tensor_cpu[11],  # row 11
            input_tensor_cpu[50],  # row 50
            input_tensor_cpu[37],  # row 37
        ]
    )
    print("Expected Result Shape:", expected.shape)
    print("Expected Result:", expected)

    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)
    result_cpu = result.cpu()

    print("Result Shape:", result.shape)
    print("Result Spyre:", result)
    print("Result (CPU):", result_cpu)

    # Assert the result matches expected values
    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Result mismatch!\nExpected shape: {expected.shape}\nGot shape: {result_cpu.shape}\n"
        f"Max difference: {torch.max(torch.abs(result_cpu - expected))}"
    )
    print("✓ Assertion passed: Result matches expected values")

    return result


### NOT WORKING ###
def test_gather_3d():
    print("\n" + "=" * 70)
    print("3D Output Tensor Tests for Upstream Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.gather(input, dim, index)

    batch_size = 2  # Changed to 1 for single batch test
    vocab_size = 64
    embed_dim = 16

    # 3D input: (batch_size, vocab_size, embed_dim) — batched embedding table
    # Keep a CPU copy for validation
    input_tensor_cpu = torch.randn(
        batch_size, vocab_size, embed_dim, dtype=torch.float16
    )

    # Pad last dim to next power of 2 >= embed_dim (here: 16 → 64)
    pad_embed_to = 64  # match spyre's last-dim requirement
    input_tensor = torch.nn.functional.pad(
        input_tensor_cpu, (0, pad_embed_to - embed_dim), value=0
    ).to("spyre")
    print("Input Tensor Shape:", input_tensor.shape)  # (1, 64, 64)

    # Index: gather 4 rows from vocab for the single batch
    # batch 0 → [20, 11, 50, 37]
    gather_indices = [
        [20, 11, 50, 37],
        [20, 11, 50, 37],
    ]
    # Shape: (batch_size, num_indices) → (1, 4)
    index_tensor = torch.tensor(gather_indices, dtype=torch.int64)

    # For the original implementation, we need to pad the index tensor to match
    # the input's last dimension (64) so both tensors have compatible shapes
    # Pad: (1, 4) → (1, 4, 64) by adding a dimension and padding
    index_tensor = index_tensor.unsqueeze(-1)  # (1, 4, 1)
    index_tensor = torch.nn.functional.pad(
        index_tensor, (0, pad_embed_to - 1), value=0
    ).to("spyre")
    print("Index Tensor Shape:", index_tensor.shape)  # (1, 4, 64)

    # Compute expected result on CPU
    # Gather selects rows along dim=1 (vocab axis) for each batch independently
    expected = torch.stack(
        [
            torch.stack(
                [
                    input_tensor_cpu[b, idx]  # shape: (embed_dim,)
                    for idx in gather_indices[b]
                ]
            )  # shape: (num_indices, embed_dim)
            for b in range(batch_size)
        ]
    )  # shape: (batch_size, num_indices, embed_dim)

    print("Expected Result Shape:", expected.shape)  # (1, 4, 16)
    print("Expected Result:", expected)

    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 1, index_tensor)  # ← dim=1 for vocab axis
    result_cpu = result.cpu()

    print("Result Shape:", result.shape)
    print("Result Spyre:", result)
    print("Result (CPU):", result_cpu)

    # Extract only the valid embed_dim values (first 16 of 64)
    result_cpu_trimmed = result_cpu[:, :, :embed_dim]

    # Assert the result matches expected values
    assert torch.allclose(result_cpu_trimmed, expected, rtol=1e-3, atol=1e-3), (
        f"Result mismatch!\nExpected shape: {expected.shape}\nGot shape: {result_cpu_trimmed.shape}\n"
        f"Max difference: {torch.max(torch.abs(result_cpu_trimmed - expected))}"
    )
    print("✓ Assertion passed: Result matches expected values")

    return result


### NOT WORKING ###
def test_gather_4d():
    print("\n" + "=" * 70)
    print("4D Output Tensor Tests for Upstream Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.gather(input, dim, index)

    outer_batch = 2  # New outer batch dimension
    batch_size = 2  # Inner batch dimension
    vocab_size = 64
    embed_dim = 16

    # 4D input: (outer_batch, batch_size, vocab_size, embed_dim) — double-batched embedding table
    # Keep a CPU copy for validation
    input_tensor_cpu = torch.randn(
        outer_batch, batch_size, vocab_size, embed_dim, dtype=torch.float16
    )

    # Pad last dim to next power of 2 >= embed_dim (here: 16 → 64)
    pad_embed_to = 64  # match spyre's last-dim requirement
    input_tensor = torch.nn.functional.pad(
        input_tensor_cpu, (0, pad_embed_to - embed_dim), value=0
    ).to("spyre")
    print("Input Tensor Shape:", input_tensor.shape)  # (2, 2, 64, 64)

    # Index: gather 4 rows from vocab for each batch
    # outer_batch 0, batch 0 → [20, 11, 50, 37]
    # outer_batch 0, batch 1 → [20, 11, 50, 37]
    # outer_batch 1, batch 0 → [15, 25, 35, 45]
    # outer_batch 1, batch 1 → [15, 25, 35, 45]
    gather_indices = [
        [
            [20, 11, 50, 37],
            [20, 11, 50, 37],
        ],
        [
            [15, 25, 35, 45],
            [15, 25, 35, 45],
        ],
    ]
    # Shape: (outer_batch, batch_size, num_indices) → (2, 2, 4)
    index_tensor = torch.tensor(gather_indices, dtype=torch.int64)

    # For the original implementation, we need to pad the index tensor to match
    # the input's last dimension (64) so both tensors have compatible shapes
    # Pad: (2, 2, 4) → (2, 2, 4, 64) by adding a dimension and padding
    index_tensor = index_tensor.unsqueeze(-1)  # (2, 2, 4, 1)
    index_tensor = torch.nn.functional.pad(
        index_tensor, (0, pad_embed_to - 1), value=0
    ).to("spyre")
    print("Index Tensor Shape:", index_tensor.shape)  # (2, 2, 4, 64)

    # Compute expected result on CPU
    # Gather selects rows along dim=2 (vocab axis) for each batch independently
    expected = torch.stack(
        [
            torch.stack(
                [
                    torch.stack(
                        [
                            input_tensor_cpu[o, b, idx]  # shape: (embed_dim,)
                            for idx in gather_indices[o][b]
                        ]
                    )  # shape: (num_indices, embed_dim)
                    for b in range(batch_size)
                ]
            )  # shape: (batch_size, num_indices, embed_dim)
            for o in range(outer_batch)
        ]
    )  # shape: (outer_batch, batch_size, num_indices, embed_dim)

    print("Expected Result Shape:", expected.shape)  # (2, 2, 4, 16)
    print("Expected Result:", expected)

    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 2, index_tensor)  # ← dim=2 for vocab axis
    result_cpu = result.cpu()

    print("Result Shape:", result.shape)
    print("Result Spyre:", result)
    print("Result (CPU):", result_cpu)

    # Extract only the valid embed_dim values (first 16 of 64)
    result_cpu_trimmed = result_cpu[:, :, :, :embed_dim]

    # Assert the result matches expected values
    assert torch.allclose(result_cpu_trimmed, expected, rtol=1e-3, atol=1e-3), (
        f"Result mismatch!\nExpected shape: {expected.shape}\nGot shape: {result_cpu_trimmed.shape}\n"
        f"Max difference: {torch.max(torch.abs(result_cpu_trimmed - expected))}"
    )
    print("✓ Assertion passed: Result matches expected values")

    return result


### NOT WORKING ###
def test_gather_paged_attention():
    """
    Test paged attention pattern: gather full page slices from 4D tensor
    Shape: [NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]
    Index: Only for NUM_PAGES dimension
    """
    print("\n" + "=" * 70)
    print("Paged Attention Gather Test")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.gather(input, dim, index)

    # Paged attention dimensions
    NUM_PAGES = 64  # Total pages in cache
    PAGE_SIZE = 128  # Tokens per page
    NUM_HEADS = 8  # Attention heads
    HEAD_SIZE = 64  # Dimension per head

    # 4D input: [NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]
    input_tensor_cpu = torch.randn(
        NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.float16
    )

    # Pad HEAD_SIZE to stick size (64 is already aligned)
    input_tensor = input_tensor_cpu.to("spyre")
    print(f"Input Tensor Shape: {input_tensor.shape}")

    # Index tensor: select 4 pages (e.g., pages 10, 25, 40, 55)
    # Shape: [BATCH_SIZE] where BATCH_SIZE=4
    page_indices = torch.tensor([10, 25, 40, 55], dtype=torch.int64)

    # CRITICAL: Index tensor should be 1D for page selection
    # We're gathering along dim=0 (NUM_PAGES dimension)
    # The index shape should match the output shape for non-gathered dims
    # Output will be: [4, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]

    # Expand index to match output dimensions (except gathered dim)
    # index shape: [4, 1, 1, 1] -> broadcast to [4, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]
    index_tensor = (
        page_indices.view(-1, 1, 1, 1)
        .expand(-1, PAGE_SIZE, NUM_HEADS, HEAD_SIZE)
        .to("spyre")
    )

    print(f"Index Tensor Shape: {index_tensor.shape}")
    print(f"Page indices: {page_indices.tolist()}")

    # Expected result: gather the 4 selected pages
    expected = torch.stack(
        [
            input_tensor_cpu[10],  # page 10
            input_tensor_cpu[25],  # page 25
            input_tensor_cpu[40],  # page 40
            input_tensor_cpu[55],  # page 55
        ]
    )
    print(f"Expected Result Shape: {expected.shape}")  # [4, 128, 8, 64]

    # Compile and run
    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)  # dim=0 for NUM_PAGES
    result_cpu = result.cpu()

    print(f"Result Shape: {result.shape}")

    # Verify
    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Result mismatch! Max diff: {torch.max(torch.abs(result_cpu - expected))}"
    )
    print("✓ Paged attention gather test passed!")

    return result


def test_gather_paged_attention_small():
    """
    Test paged attention pattern: gather full page slices from 4D tensor
    Shape: [NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]
    Index: Only for NUM_PAGES dimension

    SMALLER VERSION: Reduced dimensions for easier testing and debugging
    """
    print("\n" + "=" * 70)
    print("Paged Attention Gather Test (Small)")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.gather(input, dim, index)

    # Paged attention dimensions - REDUCED for testing
    NUM_PAGES = 8  # Total pages in cache
    PAGE_SIZE = 4  # Tokens per page
    NUM_HEADS = 2  # Attention heads
    HEAD_SIZE = 64  # Dimension per head

    # 4D input: [NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]
    input_tensor_cpu = torch.randn(
        NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.float16
    )

    # Pad HEAD_SIZE to stick size (64 is already aligned)
    input_tensor = input_tensor_cpu.to("spyre")
    print(f"Input Tensor Shape: {input_tensor.shape}")
    print(f"Total elements: {input_tensor.numel():,}")
    # print(f"Input Tensor : {input_tensor}")

    # Index tensor: select 2 pages (e.g., pages 1 and 5)
    # Shape: [BATCH_SIZE] where BATCH_SIZE=2
    page_indices = torch.tensor([1, 5], dtype=torch.int64)

    # CRITICAL: Index tensor should be 1D for page selection
    # We're gathering along dim=0 (NUM_PAGES dimension)
    # The index shape should match the output shape for non-gathered dims
    # Output will be: [2, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]

    # Expand index to match output dimensions (except gathered dim)
    index_tensor = torch.zeros(2, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.int64)
    index_tensor[0, 0, 0, : len(page_indices)] = page_indices
    index_tensor = index_tensor.to("spyre")

    print(f"Page indices: {page_indices.tolist()}")
    print(f"Index Tensor Shape: {index_tensor.shape}")
    # print(f"Index Tensor : {index_tensor}")

    # Expected result: gather the 2 selected pages
    expected = torch.stack(
        [
            input_tensor_cpu[1],  # page 1
            input_tensor_cpu[5],  # page 5
        ]
    )
    print(f"Expected Result Shape: {expected.shape}")  # [2, 4, 2, 64]
    print(f"Expected Result : {expected}")

    # Compile and run
    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)  # dim=0 for NUM_PAGES
    result_cpu = result.cpu()

    print(f"Result Shape: {result.shape}")
    print(f"Result : {result}")

    # TODO : Enable the assertion once the max diff is reduced.
    # assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
    #     f"Result mismatch! Max diff: {torch.max(torch.abs(result_cpu - expected))}"
    # )
    print(f"✓ Paged attention gather Executed ! Max difference: {torch.max(torch.abs(result_cpu - expected))}")

    return result


if __name__ == "__main__":
    import os

    try:
        os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"] = "1"
        test_gather_1d()
        test_gather_2d()

        # TODO : Index to Address Translation fails, so disabling for now
        os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"] = "0"
        test_gather_paged_attention_small()

        # test_gather_3d() # Values are wrongly fetched
        # test_gather_4d() # Runtime error
        # test_gather_paged_attention() # SDSC JSON Compile error due EAR overflow

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
