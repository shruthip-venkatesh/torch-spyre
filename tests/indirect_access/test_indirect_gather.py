import torch
import os

# TODO : Address computation for multicore : WIP
# Until then set `SENCORES=1` testing indirect access

def test_gather_indirect_1d():

    print("\n" + "=" * 70)
    print("1D Output Tensor Tests for Indirect Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.ops.spyre.indirect_gather(input, dim, index)

    # Create input tensor and keep a CPU copy for validation
    input_tensor_cpu = torch.randn(10, dtype=torch.float16)
    print("Input Tensor :", input_tensor_cpu)
    
    # Prepare input for spyre device
    input_tensor = torch.nn.functional.pad(input_tensor_cpu.reshape(10,1), (0, 63), value=0).to("spyre")
    print("Input Tensor Shape:", input_tensor.shape)
    
    # Index tensor: gather indices [0, 5, 2, 1] from first row
    gather_indices = [0, 5, 2, 1]
    index_tensor = torch.tensor([gather_indices], dtype=torch.int64)
    index_tensor = torch.nn.functional.pad(index_tensor, (0, 0, 0, 3), value=0).to("spyre")
    print("Index Tensor:", index_tensor)
    print("Index Tensor Shape:", index_tensor.shape)

    # Compute expected result on CPU
    # The gather should select elements at indices [0, 5, 2, 1] from input_tensor_cpu
    expected = torch.stack([
        input_tensor_cpu[0],  # index 0
        input_tensor_cpu[5],  # index 5
        input_tensor_cpu[2],  # index 2
        input_tensor_cpu[1]   # index 1
    ])
    print("Expected Result:", expected)

    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)
    
    print("Result Shape:", result.shape)
    print("Result Spyre:", result)
    result_cpu = result.cpu()[:,0]
    print("Result CPU :", result_cpu)
    
    # Assert the result matches expected values
    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), \
        f"Result mismatch!\nExpected: {expected}\nGot: {result_cpu}"
    print("✓ Assertion passed: Result matches expected values")
    
    return result

def test_gather_indirect_2d():

    print("\n" + "=" * 70)
    print("2D Output Tensor Tests for Indirect Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.ops.spyre.indirect_gather(input, dim, index)

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
    input_tensor = torch.nn.functional.pad(input_tensor_cpu, (0, pad_embed_to - embed_dim), value=0).to("spyre")
    print("Input Tensor Shape:", input_tensor.shape)  # (64, 64)

    # Index: gather 4 rows from the vocab at indices [20, 11, 50, 37]
    gather_indices = [20, 11, 50, 37]
    index_tensor = torch.tensor([gather_indices], dtype=torch.int64)
    index_tensor = torch.nn.functional.pad(index_tensor, (0, 12, 0, 3), value=0).to("spyre")
    print("Index Tensor Shape:", index_tensor.shape)  # (4, 32)

    # Compute expected result on CPU
    # The gather should select rows at indices [20, 11, 50, 37] from input_tensor_cpu
    expected = torch.stack([
        input_tensor_cpu[20],  # row 20
        input_tensor_cpu[11],  # row 11
        input_tensor_cpu[50],  # row 50
        input_tensor_cpu[37]   # row 37
    ])
    print("Expected Result Shape:", expected.shape)
    print("Expected Result:", expected)

    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)
    result_cpu = result.cpu()

    print("Result Shape:", result.shape)
    print("Result Spyre:", result)
    print("Result (CPU):", result_cpu)
    
    # Assert the result matches expected values
    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), \
        f"Result mismatch!\nExpected shape: {expected.shape}\nGot shape: {result_cpu.shape}\n" \
        f"Max difference: {torch.max(torch.abs(result_cpu - expected))}"
    print("✓ Assertion passed: Result matches expected values")

    return result

### NOT WORKING ###
def test_gather_indirect_3d():

    print("\n" + "=" * 70)
    print("3D Output Tensor Tests for Indirect Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.ops.spyre.indirect_gather(input, dim, index)

    batch_size = 1  # Changed to 1 for single batch test
    vocab_size = 64
    embed_dim  = 16

    # 3D input: (batch_size, vocab_size, embed_dim) — batched embedding table
    # Keep a CPU copy for validation
    input_tensor_cpu = torch.randn(batch_size, vocab_size, embed_dim, dtype=torch.float16)

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
    expected = torch.stack([
        torch.stack([
            input_tensor_cpu[b, idx]        # shape: (embed_dim,)
            for idx in gather_indices[b]
        ])                                  # shape: (num_indices, embed_dim)
        for b in range(batch_size)
    ])                                      # shape: (batch_size, num_indices, embed_dim)

    print("Expected Result Shape:", expected.shape)   # (1, 4, 16)
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
    assert torch.allclose(result_cpu_trimmed, expected, rtol=1e-3, atol=1e-3), \
        f"Result mismatch!\nExpected shape: {expected.shape}\nGot shape: {result_cpu_trimmed.shape}\n" \
        f"Max difference: {torch.max(torch.abs(result_cpu_trimmed - expected))}"
    print("✓ Assertion passed: Result matches expected values")

    return result

if __name__ == "__main__":

    try:
        test_gather_indirect_1d()
        test_gather_indirect_2d()
        #test_gather_indirect_3d() # Values are wrongly fetched
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()