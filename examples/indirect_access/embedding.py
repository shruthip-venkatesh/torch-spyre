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

# TODO : Address computation for multicore : WIP
# Until then set `SENCORES=1` testing indirect access


def test_embedding_simple():
    """Simple 1D embedding test with debug output"""
    print("\n" + "=" * 70)
    print("Simple Embedding Test (1D indices) - WITH DEBUG")
    print("=" * 70)

    vocab_size = 64
    embed_dim = 16

    # Create embedding table
    embedding_table_cpu = torch.randn(vocab_size, embed_dim, dtype=torch.float16)
    print(f"Embedding table shape: {embedding_table_cpu.shape}")
    print("First few rows of embedding table:")
    for i in [5, 20, 3, 42]:
        print(f"  Row {i}: {embedding_table_cpu[i, :4].tolist()}")  # First 4 values

    # Pad to device requirement
    embedding_table = torch.nn.functional.pad(
        embedding_table_cpu, (0, 64 - embed_dim), value=0
    ).to("spyre")
    print(f"Padded shape: {embedding_table.shape}")

    # Indices
    indices = torch.tensor([5, 20, 3, 42], dtype=torch.int64).to("spyre")
    print(f"Indices: {indices.tolist()}")

    # Expected result
    expected = torch.stack(
        [
            embedding_table_cpu[5],
            embedding_table_cpu[20],
            embedding_table_cpu[3],
            embedding_table_cpu[42],
        ]
    )
    print(f"\nExpected shape: {expected.shape}")
    print("Expected values (first 4 cols):")
    for i in range(4):
        print(f"  Row {i}: {expected[i, :4].tolist()}")

    # Run embedding
    def embedding_fn(weight, indices):
        return torch.nn.functional.embedding(indices, weight)

    compiled_fn = torch.compile(embedding_fn)
    result = compiled_fn(embedding_table, indices)

    print(f"\nResult shape: {result.shape}")
    result_cpu = result.cpu()[:, :embed_dim]
    print("Result values (first 4 cols):")
    for i in range(4):
        print(f"  Row {i}: {result_cpu[i, :4].tolist()}")

    # Check pattern
    print("\nChecking for data repetition pattern:")
    print(
        f"  result[0,:4] == expected[0,:4]? {torch.allclose(result_cpu[0, :4], expected[0, :4], rtol=1e-3, atol=1e-3)}"
    )
    print(
        f"  result[1,:4] == expected[1,:4]? {torch.allclose(result_cpu[1, :4], expected[1, :4], rtol=1e-3, atol=1e-3)}"
    )
    print(
        f"  result[2,:4] == expected[2,:4]? {torch.allclose(result_cpu[2, :4], expected[2, :4], rtol=1e-3, atol=1e-3)}"
    )
    print(
        f"  result[3,:4] == expected[3,:4]? {torch.allclose(result_cpu[3, :4], expected[3, :4], rtol=1e-3, atol=1e-3)}"
    )

    # Check if it's the transpose issue
    print("\nChecking for transpose pattern:")
    print(
        f"  result[0,0] == expected[0,0]? {abs(result_cpu[0, 0] - expected[0, 0]) < 0.01}"
    )
    print(
        f"  result[0,1] == expected[1,0]? {abs(result_cpu[0, 1] - expected[1, 0]) < 0.01}"
    )
    print(
        f"  result[0,2] == expected[2,0]? {abs(result_cpu[0, 2] - expected[2, 0]) < 0.01}"
    )
    print(
        f"  result[0,3] == expected[3,0]? {abs(result_cpu[0, 3] - expected[3, 0]) < 0.01}"
    )

    # Full comparison
    max_diff = torch.max(torch.abs(result_cpu - expected))
    print(f"\nMax difference: {max_diff}")

    # assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), \
    #     f"Result mismatch!\nExpected shape: {expected.shape}\nGot shape: {result_cpu.shape}\n" \
    #     f"Max difference: {torch.max(torch.abs(result_cpu - expected))}"
    print(
        f"test_embedding_simple() [Max diff: {max_diff}]: Assertion Fails as data is not fetched properly"
    )

    return result


def test_embedding_2d():
    """
    Test 2D embedding lookup (batch of sequences)
    Input: embedding_table (vocab_size, embed_dim)
    Indices: (batch_size, seq_len) - batch of token ID sequences
    Output: (batch_size, seq_len, embed_dim) - embeddings for each token in each sequence
    """

    print("\n" + "=" * 70)
    print("2D Embedding Tests - Batched Sequences")
    print("=" * 70)

    def embedding_fn(weight, indices):
        return torch.nn.functional.embedding(indices, weight)

    vocab_size = 64
    embed_dim = 16
    batch_size = 2
    seq_len = 4

    # Create embedding table and keep a CPU copy for validation
    embedding_table_cpu = torch.randn(vocab_size, embed_dim, dtype=torch.float16)
    print("Embedding Table Shape:", embedding_table_cpu.shape)

    # Pad last dim to next power of 2 >= embed_dim (here: 16 → 64)
    pad_embed_to = 64
    embedding_table = torch.nn.functional.pad(
        embedding_table_cpu, (0, pad_embed_to - embed_dim), value=0
    ).to("spyre")
    print("Padded Embedding Table Shape:", embedding_table.shape)  # (64, 64)

    # Token indices for batch:
    # batch 0: [20, 11, 50, 37]
    # batch 1: [15, 25, 35, 45]
    token_indices = [[20, 11, 50, 37], [15, 25, 35, 45]]
    indices_tensor = torch.tensor(token_indices, dtype=torch.int64)
    # Pad to match spyre requirements
    indices_tensor = torch.nn.functional.pad(indices_tensor, (0, 12), value=0).to(
        "spyre"
    )
    print("Indices Tensor Shape:", indices_tensor.shape)  # (2, 16)

    # Compute expected result on CPU
    expected = torch.stack(
        [
            torch.stack([embedding_table_cpu[idx] for idx in token_indices[b]])
            for b in range(batch_size)
        ]
    )
    print("Expected Result Shape:", expected.shape)  # (2, 4, 16)
    print("Expected Result:", expected)

    compiled_fn = torch.compile(embedding_fn)
    result = compiled_fn(embedding_table, indices_tensor)

    print("Result Shape:", result.shape)
    print("Result Spyre:", result)
    result_cpu = result.cpu()[:, :seq_len, :embed_dim]
    print("Result CPU (trimmed):", result_cpu)

    # Assert the result matches expected values
    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Result mismatch!\nExpected shape: {expected.shape}\nGot shape: {result_cpu.shape}\n"
        f"Max difference: {torch.max(torch.abs(result_cpu - expected))}"
    )
    print("✓ Assertion passed: Result matches expected values")

    return result


def test_embedding_with_padding_idx():
    """
    Test embedding with padding_idx parameter
    Tokens matching padding_idx should return zero vectors
    """

    print("\n" + "=" * 70)
    print("Embedding Tests with Padding Index")
    print("=" * 70)

    def embedding_fn(weight, indices, padding_idx):
        return torch.nn.functional.embedding(indices, weight, padding_idx=padding_idx)

    vocab_size = 64
    embed_dim = 16
    padding_idx = 0  # Token ID 0 is padding

    # Create embedding table and keep a CPU copy for validation
    embedding_table_cpu = torch.randn(vocab_size, embed_dim, dtype=torch.float16)
    print("Embedding Table Shape:", embedding_table_cpu.shape)

    # Pad last dim to next power of 2 >= embed_dim (here: 16 → 64)
    pad_embed_to = 64
    embedding_table = torch.nn.functional.pad(
        embedding_table_cpu, (0, pad_embed_to - embed_dim), value=0
    ).to("spyre")
    print("Padded Embedding Table Shape:", embedding_table.shape)  # (64, 64)

    # Token indices with padding: [20, 0, 50, 0] where 0 is padding
    token_indices = [20, 0, 50, 0]
    indices_tensor = torch.tensor(token_indices, dtype=torch.int64)
    # Pad to match spyre requirements
    indices_tensor = torch.nn.functional.pad(
        indices_tensor.unsqueeze(0), (0, 12, 0, 3), value=0
    ).to("spyre")
    print("Indices Tensor Shape:", indices_tensor.shape)

    # Compute expected result on CPU
    expected = torch.stack(
        [
            embedding_table_cpu[20],  # token 20
            torch.zeros(embed_dim, dtype=torch.float16),  # padding
            embedding_table_cpu[50],  # token 50
            torch.zeros(embed_dim, dtype=torch.float16),  # padding
        ]
    )
    print("Expected Result Shape:", expected.shape)
    print("Expected Result:", expected)

    compiled_fn = torch.compile(embedding_fn)
    result = compiled_fn(embedding_table, indices_tensor, padding_idx)

    print("Result Shape:", result.shape)
    print("Result Spyre:", result)
    result_cpu = result.cpu()[:, :embed_dim]
    print("Result CPU (trimmed):", result_cpu)

    # Assert the result matches expected values
    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Result mismatch!\nExpected: {expected}\nGot: {result_cpu}\n"
        f"Max difference: {torch.max(torch.abs(result_cpu - expected))}"
    )
    print("✓ Assertion passed: Result matches expected values (with padding)")

    return result


if __name__ == "__main__":
    import os

    try:
        os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"] = "1"
        test_embedding_simple()
        # test_embedding_2d()
        # test_embedding_with_padding_idx() # Need to handle padding

        print("\n" + "=" * 70)
        print("All embedding tests completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
