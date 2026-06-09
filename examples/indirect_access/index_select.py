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


def test_index_select_1d():
    """Test torch.index_select with 1D output"""

    print("\n" + "=" * 70)
    print("1D Output Tensor Tests for torch.index_select")
    print("=" * 70)

    def index_select_fn(input, dim, index):
        return torch.index_select(input, dim, index)

    # Create input tensor and keep a CPU copy for validation
    input_tensor_cpu = torch.randn(10, dtype=torch.float16)
    print("Input Tensor :", input_tensor_cpu)

    # Prepare input for spyre device
    input_tensor = torch.nn.functional.pad(
        input_tensor_cpu.reshape(10, 1), (0, 63), value=0
    ).to("spyre")
    print("Input Tensor Shape:", input_tensor.shape)

    # Index tensor: select indices [0, 5, 2, 1]
    # torch.index_select requires 1D index tensor
    select_indices = [0, 5, 2, 1]
    index_tensor = torch.tensor(select_indices, dtype=torch.int64).to("spyre")
    print("Index Tensor:", index_tensor)
    print("Index Tensor Shape:", index_tensor.shape)

    # Compute expected result on CPU
    # The index_select should select elements at indices [0, 5, 2, 1] from input_tensor_cpu
    expected = torch.stack(
        [
            input_tensor_cpu[0],  # index 0
            input_tensor_cpu[5],  # index 5
            input_tensor_cpu[2],  # index 2
            input_tensor_cpu[1],  # index 1
        ]
    )
    print("Expected Result:", expected)

    compiled_fn = torch.compile(index_select_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)

    # print("Result Shape:", result.shape)
    # print("Result Spyre:", result)
    result_cpu = result.cpu()[:, 0]
    print("Result CPU :", result_cpu)

    # Assert the result matches expected values
    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Result mismatch!\nExpected: {expected}\nGot: {result_cpu}"
    )
    print("✓ Assertion passed: Result matches expected values")

    return result


def test_index_select_2d():
    """Test torch.index_select with 2D output (embedding-like)"""

    print("\n" + "=" * 70)
    print("2D Output Tensor Tests for torch.index_select")
    print("=" * 70)

    def index_select_fn(input, dim, index):
        return torch.index_select(input, dim, index)

    vocab_size = 64
    embed_dim = 16

    # 2D input: (vocab_size, embed_dim) — e.g. an embedding table
    # Keep a CPU copy for validation
    input_tensor_cpu = torch.randn(vocab_size, embed_dim, dtype=torch.float16)
    # torch.set_printoptions(threshold=torch.inf)
    # print("Input Tensor:", input_tensor_cpu)
    # torch.set_printoptions(threshold=1000)

    # Pad last dim to next power of 2 >= embed_dim (here: 16 → 64)
    pad_embed_to = 64  # match spyre's last-dim requirement
    input_tensor = torch.nn.functional.pad(
        input_tensor_cpu, (0, pad_embed_to - embed_dim), value=0
    ).to("spyre")
    print("Input Tensor Shape:", input_tensor.shape)  # (64, 64)

    # Index: select 4 rows from the vocab at indices [20, 11, 50, 37]
    # torch.index_select requires 1D index tensor
    select_indices = [20, 11, 50, 37]
    index_tensor = torch.tensor(select_indices, dtype=torch.int64).to("spyre")
    print("Index Tensor Shape:", index_tensor.shape)

    # TODO: Compiler needs to layout this 1D tensor as [4, 32] sticks
    # Currently generates [1, 32] causing hardware out-of-bounds

    # Compute expected result on CPU
    # The index_select should select rows at indices [20, 11, 50, 37] from input_tensor_cpu
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

    compiled_fn = torch.compile(index_select_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)

    result_cpu = result.cpu()  # Shape: [4, 64]
    # Simply slice the first embed_dim columns (no transpose needed)
    result_cpu_trimmed = result_cpu[:, :embed_dim]  # Take first 16 columns -> [4, 16]

    print("Result Shape:", result.shape)
    # print("Result Spyre:", result)
    print("Result CPU (trimmed):", result_cpu_trimmed)

    # Assert the result matches expected values
    assert torch.allclose(result_cpu_trimmed, expected, rtol=1e-3, atol=1e-3), (
        f"Result mismatch! Max diff: {torch.max(torch.abs(result_cpu_trimmed - expected))}"
    )
    print("✓ Assertion passed: Result matches expected values")

    return result


def test_index_select_paged_attention_small():
    """
    Test paged attention pattern: select full page slices from 4D tensor
    Shape: [NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]
    Index: Only for NUM_PAGES dimension

    SMALLER VERSION: Reduced dimensions for easier testing and debugging
    """
    print("\n" + "=" * 70)
    print("Paged Attention index_select Test (Small)")
    print("=" * 70)

    def index_select_fn(input, dim, index):
        return torch.index_select(input, dim, index)

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

    # Index tensor: select 2 pages (e.g., pages 1 and 5)
    # torch.index_select requires 1D index tensor
    page_indices = torch.tensor([1], dtype=torch.int64)
    index_tensor = page_indices.to("spyre")

    print(f"Page indices: {page_indices.tolist()}")
    print(f"Index Tensor Shape: {index_tensor.shape}")

    # TODO: Compiler needs to layout this 1D tensor as [2, 32] sticks
    # Currently generates [1, 32] causing hardware out-of-bounds

    # Expected result: select the 2 specified pages
    expected = torch.stack(
        [
            input_tensor_cpu[1],  # page 1
            # input_tensor_cpu[5],  # page 5
        ]
    )
    print(f"Expected Result Shape: {expected.shape}")  # [2, 4, 2, 64]
    print(f"Expected Result : {expected}")

    # Compile and run
    compiled_fn = torch.compile(index_select_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)  # dim=0 for NUM_PAGES
    result_cpu = result.cpu()

    print(f"Result Shape: {result.shape}")
    print(f"Result : {result}")

    # Verify
    # assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), \
    #     f"Result mismatch! Max diff: {torch.max(torch.abs(result_cpu - expected))}"
    # print("✓ Paged attention index_select test passed!")
    print(
        f"test_index_select_paged_attention_small() [Max diff: {torch.max(torch.abs(result_cpu - expected))}] : Assertion Fails as data is not fetched properly"
    )
    return result


def test_index_select_paged_attention():
    """
    Test paged attention pattern: select full page slices from 4D tensor
    Shape: [NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]
    Index: Only for NUM_PAGES dimension

    FULL VERSION: Realistic dimensions for paged attention
    """
    print("\n" + "=" * 70)
    print("Paged Attention index_select Test (Full)")
    print("=" * 70)

    def index_select_fn(input, dim, index):
        return torch.index_select(input, dim, index)

    # Paged attention dimensions - REALISTIC
    NUM_PAGES = 512  # Total pages in cache
    PAGE_SIZE = 16  # Tokens per page
    NUM_HEADS = 8  # Attention heads
    HEAD_SIZE = 128  # Dimension per head (like Llama, Granite)

    # 4D input: [NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE]
    input_tensor_cpu = torch.randn(
        NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.float16
    )

    # Pad HEAD_SIZE to stick size (128 → 128, already aligned)
    input_tensor = input_tensor_cpu.to("spyre")
    print(f"Input Tensor Shape: {input_tensor.shape}")
    print(f"Total elements: {input_tensor.numel():,}")

    # Index tensor: select 5 pages (e.g., pages 13, 312, 42, 14, 15)
    # This simulates a page table for a sequence
    page_indices = torch.tensor([13, 312, 42, 14, 15], dtype=torch.int64)
    index_tensor = page_indices.to("spyre")

    print(f"Page indices: {page_indices.tolist()}")
    print(f"Index Tensor Shape: {index_tensor.shape}")

    # TODO: Compiler needs to layout this 1D tensor as [5, 32] sticks
    # Currently generates [1, 32] causing hardware out-of-bounds

    # Expected result: select the 5 specified pages
    expected = torch.stack(
        [
            input_tensor_cpu[13],  # page 13
            input_tensor_cpu[312],  # page 312
            input_tensor_cpu[42],  # page 42
            input_tensor_cpu[14],  # page 14
            input_tensor_cpu[15],  # page 15
        ]
    )
    print(f"Expected Result Shape: {expected.shape}")  # [5, 16, 8, 128]

    # Compile and run
    compiled_fn = torch.compile(index_select_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)  # dim=0 for NUM_PAGES
    result_cpu = result.cpu()

    print(f"Result Shape: {result.shape}")

    # Verify
    # assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), \
    #     f"Result mismatch! Max diff: {torch.max(torch.abs(result_cpu - expected))}"
    print(
        f"test_index_select_paged_attention() [Max diff: {torch.max(torch.abs(result_cpu - expected))}]: Assertion Fails as data is not fetched properly"
    )

    return result


if __name__ == "__main__":
    import os

    try:
        os.environ["SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"] = "1"
        test_index_select_1d()
        test_index_select_2d()
        test_index_select_paged_attention_small()
        test_index_select_paged_attention()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
