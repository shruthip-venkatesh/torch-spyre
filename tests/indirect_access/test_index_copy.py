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

# TODO: Address computation for multicore: WIP
# Until then set `SENCORES=1` when testing indirect access

import torch


def test_index_copy_1d():
    """Test index_copy operation for 1D tensors.

    index_copy semantics: output[index[i]] = src[i]

    This uses the standard torch.index_copy operation which should work
    with the Spyre backend.
    """
    print("\n" + "=" * 70)
    print("1D Index Copy Test (torch.index_copy)")
    print("=" * 70)

    def index_copy_fn(input, dim, index, src):
        return torch.index_copy(input, dim, index, src)

    # Source values: 5 elements to copy
    src_cpu = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float16)
    src = torch.nn.functional.pad(src_cpu.reshape(5, 1), (0, 63), value=0).to("spyre")
    print("src shape:", src.shape)

    # Copy indices: where each src element should land in output
    copy_indices = torch.tensor([2, 0, 4, 1, 3], dtype=torch.int64).to("spyre")
    print("index shape:", copy_indices.shape)

    # Output buffer: 5 slots (zero-initialized)
    output_size = 5
    input_tensor = torch.zeros(output_size, 1, dtype=torch.float16)
    input_tensor = torch.nn.functional.pad(input_tensor, (0, 63), value=0).to("spyre")
    print("input_tensor shape:", input_tensor.shape)

    # Expected result: reverse-map each copy index
    expected = torch.zeros(output_size, dtype=torch.float16)
    for val, idx in zip(src_cpu.tolist(), copy_indices.cpu().tolist()):
        expected[idx] = val
    print("Expected:", expected.tolist())

    compiled_fn = torch.compile(index_copy_fn)
    result = compiled_fn(input_tensor, 0, copy_indices, src)
    result_cpu = result.cpu()[:, 0]
    print("Result:", result_cpu.tolist())

    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Mismatch!\nExpected: {expected}\nGot: {result_cpu}"
    )
    print("✓ 1D index_copy passed")
    return result


def test_index_copy_2d():
    """Test index_copy for 2D src/output tensors (row-wise copy).

    index_copy semantics: output[index[i], :] = src[i, :]
    
    This uses the standard torch.index_copy operation.
    """
    print("\n" + "=" * 70)
    print("2D Index Copy Test (torch.index_copy)")
    print("=" * 70)

    def index_copy_fn(input, dim, index, src):
        return torch.index_copy(input, dim, index, src)

    vocab_size = 64
    embed_dim = 16
    num_writes = 4

    # Source: rows to copy into the output table
    src_cpu = torch.randn(num_writes, embed_dim, dtype=torch.float16)
    pad_to = 64
    src = torch.nn.functional.pad(src_cpu, (0, pad_to - embed_dim), value=0).to("spyre")
    print("src shape:", src.shape)  # (4, 64)

    # Copy indices: which output rows to overwrite
    copy_rows = torch.tensor([20, 11, 50, 37], dtype=torch.int64).to("spyre")
    print("index shape:", copy_rows.shape)  # (4,)

    # Output buffer: zero-initialized embedding table
    input_tensor = torch.zeros(vocab_size, pad_to, dtype=torch.float16, device="spyre")
    print("input_tensor shape:", input_tensor.shape)  # (64, 64)

    # Expected: only the copy_rows get filled
    expected = torch.zeros(vocab_size, embed_dim, dtype=torch.float16)
    for write_idx, row_idx in enumerate(copy_rows.cpu().tolist()):
        expected[row_idx] = src_cpu[write_idx]
    print("Expected non-zero rows:", copy_rows.cpu().tolist())

    compiled_fn = torch.compile(index_copy_fn)
    result = compiled_fn(input_tensor, 0, copy_rows, src)
    result_cpu = result.cpu()[:, :embed_dim]
    print("Result shape:", result_cpu.shape)

    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Mismatch!\n"
        f"Max diff: {torch.max(torch.abs(result_cpu - expected))}"
    )
    print("✓ 2D index_copy passed")
    return result


def test_index_copy_embedding_update():
    """Test index_copy for updating specific rows in an embedding table.
    
    This simulates a common use case: updating embeddings during training.
    """
    print("\n" + "=" * 70)
    print("Embedding Update Index Copy Test (torch.index_copy)")
    print("=" * 70)

    def index_copy_fn(input, dim, index, src):
        return torch.index_copy(input, dim, index, src)

    vocab_size = 128
    embed_dim = 32
    num_updates = 8

    # Source: new embedding vectors to copy
    src_cpu = torch.randn(num_updates, embed_dim, dtype=torch.float16)
    pad_to = 64
    src = torch.nn.functional.pad(src_cpu, (0, pad_to - embed_dim), value=0).to("spyre")
    print("src shape:", src.shape)

    # Copy indices: which embedding rows to update
    update_rows = torch.tensor([10, 25, 40, 55, 70, 85, 100, 115], dtype=torch.int64).to("spyre")
    print("index shape:", update_rows.shape)

    # Output buffer: existing embedding table (zero-initialized for test)
    input_tensor = torch.zeros(vocab_size, pad_to, dtype=torch.float16, device="spyre")
    print("input_tensor shape:", input_tensor.shape)

    # Expected: only the update_rows get new values
    expected = torch.zeros(vocab_size, embed_dim, dtype=torch.float16)
    for update_idx, row_idx in enumerate(update_rows.cpu().tolist()):
        expected[row_idx] = src_cpu[update_idx]
    print("Expected non-zero rows:", update_rows.cpu().tolist())

    compiled_fn = torch.compile(index_copy_fn)
    result = compiled_fn(input_tensor, 0, update_rows, src)
    result_cpu = result.cpu()[:, :embed_dim]
    print("Result shape:", result_cpu.shape)

    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Mismatch!\n"
        f"Max diff: {torch.max(torch.abs(result_cpu - expected))}"
    )
    print("✓ Embedding update index_copy passed")
    return result


def test_index_copy_small_batch():
    """Test index_copy with a small batch size for easier debugging."""
    print("\n" + "=" * 70)
    print("Small Batch Index Copy Test (torch.index_copy)")
    print("=" * 70)

    def index_copy_fn(input, dim, index, src):
        return torch.index_copy(input, dim, index, src)

    # Small dimensions for debugging
    output_size = 16
    num_writes = 3
    feature_dim = 8

    # Source values
    src_cpu = torch.randn(num_writes, feature_dim, dtype=torch.float16)
    pad_to = 64
    src = torch.nn.functional.pad(src_cpu, (0, pad_to - feature_dim), value=0).to("spyre")
    print("src shape:", src.shape)

    # Copy indices
    copy_rows = torch.tensor([2, 7, 12], dtype=torch.int64).to("spyre")
    print("index shape:", copy_rows.shape)

    # Output buffer
    input_tensor = torch.zeros(output_size, pad_to, dtype=torch.float16, device="spyre")
    print("input_tensor shape:", input_tensor.shape)

    # Expected result
    expected = torch.zeros(output_size, feature_dim, dtype=torch.float16)
    for write_idx, row_idx in enumerate(copy_rows.cpu().tolist()):
        expected[row_idx] = src_cpu[write_idx]
    print("Expected non-zero rows:", copy_rows.cpu().tolist())
    print("Expected values at row 2:", expected[2].tolist())

    compiled_fn = torch.compile(index_copy_fn)
    result = compiled_fn(input_tensor, 0, copy_rows, src)
    result_cpu = result.cpu()[:, :feature_dim]
    print("Result shape:", result_cpu.shape)
    print("Result values at row 2:", result_cpu[2].tolist())

    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Mismatch!\n"
        f"Max diff: {torch.max(torch.abs(result_cpu - expected))}"
    )
    print("✓ Small batch index_copy passed")
    return result


def test_index_copy_paged_attention():
    """Test index_copy for paged attention pattern.
    
    This simulates copying specific pages from a source to a KV cache.
    """
    print("\n" + "=" * 70)
    print("Paged Attention Index Copy Test (torch.index_copy)")
    print("=" * 70)

    def index_copy_fn(input, dim, index, src):
        return torch.index_copy(input, dim, index, src)

    # Paged attention dimensions - SMALL for testing
    NUM_PAGES = 8       # Total pages in cache
    PAGE_SIZE = 4       # Tokens per page
    NUM_HEADS = 2       # Attention heads
    HEAD_SIZE = 64      # Dimension per head

    # Source: 2 pages to copy
    num_src_pages = 2
    src_cpu = torch.randn(num_src_pages, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
    src = src_cpu.to("spyre")
    print(f"src shape: {src.shape}")

    # Copy indices: which pages to update
    page_indices = torch.tensor([1, 5], dtype=torch.int64).to("spyre")
    print(f"page_indices: {page_indices.cpu().tolist()}")

    # Output buffer: KV cache
    input_tensor = torch.zeros(NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.float16, device="spyre")
    print(f"input_tensor shape: {input_tensor.shape}")

    # Expected result
    expected = torch.zeros(NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
    for src_idx, page_idx in enumerate(page_indices.cpu().tolist()):
        expected[page_idx] = src_cpu[src_idx]
    print(f"Expected non-zero pages: {page_indices.cpu().tolist()}")

    compiled_fn = torch.compile(index_copy_fn)
    result = compiled_fn(input_tensor, 0, page_indices, src)
    result_cpu = result.cpu()

    print(f"Result shape: {result.shape}")
    
    assert torch.allclose(result_cpu, expected, rtol=1e-3, atol=1e-3), (
        f"Mismatch! Max diff: {torch.max(torch.abs(result_cpu - expected))}"
    )
    print("✓ Paged attention index_copy passed")
    return result


if __name__ == "__main__":
    try:
        test_index_copy_1d()
        # test_index_copy_2d()
        # test_index_copy_embedding_update()
        # test_index_copy_small_batch()
        # test_index_copy_paged_attention()
        print("\n" + "=" * 70)
        print("All index_copy tests passed! ✓")
        print("=" * 70)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
