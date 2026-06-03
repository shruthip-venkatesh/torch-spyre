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

"""
Paged Fetch Operation for True Paged Attention

This implements a custom operation where:
- Value tensor: [num_pages, page_size, heads, head_size]
- Index tensor: [num_requests] containing page numbers
- Output: [num_requests, page_size, heads, head_size]

Each request fetches a complete page from the value tensor.
"""

import torch
from torch import Tensor
from typing import Optional


def paged_fetch(
    value_tensor: Tensor,
    page_indices: Tensor,
) -> Tensor:
    """
    Fetch complete pages from a paged value tensor.
    
    Args:
        value_tensor: Tensor of shape [num_pages, page_size, heads, head_size]
                     containing the paged data
        page_indices: Tensor of shape [num_requests] containing page numbers
                     to fetch (int32 or int64)
    
    Returns:
        Tensor of shape [num_requests, page_size, heads, head_size] containing
        the fetched pages
    
    Example:
        >>> value = torch.randn(64, 128, 8, 64)  # 64 pages
        >>> indices = torch.tensor([10, 25, 40, 55])  # Fetch 4 pages
        >>> result = paged_fetch(value, indices)
        >>> result.shape
        torch.Size([4, 128, 8, 64])
    """
    # Validate inputs
    if value_tensor.dim() != 4:
        raise ValueError(
            f"value_tensor must be 4D [num_pages, page_size, heads, head_size], "
            f"got shape {value_tensor.shape}"
        )
    
    if page_indices.dim() != 1:
        raise ValueError(
            f"page_indices must be 1D [num_requests], got shape {page_indices.shape}"
        )
    
    num_pages = value_tensor.shape[0]
    
    # Check indices are in valid range
    if torch.any(page_indices < 0) or torch.any(page_indices >= num_pages):
        raise ValueError(
            f"page_indices must be in range [0, {num_pages}), "
            f"got min={page_indices.min()}, max={page_indices.max()}"
        )
    
    # Simple indexing: value_tensor[page_indices]
    # This fetches complete pages for each request
    return value_tensor[page_indices]


# Register as a custom op for torch.compile
@torch.library.custom_op("spyre::paged_fetch", mutates_args=())
def paged_fetch_op(value_tensor: Tensor, page_indices: Tensor) -> Tensor:
    """Custom op registration for paged_fetch"""
    return paged_fetch(value_tensor, page_indices)


@paged_fetch_op.register_fake
def paged_fetch_fake(value_tensor: Tensor, page_indices: Tensor) -> Tensor:
    """Fake implementation for shape inference"""
    if value_tensor.dim() != 4:
        raise ValueError("value_tensor must be 4D")
    if page_indices.dim() != 1:
        raise ValueError("page_indices must be 1D")
    
    num_requests = page_indices.shape[0]
    page_size, heads, head_size = value_tensor.shape[1:]
    
    return torch.empty(
        (num_requests, page_size, heads, head_size),
        dtype=value_tensor.dtype,
        device=value_tensor.device
    )


# Convenience function that works with torch.compile
def paged_attention_fetch(
    value_tensor: Tensor,
    page_indices: Tensor,
) -> Tensor:
    """
    High-level API for paged attention fetch.
    
    This function can be used directly in models and will be compiled
    efficiently by torch.compile.
    
    Args:
        value_tensor: [num_pages, page_size, heads, head_size]
        page_indices: [num_requests] page numbers to fetch
    
    Returns:
        [num_requests, page_size, heads, head_size] fetched pages
    
    Example:
        >>> # In a transformer model
        >>> kv_cache = torch.randn(1000, 128, 32, 64)  # 1000 pages
        >>> active_pages = torch.tensor([10, 25, 40])   # 3 active requests
        >>> 
        >>> @torch.compile
        >>> def attention_with_paging(cache, pages):
        >>>     kv_data = paged_attention_fetch(cache, pages)
        >>>     # ... rest of attention computation
        >>>     return output
        >>> 
        >>> result = attention_with_paging(kv_cache, active_pages)
    """
    return torch.ops.spyre.paged_fetch(value_tensor, page_indices)

# Made with Bob
