// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "spyre_address_computation.h"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <vector>

#include "spyre_tensor_impl.h"

namespace spyre {

at::Tensor indices_to_addresses_nd(const at::Tensor& logical_indices,
                                   const at::Tensor& value_tensor, int64_t dim,
                                   int64_t virtual_offset) {
  // Store the original device for later
  auto original_device = logical_indices.device();

  // Only move indices to CPU for computation (value tensor stays on device)
  auto indices_cpu = logical_indices.cpu();

  // Get shapes (no need to move value_tensor to CPU for metadata)
  auto indices_shape = indices_cpu.sizes();
  auto value_shape = value_tensor.sizes();
  int64_t value_ndim = value_shape.size();

  // DeepTools stick size (128 bytes for Sen1.0)
  constexpr int64_t STICK_SIZE_BYTES = 128;

  // Use virtual_offset as the base address for address computation
  // This allows for paged attention and other use cases requiring non-zero base
  // addresses
  int64_t base_address = virtual_offset;
  int64_t element_size = value_tensor.element_size();

  // Use the tensor's actual strides from SpyreTensorLayout.
  // The stride_map contains host memory offsets for each device dimension.
  // Since device dimensions include the stick dimension, we need to use
  // the tensor's logical strides which correspond to host memory layout.
  auto value_strides = value_tensor.strides();

  // Determine indexing mode based on indices shape
  bool is_multidim_index =
      (indices_shape.size() > 0 &&
       indices_shape[indices_shape.size() - 1] == value_ndim);

  at::Tensor addresses;

  if (is_multidim_index) {
    // Multi-dimensional indexing: indices shape is [..., ndim]
    // Each index tuple specifies coordinates in all dimensions

    auto flat_indices = indices_cpu.reshape({-1, value_ndim});
    int64_t num_indices = flat_indices.size(0);

    addresses = at::zeros({num_indices}, at::TensorOptions().dtype(at::kFloat));
    auto flat_indices_accessor = flat_indices.accessor<int64_t, 2>();
    auto addresses_accessor = addresses.accessor<float, 1>();

    for (int64_t i = 0; i < num_indices; ++i) {
      int64_t element_offset = 0;

      // Compute offset using all dimensions with actual tensor strides
      for (int64_t d = 0; d < value_ndim; ++d) {
        int64_t coord = flat_indices_accessor[i][d];

        // Validate coordinate
        TORCH_CHECK(coord >= 0 && coord < value_shape[d], "Index ", coord,
                    " out of bounds for dimension ", d,
                    " (size: ", value_shape[d], ")");

        element_offset += coord * value_strides[d];
      }

      // Convert to byte address
      int64_t byte_address = base_address + (element_offset * element_size);

      // Verify stick alignment
      TORCH_CHECK(byte_address % STICK_SIZE_BYTES == 0, "Address ",
                  byte_address, " is not stick-aligned. ",
                  "Element offset: ", element_offset);

      // Convert to stick address
      int64_t stick_address = byte_address / STICK_SIZE_BYTES;
      addresses_accessor[i] = static_cast<float>(stick_address);
    }

    // Reshape back to original shape (without last dimension)
    std::vector<int64_t> output_shape(indices_shape.begin(),
                                      indices_shape.end() - 1);
    addresses = addresses.reshape(output_shape);

  } else {
    // Single-dimension indexing: indices shape is [...]
    // Index along specified dimension (default: dim=0, row-major)

    TORCH_CHECK(dim >= 0 && dim < value_ndim, "Dimension ", dim,
                " out of bounds for ", value_ndim, "D tensor");

    auto flat_indices = indices_cpu.reshape({-1});
    int64_t num_indices = flat_indices.size(0);

    addresses = at::zeros({num_indices}, at::TensorOptions().dtype(at::kLong));
    auto flat_indices_accessor = flat_indices.accessor<int64_t, 1>();
    auto addresses_accessor = addresses.accessor<int64_t, 1>();

    // Compute stride for the indexed dimension using actual tensor strides
    int64_t dim_stride_elements = value_strides[dim];
    int64_t dim_stride_bytes = dim_stride_elements * element_size;

    for (int64_t i = 0; i < num_indices; ++i) {
      int64_t index = flat_indices_accessor[i];

      // Validate index
      TORCH_CHECK(index >= 0 && index < value_shape[dim], "Index ", index,
                  " out of bounds for dimension ", dim,
                  " (size: ", value_shape[dim], ")");

      // Compute address: base + (index * stride)
      int64_t byte_address = base_address + (index * dim_stride_bytes);

      // Verify stick alignment
      TORCH_CHECK(byte_address % STICK_SIZE_BYTES == 0, "Address ",
                  byte_address, " for index ", index,
                  " is not stick-aligned. Stride: ", dim_stride_bytes,
                  " bytes");

      // Convert to stick address
      int64_t stick_address = byte_address / STICK_SIZE_BYTES;
      addresses_accessor[i] = static_cast<int64_t>(stick_address);
    }

    // Reshape back to original indices shape
    addresses = addresses.reshape(indices_shape);
  }

  // Move the result back to the original device (spyre)
  return addresses.to(original_device);
}

}  // namespace spyre
