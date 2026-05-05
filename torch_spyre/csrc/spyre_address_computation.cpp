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

namespace spyre {

at::Tensor compute_addresses_from_indices(
    const at::Tensor& indices,
    int64_t virtual_offset,
    const std::vector<int64_t>& device_size,
    const std::vector<int64_t>& device_stride,
    int64_t element_size) {

  // Store the original device for later
  auto original_device = indices.device();

  // Ensure indices tensor is on CPU
  auto indices_cpu = indices.cpu();
  
  // Get the shape of indices
  auto indices_shape = indices_cpu.sizes();
  int64_t ndim = indices_shape[indices_shape.size() - 1];  // Number of dimensions
  
  // Validate that ndim matches device_size
  TORCH_CHECK(
      ndim == static_cast<int64_t>(device_size.size()),
      "Index dimension mismatch: indices have ", ndim, " dimensions, "
      "but device_size has ", device_size.size(), " dimensions");
  
  TORCH_CHECK(
      ndim == static_cast<int64_t>(device_stride.size()),
      "Stride dimension mismatch: indices have ", ndim, " dimensions, "
      "but device_stride has ", device_stride.size(), " dimensions");
  
  // Flatten indices to process them in batch
  // Shape: [num_indices, ndim]
  auto flat_indices = indices_cpu.reshape({-1, ndim});
  int64_t num_indices = flat_indices.size(0);
  
  // Create output address tensor as float32 (to match SDSC SENUINT32 format)
  auto addresses = at::zeros({num_indices}, at::TensorOptions().dtype(at::kFloat));
  
  // Get accessor for efficient access
  auto flat_indices_accessor = flat_indices.accessor<int64_t, 2>();
  auto addresses_accessor = addresses.accessor<float, 1>();
  
  // Compute address for each index tuple
  for (int64_t i = 0; i < num_indices; ++i) {
    // Start with virtual offset (in bytes)
    int64_t address = virtual_offset;
    
    // Compute element offset using strides (in elements)
    int64_t element_offset = 0;
    for (int64_t dim = 0; dim < ndim; ++dim) {
      int64_t coord = flat_indices_accessor[i][dim];
      
      // Validate coordinate is within bounds
      TORCH_CHECK(
          coord >= 0 && coord < device_size[dim],
          "Index ", coord, " out of bounds for dimension ", dim,
          " (size: ", device_size[dim], ")");
      
      element_offset += coord * device_stride[dim];
    }
    
    // Convert element offset to byte offset and add to base address
    address += element_offset * element_size;
    
    // Store as float32 (bits will represent the address)
    addresses_accessor[i] = static_cast<float>(address);
  }
  
  // Reshape back to original shape (without the last dimension)
  std::vector<int64_t> output_shape(
      indices_shape.begin(), indices_shape.end() - 1);
  auto reshaped_addresses = addresses.reshape(output_shape);

  // Move the result back to the original device (spyre)
  return reshaped_addresses.to(original_device);
}

} // namespace spyre
