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
  
  // Create output address tensor as float32 for device copy support
  // The SENUINT32 interpretation happens in SDSC JSON metadata
  auto addresses = at::zeros({num_indices}, at::TensorOptions().dtype(at::kFloat));
  
  // Get accessor for efficient access
  auto flat_indices_accessor = flat_indices.accessor<int64_t, 2>();
  auto addresses_accessor = addresses.accessor<float, 1>();
  
  // DeepTools stick size (128 bytes for Sen1.0)
  constexpr int64_t STICK_SIZE_BYTES = 128;
  // Elements per stick for FP16 (128 bytes / 2 bytes per element = 64 elements)
  constexpr int64_t ELEMENTS_PER_STICK_FP16 = 64;

  // Compute address for each index tuple
  for (int64_t i = 0; i < num_indices; ++i) {
    // HBM Layout for 2D tensors:
    // A [32, 128] FP16 tensor has:
    // - 32 rows × 128 columns = 4096 elements
    // - Each stick holds 64 FP16 elements
    // - Total sticks = 4096 / 64 = 64 sticks
    // - Device layout: [64, 64] (64 sticks, 64 elements per stick)
    //
    // Mapping from logical [32, 128] to device [64, 64]:
    // - Each logical row (128 elements) spans 2 sticks (128 / 64 = 2)
    // - Logical element (row, col) maps to:
    //   - stick_index = row * 2 + (col / 64)
    //   - within_stick_offset = col % 64
    
    // Start with virtual offset (in bytes)
    int64_t byte_address = virtual_offset;

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
    byte_address += element_offset * element_size;
    
    // DeepTools expects stick addresses (not byte addresses)
    // Convert from bytes to sticks (128-byte units)
    TORCH_CHECK(
        byte_address % STICK_SIZE_BYTES == 0,
        "Address ", byte_address, " is not stick-aligned (must be multiple of ",
        STICK_SIZE_BYTES, " bytes). Element offset: ", element_offset,
        ", byte offset: ", element_offset * element_size);

    int64_t stick_address = byte_address / STICK_SIZE_BYTES;
    
    // Store as float (will be interpreted as SENUINT32 via SDSC metadata)
    addresses_accessor[i] = static_cast<float>(stick_address);
  }
  
  // Reshape back to original shape (without the last dimension)
  std::vector<int64_t> output_shape(
      indices_shape.begin(), indices_shape.end() - 1);
  auto reshaped_addresses = addresses.reshape(output_shape);

  // Move the result back to the original device (spyre)
  return reshaped_addresses.to(original_device);
}

at::Tensor compute_stick_addresses_from_rows(
    const at::Tensor& row_indices,
    int64_t virtual_offset,
    int64_t rows_per_group,
    int64_t cols_per_row,
    int64_t element_size) {

  // Store the original device for later
  auto original_device = row_indices.device();

  // Ensure row_indices tensor is on CPU
  auto row_indices_cpu = row_indices.cpu();
  
  // Get the shape of row_indices: [num_groups, rows_per_group]
  auto indices_shape = row_indices_cpu.sizes();
  TORCH_CHECK(
      indices_shape.size() == 2,
      "Row indices must be 2D tensor, got ", indices_shape.size(), "D");
  
  int64_t num_groups = indices_shape[0];
  int64_t rows_to_gather = indices_shape[1];
  
  // Create output address tensor as float32
  auto addresses = at::zeros({num_groups, rows_to_gather},
                             at::TensorOptions().dtype(at::kFloat));
  
  // Get accessors for efficient access
  auto row_indices_accessor = row_indices_cpu.accessor<int64_t, 2>();
  auto addresses_accessor = addresses.accessor<float, 2>();
  
  // DeepTools stick size (128 bytes for Sen1.0)
  constexpr int64_t STICK_SIZE_BYTES = 128;
  
  // Compute stick address for each row index
  for (int64_t group = 0; group < num_groups; ++group) {
    // Base row for this stick group
    int64_t base_row = group * rows_per_group;
    
    for (int64_t idx = 0; idx < rows_to_gather; ++idx) {
      int64_t row_in_group = row_indices_accessor[group][idx];
      
      // Validate row index is within bounds
      TORCH_CHECK(
          row_in_group >= 0 && row_in_group < rows_per_group,
          "Row index ", row_in_group, " out of bounds for group ", group,
          " (rows_per_group: ", rows_per_group, ")");
      
      // Compute absolute row index
      int64_t absolute_row = base_row + row_in_group;
      
      // Compute byte address for the start of this row (column 0)
      // element_offset = absolute_row * cols_per_row
      // byte_offset = element_offset * element_size
      int64_t element_offset = absolute_row * cols_per_row;
      int64_t byte_offset = element_offset * element_size;
      int64_t absolute_byte_address = virtual_offset + byte_offset;
      
      // Convert to stick address
      TORCH_CHECK(
          absolute_byte_address % STICK_SIZE_BYTES == 0,
          "Address ", absolute_byte_address, " for row ", absolute_row,
          " is not stick-aligned (must be multiple of ", STICK_SIZE_BYTES, " bytes)");
      
      int64_t stick_address = absolute_byte_address / STICK_SIZE_BYTES;
      
      // Store as float (will be interpreted as SENUINT32 via SDSC metadata)
      addresses_accessor[group][idx] = static_cast<float>(stick_address);
    }
  }
  
  // Move the result back to the original device (spyre)
  return addresses.to(original_device);
}

at::Tensor indices_to_addresses_2d(
    const at::Tensor& logical_indices,
    const at::Tensor& value_tensor) {
  
  // Store the original device for later
  auto original_device = logical_indices.device();
  
  // Only move indices to CPU for computation (value tensor stays on device)
  auto indices_cpu = logical_indices.cpu();
  
  // Get shapes (no need to move value_tensor to CPU for metadata)
  auto indices_shape = indices_cpu.sizes();
  auto value_shape = value_tensor.sizes();
  
  // Validate: value_tensor must be 2D [IN, OUT]
  TORCH_CHECK(
      value_shape.size() == 2,
      "value_tensor must be 2D [IN, OUT], got ", value_shape.size(), "D");
  
  int64_t in_dim = value_shape[0];
  int64_t out_dim = value_shape[1];
  
  // Get element size and compute row stride
  int64_t element_size = value_tensor.element_size();
  int64_t row_stride_bytes = out_dim * element_size;
  
  // Use relative addressing starting from 0
  // The actual HBM base address will be set in SDSC JSON
  int64_t base_address = 0;
  
  // Flatten indices for processing
  auto flat_indices = indices_cpu.reshape({-1});
  int64_t num_indices = flat_indices.size(0);
  
  // Create output address tensor as float32
  auto addresses = at::zeros({num_indices}, at::TensorOptions().dtype(at::kFloat));
  
  // Get accessors
  auto flat_indices_accessor = flat_indices.accessor<int64_t, 1>();
  auto addresses_accessor = addresses.accessor<float, 1>();
  
  // DeepTools stick size (128 bytes for Sen1.0)
  constexpr int64_t STICK_SIZE_BYTES = 128;
  
  // Compute address for each index
  for (int64_t i = 0; i < num_indices; ++i) {
    int64_t row_index = flat_indices_accessor[i];
    
    // Validate row index is within bounds
    TORCH_CHECK(
        row_index >= 0 && row_index < in_dim,
        "Row index ", row_index, " out of bounds for dimension IN=", in_dim);
    
    // DeepTools address formula:
    // address = base_address + (row_index * row_stride)
    int64_t byte_address = base_address + (row_index * row_stride_bytes);
    
    // Verify stick alignment
    TORCH_CHECK(
        byte_address % STICK_SIZE_BYTES == 0,
        "Address ", byte_address, " for row ", row_index,
        " is not stick-aligned (must be multiple of ", STICK_SIZE_BYTES, " bytes). ",
        "Row stride: ", row_stride_bytes, " bytes");
    
    // Convert to stick address
    int64_t stick_address = byte_address / STICK_SIZE_BYTES;
    
    // Store as float (will be interpreted as SENUINT32 via SDSC metadata)
    addresses_accessor[i] = static_cast<float>(stick_address);
  }
  
  // Reshape back to original shape
  auto reshaped_addresses = addresses.reshape(indices_shape);
  
  // Move the result back to the original device (spyre)
  return reshaped_addresses.to(original_device);
}

at::Tensor indices_to_addresses_nd(
    const at::Tensor& logical_indices,
    const at::Tensor& value_tensor,
    int64_t dim) {
  
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
  
  // Use relative addressing starting from 0
  // The actual HBM base address will be set in SDSC JSON
  int64_t base_address = 0;
  int64_t element_size = value_tensor.element_size();
  
  // Compute strides for value tensor (row-major order)
  std::vector<int64_t> value_strides(value_ndim);
  int64_t stride = 1;
  for (int64_t i = value_ndim - 1; i >= 0; --i) {
    value_strides[i] = stride;
    stride *= value_shape[i];
  }
  
  // Determine indexing mode based on indices shape
  bool is_multidim_index = (indices_shape.size() > 0 &&
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
      
      // Compute offset using all dimensions
      for (int64_t d = 0; d < value_ndim; ++d) {
        int64_t coord = flat_indices_accessor[i][d];
        
        // Validate coordinate
        TORCH_CHECK(
            coord >= 0 && coord < value_shape[d],
            "Index ", coord, " out of bounds for dimension ", d,
            " (size: ", value_shape[d], ")");
        
        element_offset += coord * value_strides[d];
      }
      
      // Convert to byte address
      int64_t byte_address = base_address + (element_offset * element_size);
      
      // Verify stick alignment
      TORCH_CHECK(
          byte_address % STICK_SIZE_BYTES == 0,
          "Address ", byte_address, " is not stick-aligned. ",
          "Element offset: ", element_offset);
      
      // Convert to stick address
      int64_t stick_address = byte_address / STICK_SIZE_BYTES;
      addresses_accessor[i] = static_cast<float>(stick_address);
    }
    
    // Reshape back to original shape (without last dimension)
    std::vector<int64_t> output_shape(
        indices_shape.begin(), indices_shape.end() - 1);
    addresses = addresses.reshape(output_shape);
    
  } else {
    // Single-dimension indexing: indices shape is [...]
    // Index along specified dimension (default: dim=0, row-major)
    
    TORCH_CHECK(
        dim >= 0 && dim < value_ndim,
        "Dimension ", dim, " out of bounds for ", value_ndim, "D tensor");
    
    auto flat_indices = indices_cpu.reshape({-1});
    int64_t num_indices = flat_indices.size(0);
    
    addresses = at::zeros({num_indices}, at::TensorOptions().dtype(at::kFloat));
    auto flat_indices_accessor = flat_indices.accessor<int64_t, 1>();
    auto addresses_accessor = addresses.accessor<float, 1>();
    
    // Compute stride for the indexed dimension
    int64_t dim_stride_elements = value_strides[dim];
    int64_t dim_stride_bytes = dim_stride_elements * element_size;
    
    for (int64_t i = 0; i < num_indices; ++i) {
      int64_t index = flat_indices_accessor[i];
      
      // Validate index
      TORCH_CHECK(
          index >= 0 && index < value_shape[dim],
          "Index ", index, " out of bounds for dimension ", dim,
          " (size: ", value_shape[dim], ")");
      
      // Compute address: base + (index * stride)
      int64_t byte_address = base_address + (index * dim_stride_bytes);
      
      // Verify stick alignment
      TORCH_CHECK(
          byte_address % STICK_SIZE_BYTES == 0,
          "Address ", byte_address, " for index ", index,
          " is not stick-aligned. Stride: ", dim_stride_bytes, " bytes");
      
      // Convert to stick address
      int64_t stick_address = byte_address / STICK_SIZE_BYTES;
      addresses_accessor[i] = static_cast<float>(stick_address);
    }
    
    // Reshape back to original indices shape
    addresses = addresses.reshape(indices_shape);
  }
  
  // Move the result back to the original device (spyre)
  return addresses.to(original_device);
}

} // namespace spyre
