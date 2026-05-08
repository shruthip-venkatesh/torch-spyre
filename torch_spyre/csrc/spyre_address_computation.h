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

#pragma once

#include <ATen/ATen.h>
#include <vector>

namespace spyre {

/**
 * Compute HBM byte addresses from logical indices.
 *
 * This function converts logical tensor indices to physical HBM byte addresses
 * by computing the address based on the device layout (size and stride),
 * element size, and adding the virtual offset.
 *
 * @param indices Tensor of logical indices, shape [..., ndim]
 * @param virtual_offset Base virtual address offset of the tensor in HBM (bytes)
 * @param device_size Size of each dimension in the device layout
 * @param device_stride Stride of each dimension in the device layout (elements)
 * @param element_size Size of each element in bytes (dtype.itemsize)
 * @return Tensor of HBM byte addresses with shape [...], as float32
 */
at::Tensor compute_addresses_from_indices(
    const at::Tensor& indices,
    int64_t virtual_offset,
    const std::vector<int64_t>& device_size,
    const std::vector<int64_t>& device_stride,
    int64_t element_size);

/**
 * Compute stick addresses from row indices for 2D gather operations.
 *
 * This function converts row indices to stick addresses for gathering rows
 * from a 2D tensor. Each row index is converted to the stick address of the
 * first stick in that row.
 *
 * For a [rows, cols] tensor with stick groups:
 * - Each stick group contains rows_per_group rows
 * - Row index is relative to the stick group
 * - Absolute row = stick_group * rows_per_group + row_in_group
 * - Stick address = absolute_row * sticks_per_row
 *
 * @param row_indices Tensor of row indices, shape [num_groups, rows_per_group]
 * @param virtual_offset Base virtual address offset of the tensor in HBM (bytes)
 * @param rows_per_group Number of rows in each stick group
 * @param cols_per_row Number of columns per row
 * @param element_size Size of each element in bytes (dtype.itemsize)
 * @return Tensor of stick addresses with same shape as row_indices, as float32
 */
at::Tensor compute_stick_addresses_from_rows(
    const at::Tensor& row_indices,
    int64_t virtual_offset,
    int64_t rows_per_group,
    int64_t cols_per_row,
    int64_t element_size);

} // namespace torch_spyre
