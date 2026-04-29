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

} // namespace torch_spyre
