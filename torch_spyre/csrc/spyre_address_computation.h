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
 * Convert N-dimensional logical indices to physical memory addresses.
 *
 * Supports 2D, 3D, 4D, and higher-dimensional tensors with flexible indexing:
 *   - 2D: value[IN, OUT], indices[MB, IJ] → addresses row-major
 *   - 3D: value[D0, D1, D2], indices[MB, IJ, 3] → addresses for [d0, d1, d2]
 *   - 4D: value[D0, D1, D2, D3], indices[MB, IJ, 4] → addresses for [d0, d1,
 * d2, d3]
 *
 * Address formula:
 *   address = base_address + sum(index[i] * stride[i] * element_size)
 *
 * @param logical_indices Tensor of indices, shape [..., ndim] or [...] for 2D
 * @param value_tensor The tensor being accessed, shape [D0, D1, ..., Dn]
 * @param dim Dimension to index along (default: 0 for row-major)
 * @return Tensor of stick addresses, as float32
 */
at::Tensor indices_to_addresses_nd(const at::Tensor& logical_indices,
                                   const at::Tensor& value_tensor,
                                   int64_t dim = 0, int64_t virtual_offset = 0);

}  // namespace spyre
