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


def is_indirect_access_operation(op) -> bool:
    """Check if an operation uses indirect access (e.g., gather, scatter, index_select).

    Indirect access operations have op_info with either 'index_args' or 'index_value_pairs'.

    Args:
        op: Operation to check (typically a ComputedBuffer or SchedulerNode)

    Returns:
        True if the operation uses indirect access, False otherwise
    """
    if not hasattr(op, "data"):
        return False
    if not hasattr(op.data, "op_info"):
        return False
    if not op.data.op_info:
        return False

    return "index_args" in op.data.op_info or "index_value_pairs" in op.data.op_info
