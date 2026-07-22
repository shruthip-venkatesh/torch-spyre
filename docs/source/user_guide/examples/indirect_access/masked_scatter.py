# Copyright 2026 The Torch-Spyre Authors.
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

"""Simple correctness example for the spyre_masked_scatter decomposition.

Tests a row-broadcast mask (stride(-1) == 0) on a [1, ROWS, COLS] tensor,
which is the primary supported shape.

Run:
    SENCORES=1 python masked_scatter.py
"""

import torch

DEV = "spyre"
DTYPE = torch.float16

ROWS, COLS, SRC_ROWS, N_TRUE = 855, 5120, 266, 266


def _masked_scatter(inp, mask, src):
    return torch.masked_scatter(inp, mask, src)


def main():
    torch.manual_seed(0)

    # Inputs on CPU
    inp_cpu = torch.rand(1, ROWS, COLS, dtype=DTYPE)
    src_cpu = torch.rand(SRC_ROWS, COLS, dtype=DTYPE)

    # Row-broadcast mask: shape [1, ROWS] expanded to [1, ROWS, COLS]
    mask_1d = torch.zeros(1, ROWS, dtype=torch.bool)
    mask_1d[0, torch.randperm(ROWS)[:N_TRUE]] = True
    mask_nd_cpu = mask_1d.unsqueeze(-1).expand(1, ROWS, COLS)

    # Move to device
    inp_dev = inp_cpu.to(DEV)
    src_dev = src_cpu.to(DEV)
    mask_nd_dev = mask_1d.unsqueeze(-1).to(DEV).expand(1, ROWS, COLS)

    # Reference: round-trip inputs through the device to neutralise SEN169_FP16
    # rounding (9 mantissa bits vs IEEE fp16's 10). masked_scatter only copies
    # values, so the round-tripped reference is an exact comparison baseline.
    inp_rt = inp_dev.cpu()
    src_rt = src_dev.cpu()
    expected = torch.masked_scatter(inp_rt, mask_nd_cpu, src_rt)

    # Compile and run on device
    got = torch.compile(_masked_scatter)(inp_dev, mask_nd_dev, src_dev).cpu()

    # Check
    n_diff = int((got != expected).sum())
    total = got.numel()
    if n_diff == 0:
        print(f"PASSED  ({total} elements, {N_TRUE} selected rows)")
    else:
        bad_rows = (got != expected).reshape(ROWS, COLS).any(dim=1).nonzero().flatten()
        print(
            f"FAILED  {n_diff}/{total} elements wrong, {bad_rows.numel()} rows affected"
        )
        assert False, f"masked_scatter mismatch: {n_diff} elements wrong"


if __name__ == "__main__":
    main()
