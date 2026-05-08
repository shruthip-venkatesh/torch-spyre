import torch

def test_gather_2d_output():
    """Test gather operation that produces a 2D output tensor.
    
    This test demonstrates how to use indirect_gather to produce a 2D output
    by using a 2D index tensor. The key is that the index tensor shape
    determines the output shape.
    """
    def gather_fn(input, index):
        return torch.ops.spyre.indirect_gather(input, index)

    # 2D Data tensor (32 rows x 256 columns = 8192 elements)
    # Device layout: [32, 256] where each row is 4 sticks (256/64 = 4 sticks per row)
    input = torch.arange(8192, dtype=torch.float16, device="spyre").reshape(32, 256)
    
    print("Input tensor shape:", input.shape)
    print("Input tensor (first 3 rows, first 8 cols):")
    print(input[:3, :8])
    
    # Create a 2D index tensor to produce 2D output
    # Shape: [num_rows_to_gather, num_cols_to_gather]
    # Example: Gather 5 rows × 4 columns = 20 elements in 2D layout
    
    # Select which rows and columns to gather
    row_indices = torch.tensor([0, 5, 10, 15, 20], dtype=torch.int64)  # 5 rows
    col_indices = torch.tensor([0, 64, 128, 192], dtype=torch.int64)   # 4 columns (stick boundaries)
    
    print(f"\nGathering {len(row_indices)} rows × {len(col_indices)} columns")
    print(f"Row indices: {row_indices.tolist()}")
    print(f"Col indices: {col_indices.tolist()}")
    
    # Create 2D grid of indices using broadcasting
    # Shape will be [5, 4] for output
    row_grid = row_indices.unsqueeze(1).expand(-1, len(col_indices))  # [5, 4]
    col_grid = col_indices.unsqueeze(0).expand(len(row_indices), -1)  # [5, 4]
    
    print(f"\nRow grid shape: {row_grid.shape}")
    print(f"Col grid shape: {col_grid.shape}")
    
    # Stack to create [5, 4, 2] tensor of (row, col) pairs
    indices_2d = torch.stack([row_grid, col_grid], dim=-1).to("spyre")
    print(f"Indices 2D shape: {indices_2d.shape}")  # [5, 4, 2]
    
    # Flatten to [20, 2] for address computation
    indices_flat = indices_2d.reshape(-1, 2)
    print(f"Indices flat shape: {indices_flat.shape}")  # [20, 2]
    
    # Compute addresses for all 20 positions
    addresses = torch.ops.spyre.indices_to_address(
        indices_flat,
        virtual_offset=0,
        device_size=[32, 256],
        device_stride=[256, 1],
        element_size=2  # FP16 = 2 bytes
    )
    
    print(f"\nAddresses shape: {addresses.shape}")  # [20]
    print(f"Addresses (first 8): {addresses[:8]}")
    
    # Reshape addresses to 2D to get 2D output
    addresses_2d = addresses.reshape(len(row_indices), len(col_indices))
    print(f"Addresses 2D shape: {addresses_2d.shape}")  # [5, 4]
    
    # Compile and execute
    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input, addresses_2d)
    
    print(f"\n✓ Result shape: {result.shape}")  # Should be [5, 4]
    print("Result (2D output):")
    print(result)
    
    # Verify the result
    expected = input[row_grid, col_grid]
    print("\nExpected (2D):")
    print(expected)
    
    if torch.allclose(result.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3):
        print("\n✓ Test passed! 2D output gather working correctly.")
    else:
        print("\n✗ Test failed - results don't match")
        print(f"Difference: {(result.cpu() - expected.cpu()).abs().max()}")
    
    print("\n" + "="*60)
    print("Key Insights:")
    print("="*60)
    print(f"1. Input tensor shape: {input.shape}")
    print(f"2. Index tensor shape: {addresses_2d.shape}")
    print(f"3. Output tensor shape: {result.shape}")
    print("4. The index tensor shape determines the output shape!")
    print("5. 2D index tensor → 2D output tensor")
    print("6. Each element in the index tensor is an address to gather from")


def test_gather_2d_output_full_rows():
    """Test gathering multiple sticks per row to produce a 2D output.
    
    This demonstrates gathering stick-aligned positions from selected rows,
    producing a 2D output tensor. Note: Spyre requires stick-aligned access.
    """
    def gather_fn(input, index):
        return torch.ops.spyre.indirect_gather(input, index)

    print("\n" + "="*60)
    print("Test: Gather Stick-Aligned Positions as 2D Output")
    print("="*60)
    
    # 2D Data tensor (32 rows x 256 columns)
    input = torch.arange(8192, dtype=torch.float16, device="spyre").reshape(32, 256)
    
    print(f"Input tensor shape: {input.shape}")
    
    # Select specific rows to gather
    rows_to_gather = torch.tensor([1, 5, 10, 15, 20, 25, 30], dtype=torch.int64)
    num_rows = len(rows_to_gather)
    
    # For FP16, stick size is 64 elements (128 bytes / 2 bytes per element)
    # We can only access at columns 0, 64, 128, 192 (stick boundaries)
    stick_size = 64
    num_sticks = 4  # 256 / 64 = 4 sticks per row
    col_indices = torch.arange(num_sticks, dtype=torch.int64) * stick_size  # [0, 64, 128, 192]
    
    print(f"\nGathering {num_rows} rows × {num_sticks} stick-aligned positions")
    print(f"Row indices: {rows_to_gather.tolist()}")
    print(f"Column indices (stick boundaries): {col_indices.tolist()}")
    
    # Create indices for stick-aligned positions in selected rows
    row_grid = rows_to_gather.unsqueeze(1).expand(-1, num_sticks)  # [7, 4]
    col_grid = col_indices.unsqueeze(0).expand(num_rows, -1)  # [7, 4]
    
    # Stack and flatten for address computation
    indices_2d = torch.stack([row_grid, col_grid], dim=-1).to("spyre")  # [7, 4, 2]
    indices_flat = indices_2d.reshape(-1, 2)  # [28, 2]
    
    print(f"Total stick-aligned positions to gather: {indices_flat.shape[0]}")
    
    # Compute addresses (only at stick boundaries)
    addresses = torch.ops.spyre.indices_to_address(
        indices_flat,
        virtual_offset=0,
        device_size=[32, 256],
        device_stride=[256, 1],
        element_size=2  # FP16 = 2 bytes
    )
    
    # Reshape to 2D for 2D output
    addresses_2d = addresses.reshape(num_rows, num_sticks)
    print(f"Addresses 2D shape: {addresses_2d.shape}")  # [7, 4]
    
    # Compile and execute
    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input, addresses_2d)
    
    print(f"\n✓ Result shape: {result.shape}")  # Should be [7, 4]
    print("Result (all rows, all stick-aligned positions):")
    print(result)
    
    # Verify
    expected = input[row_grid, col_grid]
    print("\nExpected (all rows, all stick-aligned positions):")
    print(expected)
    
    if torch.allclose(result.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3):
        print("\n✓ Stick-aligned gather test passed!")
        print(f"\nSuccessfully gathered {num_rows} rows × {num_sticks} sticks as 2D output!")
        print(f"⚠️  Note: Each position represents a stick boundary (columns 0, 64, 128, 192)")
    else:
        print("\n✗ Full row gather test failed")


def test_gather_2d_output_submatrix():
    """Test gathering a stick-aligned submatrix to produce a 2D output.
    
    This demonstrates gathering a rectangular subregion at stick boundaries.
    """
    def gather_fn(input, index):
        return torch.ops.spyre.indirect_gather(input, index)

    print("\n" + "="*60)
    print("Test: Gather Stick-Aligned Submatrix as 2D Output")
    print("="*60)
    
    # 2D Data tensor (32 rows x 256 columns)
    input = torch.arange(8192, dtype=torch.float16, device="spyre").reshape(32, 256)
    
    print(f"Input tensor shape: {input.shape}")
    
    # Define submatrix region at stick boundaries
    start_row, end_row = 10, 15  # Rows 10-14 (5 rows)
    # For stick-aligned access, use columns at stick boundaries (0, 64, 128, 192)
    stick_cols = torch.tensor([64, 128], dtype=torch.int64)  # 2 stick positions
    
    print(f"\nGathering stick-aligned submatrix:")
    print(f"  Rows: {start_row} to {end_row-1} ({end_row-start_row} rows)")
    print(f"  Stick-aligned columns: {stick_cols.tolist()}")
    
    # Create indices for the submatrix
    row_indices = torch.arange(start_row, end_row, dtype=torch.int64)
    col_indices = stick_cols
    
    num_rows = len(row_indices)
    num_cols = len(col_indices)
    
    # Create 2D grid
    row_grid = row_indices.unsqueeze(1).expand(-1, num_cols)
    col_grid = col_indices.unsqueeze(0).expand(num_rows, -1)
    
    # Stack and flatten
    indices_2d = torch.stack([row_grid, col_grid], dim=-1).to("spyre")
    indices_flat = indices_2d.reshape(-1, 2)
    
    # Compute addresses (at stick boundaries)
    addresses = torch.ops.spyre.indices_to_address(
        indices_flat,
        virtual_offset=0,
        device_size=[32, 256],
        device_stride=[256, 1],
        element_size=2
    )
    
    # Reshape to 2D
    addresses_2d = addresses.reshape(num_rows, num_cols)
    print(f"Addresses 2D shape: {addresses_2d.shape}")  # [5, 2]
    
    # Compile and execute
    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input, addresses_2d)
    
    print(f"\n✓ Result shape: {result.shape}")  # Should be [5, 2]
    print("Result (all rows, all stick-aligned columns):")
    print(result)
    
    # Verify
    expected = input[row_grid, col_grid]
    print("\nExpected (all rows, all stick-aligned columns):")
    print(expected)
    
    if torch.allclose(result.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3):
        print("\n✓ Stick-aligned submatrix gather test passed!")
        print(f"\nSuccessfully gathered {num_rows}×{num_cols} stick-aligned submatrix as 2D output!")
    else:
        print("\n✗ Submatrix gather test failed")

def test_gather_2d_output_simple():
    """Test gathering multiple sticks per row to produce a 2D output.
    
    This demonstrates gathering stick-aligned positions from selected rows,
    producing a 2D output tensor. Note: Spyre requires stick-aligned access.
    """
    def gather_fn(input, index):
        return torch.ops.spyre.indirect_gather(input, index)

    print("\n" + "="*60)
    print("Test: Gather Stick-Aligned Positions as 2D Output")
    print("="*60)
    
    # 2D Data tensor (32 rows x 256 columns)
    input = torch.arange(8192, dtype=torch.float16, device="spyre").reshape(32, 256) # -> 32 rows x 4 sticks
    print(f"Input tensor shape: {input.shape}")
    
    # TODO : Understands the gathering dimension

    index = torch.tensor([[5,2], # Stick 0 group -> row 5 and 2
                          [3,1], # Stick 1 group -> row 3 and 1
                          [10,2], # Stick 2 group -> row 10 and 2
                          [12,0]], dtype=torch.float32, device="spyre") # Stick 3 group -> row 12 and 0

    # index = torch.ones(32, dtype=torch.float32, device="spyre").reshape(32,1)
    
    # Compile and execute
    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input, index)
    
    print(f"\n✓ Result shape: {result.shape}")  # Should be [7, 4]
    print("Result (all rows, all stick-aligned positions):")
    print(result)
    
    # # Verify
    # expected = input[row_grid, col_grid]
    # print("\nExpected (all rows, all stick-aligned positions):")
    # print(expected)
    
    # if torch.allclose(result.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3):
    #     print("\n✓ Stick-aligned gather test passed!")
    #     print(f"\nSuccessfully gathered {num_rows} rows × {num_sticks} sticks as 2D output!")
    #     print(f"⚠️  Note: Each position represents a stick boundary (columns 0, 64, 128, 192)")
    # else:
    #     print("\n✗ Full row gather test failed")

def test_gather_1d():
    def gather_fn(input, index):
        return torch.ops.spyre.indirect_gather(input, index)

    print("\n" + "="*60)
    print("Test: Gather Stick-Aligned Positions as 1D Output")
    print("="*60)

    input = torch.arange(32, dtype=torch.float16)
    input_2d = torch.nn.functional.pad(input.unsqueeze(1), (0,63), value=0).to("spyre")
    print(f"Input 2D tensor shape: {input_2d.shape}")

    index = torch.tensor([0,3,4,5], dtype=torch.float32).to("spyre")
    #index_2d = torch.nn.functional.pad(index.unsqueeze(1), (0,63), value=0).to("spyre")
    print("Index shape : ", index.shape)

    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_2d, index)
    
    print(f"\n✓ Result shape: {result.shape}")  # Should be [7, 4]
    print("Result (all rows, all stick-aligned positions):")
    print(result)

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("2D Output Tensor Tests for Indirect Gather")
    print("=" * 70)
    
    try:
        #test_gather_2d_output()
        #test_gather_2d_output_full_rows()
        #test_gather_2d_output_submatrix()
        #test_gather_2d_output_simple()
        test_gather_1d()
        
        print("\n" + "=" * 70)
        print("Summary: How to Get 2D Output from Indirect Gather")
        print("=" * 70)
        print("\n✓ Key Principle: Index tensor shape determines output shape!")
        print("\nStrategies demonstrated:")
        print("1. Sparse 2D gather: Use 2D index tensor [M, N] → output [M, N]")
        print("2. Stick-aligned gather: Index shape [num_rows, num_sticks] → output [num_rows, num_sticks]")
        print("3. Submatrix gather: Index shape [rows, cols] → output [rows, cols]")
        print("\nImplementation steps:")
        print("1. Create 2D grid of (row, col) indices AT STICK BOUNDARIES")
        print("2. Flatten to compute addresses")
        print("3. Reshape addresses to desired 2D output shape")
        print("4. Pass reshaped addresses to indirect_gather")
        print("5. Output will have the same shape as the address tensor!")
        print("\n⚠️  CRITICAL: Spyre requires stick-aligned access (128-byte boundaries)")
        print("   For FP16: Only access columns at multiples of 64 (0, 64, 128, 192, ...)")
        print("   For FP32: Only access columns at multiples of 32 (0, 32, 64, 96, ...)")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: This test requires:")
        print("- torch_spyre.indirect_gather operation to be implemented")
        print("- torch_spyre.indices_to_address C++ function")
        print("- Proper handling of 2D index tensors in the backend")


# import torch

# def test_gather_with_addresses():
#     """Test 2D gather operation along dimension 0 (rows) with stick groups.
    
#     This test demonstrates gathering rows from different stick groups.
#     For a [32, 128] FP16 tensor:
#     - 32 rows × 128 columns = 4096 elements
#     - Each stick holds 64 FP16 elements (128 bytes)
#     - Each row spans 2 sticks (128 elements / 64 elements per stick)
#     - Total: 64 sticks organized into 2 stick groups (32 sticks each)
    
#     2D Index tensor [[2, 3, 1], [1, 5, 7]] means:
#     - From stick group 0: gather rows 2, 3, 1
#     - From stick group 1: gather rows 1, 5, 7
    
#     Output shape: [2, 3, 128] (2 stick groups × 3 rows × 128 columns)
#     """
#     def gather_fn(input, index):
#         return torch.ops.spyre.indirect_gather(input, index)

#     # 2D Data tensor (32 rows x 128 columns = 4096 elements)
#     # Organized into 2 stick groups, each with 32 sticks (16 rows × 2 sticks per row)
#     input = torch.arange(4096, dtype=torch.float16, device="spyre").reshape(32, 128)
    
#     print("Input tensor shape:", input.shape)
#     # print("Input tensor (first 3 rows, first 8 cols):")
#     # print(input[:3, :8])
#     print("\nHBM Layout:")
#     print("- Total elements: 4096")
#     print("- Elements per stick: 64")
#     print("- Total sticks: 64")
#     print("- Stick groups: 2 (each with 32 sticks)")
#     print("- Stick group 0: sticks 0-31 (rows 0-15)")
#     print("- Stick group 1: sticks 32-63 (rows 16-31)")
    
#     # 2D index tensor for gathering along dimension 0
#     # Shape [2, 3]: 2 stick groups, 3 rows from each group
#     # [[2, 3, 1],   -> From stick group 0: gather rows 2, 3, 1
#     #  [1, 5, 7]]   -> From stick group 1: gather rows 1, 5, 7
#     row_indices = torch.tensor([[2, 3, 1],
#                                  [1, 5, 7]], dtype=torch.int64)
    
#     print("\n2D Index tensor (rows to gather from each stick group):")
#     print(row_indices)
#     print("Shape:", row_indices.shape)
#     print("\nGathering:")
#     print("- Stick group 0: rows [2, 3, 1]")
#     print("- Stick group 1: rows [1, 5, 7]")
    
#     # Use the C++ function to compute stick addresses
#     # Each row index maps to ONE stick address (the first stick of that row)
#     virtual_offset = 0  # Base address of tensor in HBM (in bytes)
#     rows_per_group = 16  # Each stick group has 16 rows
#     cols_per_row = 128  # Number of columns per row
#     element_size = 2  # FP16 = 2 bytes per element
    
#     print("\nComputing stick addresses using C++ function:")
#     print(f"- virtual_offset: {virtual_offset}")
#     print(f"- rows_per_group: {rows_per_group}")
#     print(f"- cols_per_row: {cols_per_row}")
#     print(f"- element_size: {element_size}")
    
#     # Call the new custom op
#     addresses = torch.ops.spyre.rows_to_stick_addresses(
#         row_indices.to("spyre"),
#         virtual_offset,
#         rows_per_group,
#         cols_per_row,
#         element_size
#     )
    
#     print("\nComputed stick addresses:")
#     print("Shape:", addresses.shape, "(same as index tensor)")
#     print(addresses)
    
#     # Print details for verification
#     print("\nAddress details:")
#     for group in range(row_indices.shape[0]):
#         print(f"Stick group {group}:")
#         base_row = group * rows_per_group
#         for idx in range(row_indices.shape[1]):
#             row_in_group = row_indices[group, idx].item()
#             absolute_row = base_row + row_in_group
#             stick_addr = addresses[group, idx].item()
#             print(f"  Row {row_in_group} (absolute row {absolute_row}): stick_addr={int(stick_addr):3d}")
    
#     # Compile and execute
#     # The gather operation returns gathered sticks
#     # Output shape: [2, 3, 128] (2 groups × 3 rows × 128 columns)
#     compiled_fn = torch.compile(gather_fn)
#     result = compiled_fn(input, addresses)
    
#     print("\nResult shape:", result.shape)
#     print("Expected shape: [2, 3, 128] (2 stick groups × 3 rows × 128 columns)")
    
#     # Verify the result
#     # Expected: gather the specified rows from each stick group
#     expected_rows = []
#     for stick_group in range(row_indices.shape[0]):
#         base_row = stick_group * rows_per_group
#         group_rows = []
#         for row_in_group in row_indices[stick_group].tolist():
#             absolute_row = base_row + row_in_group
#             group_rows.append(input[absolute_row])
#         expected_rows.append(torch.stack(group_rows))
#     expected = torch.stack(expected_rows)
    
#     print("\nExpected shape:", expected.shape)
#     print("Expected (stick group 0, first 8 cols of each row):")
#     print(expected[0, :, :8])
    
#     if result.shape == expected.shape:
#         print("\n✓ Shape matches!")
#         if torch.allclose(result.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3):
#             print("✓ Values match!")
#             print("\n✓ Test passed! 2D gather with stick groups working correctly.")
#             print("\nKey points:")
#             print("- Input tensor: 2D [32, 128] = 4096 FP16 elements")
#             print("- Stick groups: 2 (each with 16 rows = 32 sticks)")
#             print("- Index tensor: 2D [2, 3] specifying rows from each stick group")
#             print("- Address tensor: [2, 3] = same shape as index tensor")
#             print("- Output: [2, 3, 128] = gathered rows from each stick group")
#             print("- Used C++ function: torch.ops.spyre.rows_to_stick_addresses")
#         else:
#             print("\n✗ Values don't match")
#             print(f"Difference: {(result.cpu() - expected.cpu()).abs().max()}")
#     else:
#         print(f"\n✗ Shape mismatch: got {result.shape}, expected {expected.shape}")

# def test_gather_simple():
#     def gather_fn(input, index):
#         return torch.ops.spyre.indirect_gather(input, index)

#     input = torch.arange(8192, dtype=torch.float16, device="spyre").reshape(32, 256)
#     print("Input tensor shape:", input.shape)

#     index = torch.ones(32, dtype=torch.float32, device="spyre").reshape(32,1)
#     print("Index : ", index)

#     compiled_fn = torch.compile(gather_fn)
#     result = compiled_fn(input, index)

#     print("Result : ", result)

# if __name__ == "__main__":    
#     print("\n" + "=" * 60)
#     print("Test: 2D Gather with Stick Groups using C++ Address Computation")
#     print("=" * 60)
#     try:
#         #test_gather_with_addresses()
#         test_gather_simple()
#     except Exception as e:
#         print(f"\nError: {e}")
#         import traceback
#         traceback.print_exc()
#         print("\nNote: This test requires:")
#         print("- torch_spyre.indirect_gather operation to be implemented")
#         print("- torch_spyre.rows_to_stick_addresses C++ function")
#         print("- Support for 2D index tensors with stick groups")
#         print("- Gather must operate at stick level (64 elements per stick)")
