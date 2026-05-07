import torch

def test_indirect_add_address():
    """Test indirect_add using stick addresses as expected by DeepTools.

    DeepTools indirect access operates at stick granularity (128 bytes).
    For FP16 data, each stick holds 64 elements (128 bytes / 2 bytes per element).

    Memory layout for 128 FP16 elements in [2, 64] device layout:
    - Stick 0: elements 0-63   (row 0)
    - Stick 1: elements 64-127 (row 1)
    """
    def indirect_add_fn(input_a, address_a, input_b, address_b):
        return torch.ops.spyre.indirect_add(input_a, address_a, input_b, address_b)

    # Data tensors (128 FP16 elements = 2 sticks)
    input_a = torch.randn(128, dtype=torch.float16, device="spyre")
    input_b = torch.randn(128, dtype=torch.float16, device="spyre")

    # IMPORTANT: DeepTools indirect access only works at stick boundaries!
    # Each stick holds 64 FP16 elements (128 bytes / 2 bytes per element)
    # We can only access elements at stick boundaries: 0, 64, 128, etc.
    # Stick 0: elements 0-63, Stick 1: elements 64-127
    index1 = torch.tensor([0, 64, 0, 64], dtype=torch.int64)   # Only stick-aligned indices
    index2 = torch.tensor([0, 0, 64, 64], dtype=torch.int64)   # Only stick-aligned indices

    print(f"Index 1: {index1.tolist()} (sticks: {[i//64 for i in index1.tolist()]})")
    print(f"Index 2: {index2.tolist()} (sticks: {[i//64 for i in index2.tolist()]})")
    print("Note: Indices must be stick-aligned (multiples of 64 elements for FP16)")

    # Convert logical indices to 2D device indices
    # For 1D tensor of 128 elements mapped to [2, 64]:
    # index -> [index // 64, index % 64]
    indices1_2d = torch.stack([index1 // 64, index1 % 64], dim=1).to("spyre")
    indices2_2d = torch.stack([index2 // 64, index2 % 64], dim=1).to("spyre")
    
    # Convert to stick addresses
    # DeepTools expects stick addresses (in units of 128 bytes), not byte addresses
    # The indices_to_address function now correctly converts to stick addresses
    address_a = torch.ops.spyre.indices_to_address(
        indices1_2d,
        virtual_offset=0,  # Base address for input_a (in bytes, will be converted to sticks)
        device_size=[2, 64],
        device_stride=[64, 1],
        element_size=2  # FP16 = 2 bytes
    )
    print(f"Stick addresses for input_a: {address_a}")
    
    address_b = torch.ops.spyre.indices_to_address(
        indices2_2d,
        virtual_offset=34359738368,  # Base address for input_b (in bytes, will be converted to sticks)
        device_size=[2, 64],
        device_stride=[64, 1],
        element_size=2  # FP16 = 2 bytes
    )
    print(f"Stick addresses for input_b: {address_b}")

    # Compile and execute
    compiled_fn = torch.compile(indirect_add_fn)
    result = compiled_fn(input_a, address_a, input_b, address_b)

    print(f"\nResult: {result}")

    # Verify the result manually
    expected = input_a[index1] + input_b[index2]
    print(f"Expected: {expected}")

    # Check if results match (with some tolerance for FP operations)
    if torch.allclose(result.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3):
        print("\n✓ Test passed! Indirect access working correctly.")
    else:
        print("\n✗ Test failed - results don't match")
        print(f"Difference: {(result.cpu() - expected.cpu()).abs().max()}")
        print(f"Result values: {result.cpu()}")
        print(f"Expected values: {expected.cpu()}")

if __name__ == "__main__":
    test_indirect_add_address()
