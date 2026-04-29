#!/usr/bin/env python3
"""
Test script for the C++ implementation of indices_to_address operator.
This tests the operator with torch.compile to ensure it works correctly.
"""

import torch

def test_indices_to_address_cpp():
    """Test the C++ implementation of indices_to_address"""
    print("Testing indices_to_address with C++ implementation...")
    
    # Create a simple 2D tensor layout
    # Shape: [3, 4], strides: [4, 1] (row-major)
    device_size = [3, 4]
    device_stride = [4, 1]
    element_size = 4  # float32
    virtual_offset = 0x0000
    
    # Create indices tensor - use int32 to avoid downcast warnings
    indices = torch.tensor([
        [0, 0],  # First element: offset = 0*4 + 0*1 = 0
        [0, 1],  # Second element: offset = 0*4 + 1*1 = 1
        [1, 0],  # Start of second row: offset = 1*4 + 0*1 = 4
        [2, 3],  # Last element: offset = 2*4 + 3*1 = 11
    ], dtype=torch.int64).to("spyre")
    
    # Expected addresses (in bytes)
    # address = virtual_offset + (element_offset * element_size)
    expected = torch.tensor([
        virtual_offset + 0 * 4,   # 0x0000
        virtual_offset + 1 * 4,   # 0x0004
        virtual_offset + 4 * 4,   # 0x0010
        virtual_offset + 11 * 4,  # 0x002C
    ], dtype=torch.float32)
    
    # Call the operator directly (without compile)
    print("\n1. Testing direct call (no compile)...")
    addresses = torch.ops.spyre.indices_to_address(
        indices, virtual_offset, device_size, device_stride, element_size
    )
    
    print(f"Indices:\n{indices}")
    print(f"Addresses (hex): {[hex(int(a)) for a in addresses]}")
    print(f"Expected (hex):  {[hex(int(e)) for e in expected]}")
    
    if torch.allclose(addresses, expected):
        print("✓ Direct call test PASSED")
    else:
        print("✗ Direct call test FAILED")
        print(f"Difference: {addresses - expected}")
        return False
    
    # Test with torch.compile
    print("\n2. Testing with torch.compile...")
    
    @torch.compile
    def compute_addresses(indices, virtual_offset, device_size, device_stride, element_size):
        return torch.ops.spyre.indices_to_address(
            indices, virtual_offset, device_size, device_stride, element_size
        )
    
    try:
        compiled_addresses = compute_addresses(
            indices, virtual_offset, device_size, device_stride, element_size
        )
        
        print(f"Compiled addresses (hex): {[hex(int(a)) for a in compiled_addresses]}")
        
        if torch.allclose(compiled_addresses, expected):
            print("✓ Compiled test PASSED")
        else:
            print("✗ Compiled test FAILED")
            print(f"Difference: {compiled_addresses - expected}")
            return False
            
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All tests PASSED!")
    return True

if __name__ == "__main__":
    success = test_indices_to_address_cpp()
    exit(0 if success else 1)
