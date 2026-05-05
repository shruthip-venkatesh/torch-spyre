import torch

def test_gather_with_addresses():
    """Test gather operation using address computation similar to add_indirect."""
    def gather_fn(input, index):
        return torch.ops.spyre.indirect_gather(input, index)

    # Data tensor (100 elements)
    input = torch.randn(100, dtype=torch.float16, device="spyre")
    
    # Logical indices (which elements to gather)
    logical_index = torch.tensor([0, 10, 20, 30, 40], dtype=torch.int64)
    
    # Convert logical indices to 2D device indices
    # For 1D tensor of 100 elements mapped to [2, 64]:
    # index -> [index // 64, index % 64]
    indices_2d = torch.stack([logical_index // 64, logical_index % 64], dim=1).to("spyre")
    
    # Compute addresses
    addresses = torch.ops.spyre.indices_to_address(
        indices_2d,
        virtual_offset=0,
        device_size=[2, 64],
        device_stride=[64, 1],
        element_size=2  # FP16 = 2 bytes
    )
    
    print("\nLogical indices:", logical_index)
    print("Address tensor:", addresses)
    
    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input, addresses)
    
    print("Result shape:", result.shape)
    print("Result:", result)


if __name__ == "__main__":    
    print("\n" + "=" * 60)
    print("Test : Gather with address computation")
    print("=" * 60)
    test_gather_with_addresses()
