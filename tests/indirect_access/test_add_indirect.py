import torch

def test_indirect_add_address():
    def indirect_add_fn(input_a, index_a, input_b, index_b):
        return torch.ops.spyre.indirect_add(input_a, index_a, input_b, index_b)

    # Data tensors (100 elements)
    input_a = torch.randn(128, dtype=torch.float16, device="spyre")
    input_b = torch.randn(128, dtype=torch.float16, device="spyre")

    # Logical indices (which elements to access)
    index1 = torch.tensor([0, 10, 20, 30], dtype=torch.int64)
    index2 = torch.tensor([20, 40, 60, 30], dtype=torch.int64)

    # From SDSC JSON: input_a has device_size=[2, 64], stride=[64, 1]
    # Virtual offset for A: 0, for B: 34359738368
    # Element size for FP16: 2 bytes
    
    # Convert logical indices to 2D device indices
    # For 1D tensor of 100 elements mapped to [2, 64]:
    # index -> [index // 64, index % 64]
    indices1_2d = torch.stack([index1 // 64, index1 % 64], dim=1).to("spyre")
    indices2_2d = torch.stack([index2 // 64, index2 % 64], dim=1).to("spyre")
    
    address_a = torch.ops.spyre.indices_to_address(
        indices1_2d,
        virtual_offset=0,
        device_size=[2, 64],
        device_stride=[64, 1],
        element_size=2  # FP16 = 2 bytes
    )
    print("Address Tensor for Index 1 : ", address_a )
    
    address_b = torch.ops.spyre.indices_to_address(
        indices2_2d,
        virtual_offset=34359738368,
        device_size=[2, 64],
        device_stride=[64, 1],
        element_size=2  # FP16 = 2 bytes
    )
    print("Address Tensor for Index 2 : ", address_b )

    compiled_fn = torch.compile(indirect_add_fn)
    result = compiled_fn(input_a, address_a, input_b, address_b)

    print(result)

if __name__ == "__main__":
    test_indirect_add_address()
