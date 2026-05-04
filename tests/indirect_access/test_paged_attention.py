import torch
import torch._dynamo
torch._dynamo.reset()

def main():
    print("=== Paged Attention Demo ===\n")
    
    # Create simple input tensors on Spyre device
    input_a = torch.randn(100, dtype=torch.float16, device="spyre")
    input_b = torch.randn(100, dtype=torch.float16, device="spyre")
    
    print(f"Input1 shape: {input_a.shape}")
    print(f"Input2 shape: {input_b.shape}")
    
    # Create index tensor
    index1 = torch.tensor([0, 10, 20, 30], dtype=torch.int64, device="spyre")
    index2 = torch.tensor([20, 40, 60, 30], dtype=torch.int64, device="spyre")

    @torch.compile
    def fn(input1, index1, input2, index2):
        return torch.ops.spyre.paged_attention(input1, index1, input2, index2)

    # Call the paged_attention operator
    print("\n=== Calling paged_attention ===")
    output = fn(input_a, index1, input_b, index1)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
