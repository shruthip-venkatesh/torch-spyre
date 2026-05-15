import torch

def test_gather_1d():

    print("\n" + "=" * 70)
    print("1D Output Tensor Tests for Indirect Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        result_2d = torch.gather(input, dim, index)
        # Squeeze back to 1D for output
        return result_2d.squeeze(-1)

    input_tensor = torch.randn(64 , dtype=torch.float16)
    input_tensor = torch.nn.functional.pad(input_tensor.reshape(64,1), (0, 63), value=0).to("spyre")
    print("Input Tensor Shape:", input_tensor.shape)
    
    index_tensor = torch.tensor([0, 10, 20, 30], dtype=torch.int64)
    index_tensor = torch.nn.functional.pad(index_tensor.reshape(4,1), (0, 31), value=0).to("spyre")
    print("Index Tensor Shape:", index_tensor.shape)  # Should be torch.Size([4])

    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)
    
    print("Result Shape:", result.shape)
    print("Result:", result)
    
    return result


def test_gather_2d():

    print("\n" + "=" * 70)
    print("2D Output Tensor Tests for Indirect Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.gather(input, dim, index)

    input_tensor = torch.arange(32 * 128, dtype=torch.float16).reshape(32, 128).to("spyre") # => (32, 128, STICK_SIZE)
    
    print("\nInput tensor shape:", input_tensor.shape)
    print("Input : ", input_tensor)

    index_tensor = torch.tensor([[5, 10],
                                  [15, 20],
                                  [25, 30],
                                  [2, 7]], dtype=torch.int64).to("spyre")
    print("\nIndex tensor shape:", index_tensor.shape)
    print("Index tensor :", index_tensor)
    #addresses_tensor = torch.ops.spyre.indices_to_addresses(index_tensor, input_tensor, 0)

    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)
    
    print("\n" + "-"*60)
    print("Result shape:", result.shape)
    print("Result: ", result)


def test_gather_3d():

    print("\n" + "=" * 70)
    print("3D Output Tensor Tests for Indirect Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.gather(input, dim, index)

    input_tensor = torch.arange(8 * 32 * 64, dtype=torch.float16).reshape(8, 32, 64).to("spyre")
    
    print("\nInput tensor shape:", input_tensor.shape)
    print("Input : ", input_tensor)

    index_tensor = torch.tensor([[[5, 10, 15, 20],
                                   [25, 30, 2, 7],
                                   [12, 18, 22, 28]],
                                  [[3, 8, 13, 17],
                                   [21, 26, 1, 6],
                                   [11, 16, 23, 29]]], dtype=torch.int64).to("spyre")
    print("\nIndex tensor shape:", index_tensor.shape)
    print("Index tensor :", index_tensor)
    
    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)
    
    print("\n" + "-"*60)
    print("Result shape:", result.shape)
    print("Result: ", result)


def test_gather_4d():

    print("\n" + "=" * 70)
    print("4D Output Tensor Tests for Indirect Gather")
    print("=" * 70)

    def gather_fn(input, dim, index):
        return torch.gather(input, dim, index)
    
    # Create 2D input tensor [32 rows x 64 columns]
    # Each row is exactly 1 stick (64 FP16 elements = 128 bytes)
    input_tensor = torch.arange(4 * 8 * 32 * 64, dtype=torch.float16).reshape(4, 8, 32, 64).to("spyre")
    
    print("\nInput tensor shape:", input_tensor.shape)
    print("Input : ", input_tensor)

    index_tensor = torch.tensor([[[[5, 10],
                                    [15, 20],
                                    [25, 30]],
                                   [[2, 7],
                                    [12, 18],
                                    [22, 28]]],
                                  [[[3, 8],
                                    [13, 17],
                                    [21, 26]],
                                   [[1, 6],
                                    [11, 16],
                                    [23, 29]]]], dtype=torch.int64).to("spyre")
    print("\nIndex tensor shape:", index_tensor.shape)
    print("Index tensor :", index_tensor)
    
    compiled_fn = torch.compile(gather_fn)
    result = compiled_fn(input_tensor, 0, index_tensor)
    
    print("\n" + "-"*60)
    print("Result shape:", result.shape)
    print("Result: ", result)


if __name__ == "__main__":

    try:
        #test_gather_1d()
        test_gather_2d()
        test_gather_3d()
        #test_gather_4d()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
