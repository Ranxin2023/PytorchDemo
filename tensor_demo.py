import torch
def basic_tensor_operations():
    print("basic tensor operations")
    x = torch.tensor([1, 2, 3])
    print(x)
    
def reshape_demo():
    print("reshape of tensor......")
    # Original tensor of shape (2, 3)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("Original tensor (2x3):")
    print(x)
    print("Shape:", x.shape)

    # Reshape to (3, 2)
    reshaped_1 = x.reshape(3, 2)
    print("\nReshaped to (3x2):")
    print(reshaped_1)
    print("Shape:", reshaped_1.shape)

    # Reshape to (1, 6)
    reshaped_2 = x.reshape(1, 6)
    print("\nReshaped to (1x6):")
    print(reshaped_2)
    print("Shape:", reshaped_2.shape)

    # Reshape to (6,) — flatten
    reshaped_3 = x.reshape(6)
    print("\nReshaped to (6,) — flattened:")
    print(reshaped_3)
    print("Shape:", reshaped_3.shape)

def tensor_demo():
    reshape_demo()