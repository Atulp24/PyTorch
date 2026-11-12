import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    print("GPU Name:", torch.cuda.get_device_name(0))
    x = torch.rand(3, 3, device=device)
    print("Sample GPU Tensor:\n", x)