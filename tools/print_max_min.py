def print_min_max(tensor, name="tensor"):
    print(f"{name} max: {tensor.max().item():.4f}, min: {tensor.min().item():.4f}")