import torch

checkpoint = torch.load('best_model_20241107_004525_binary_Transformer_100k_0.9460.pth')
print(f"Type of checkpoint: {type(checkpoint)}")
for key, value in checkpoint.items():
    print(f"{key}: {type(value)}")
    if hasattr(value, 'size'):
        print(f"Size in MB: {value.element_size() * value.nelement() / (1024*1024):.2f}")