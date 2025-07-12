import torch
# import intel_extension_for_pytorch as ipex

# Kiểm tra xem XPU có khả dụng không
print(f"XPU available: {torch.xpu.is_available()}")
print(f"XPU device count: {torch.xpu.device_count()}")

# Tạo tensor và chuyển sang XPU
device = torch.device("xpu")
x = torch.randn(3, 4).to(device)
y = torch.randn(4, 5).to(device)

# Thực hiện phép tính trên XPU
result = torch.mm(x, y)
print(f"Result device: {result.device}")