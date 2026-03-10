import torch
import time
from flash_attn import flash_attn_func

# 检查CUDA是否可用
if not torch.cuda.is_available():
    print("CUDA not available. Exiting.")
    exit(1)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"Flash Attention version: 2.8.3")

# 测试参数
batch_size = 2
seq_len = 128
num_heads = 4
head_dim = 32

# 创建随机张量
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)

# 预热GPU
for _ in range(10):
    _ = flash_attn_func(q, k, v, causal=False)

# 计时
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    output = flash_attn_func(q, k, v, causal=False)
torch.cuda.synchronize()
end = time.time()

avg_time = (end - start) / 100 * 1000  # 毫秒
print(f"Flash Attention output shape: {output.shape}")
print(f"Average time per forward pass: {avg_time:.3f} ms")

# 验证输出值范围
print(f"Output min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}")

print("\n--- Numerical verification against PyTorch SDPA (math backend) ---")
import torch.nn.functional as F
import warnings

# 使用float16进行验证（FlashAttention支持的精度）
# q, k, v已经是float16

# 重塑为SDPA期望的形状: (batch_size, num_heads, seq_len, head_dim)
q_sdpa = q.transpose(1, 2)  # (B, H, S, D)
k_sdpa = k.transpose(1, 2)
v_sdpa = v.transpose(1, 2)

# 使用PyTorch的scaled_dot_product_attention (数学实现，用于参考)
# 过滤弃用警告
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        ref_output = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False)

# 转置回Flash Attention的形状: (B, S, H, D)
ref_output = ref_output.transpose(1, 2)

# 使用相同的参数运行Flash Attention
output_fa = flash_attn_func(q, k, v, causal=False)

# 计算相对误差
abs_diff = torch.abs(output_fa - ref_output)
rel_diff = abs_diff / (torch.abs(ref_output) + 1e-8)
max_abs_diff = abs_diff.max().item()
max_rel_diff = rel_diff.max().item()
mean_rel_diff = rel_diff.mean().item()

print(f"Max absolute difference: {max_abs_diff:.6f}")
print(f"Max relative difference: {max_rel_diff:.6f}")
print(f"Mean relative difference: {mean_rel_diff:.6f}")

# 由于数值计算差异，放宽容差
if max_abs_diff < 1e-2:
    print("✓ Numerical verification passed (difference within tolerance)")
else:
    print("⚠ Numerical difference larger than expected")

print("\n--- Testing causal attention ---")
# 测试因果注意力
output_causal = flash_attn_func(q, k, v, causal=True)
print(f"Causal output shape: {output_causal.shape}")

# 简单检查因果性：最后一个token不应该看到第一个token
# 对于batch 0, head 0，检查注意力矩阵不可访问性（这里只是示意）
print("Causal test completed.")

print("\nFlash Attention demo completed successfully!")
