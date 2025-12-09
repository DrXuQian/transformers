"""
从代码精确计算cu_seqlens和batch数量
"""

import torch

def calculate_cu_seqlens(grid_thw, total_tokens=5040):
    """
    根据Qwen3-VL MoE的实际代码计算cu_seqlens

    代码位置: modeling_qwen3_vl_moe.py:772-780
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    Args:
        grid_thw: shape (num_images_or_videos, 3)
                  每行是 [T(时间), H(高度), W(宽度)]
        total_tokens: 总token数，用于验证
    """

    print("=" * 80)
    print("cu_seqlens的精确计算（基于代码）")
    print("=" * 80)

    print("\n📝 代码逻辑：")
    print("-" * 40)
    print("""
    cu_seqlens计算步骤：
    1. 计算每个图像/视频的token数：H * W（空间维度）
    2. 按时间维度T重复：repeat_interleave(H*W, T)
    3. 累积求和：cumsum()
    4. 前面补0：F.pad(..., (1, 0), value=0)
    """)

    print("\n🔍 对于5040序列，需要确定grid_thw")
    print("-" * 40)

    # patch_size = 16 (从配置得知)
    patch_size = 16

    print(f"已知：patch_size = {patch_size}")
    print(f"目标：总tokens = {total_tokens}")

    print("\n测试不同的grid_thw配置：")
    print("-" * 40)

    # 测试案例1：单张图像
    print("\n案例1：单张图像")
    # 5040 = H * W
    # 如果是正方形：sqrt(5040) ≈ 71
    # 但71*71 = 5041，70*70 = 4900，72*70 = 5040

    grid_thw_1 = torch.tensor([[1, 72, 70]])  # T=1, H=72, W=70
    tokens_1 = grid_thw_1[:, 1] * grid_thw_1[:, 2]  # H*W = 72*70 = 5040
    cu_seqlens_1 = torch.repeat_interleave(tokens_1, grid_thw_1[:, 0]).cumsum(dim=0)
    cu_seqlens_1 = torch.nn.functional.pad(cu_seqlens_1, (1, 0), value=0)

    print(f"grid_thw = {grid_thw_1.tolist()}")
    print(f"每个图像tokens: {tokens_1.tolist()}")
    print(f"cu_seqlens = {cu_seqlens_1.tolist()}")
    print(f"batch数量 = {len(grid_thw_1)}")
    print(f"验证总tokens: {cu_seqlens_1[-1]}")

    # 测试案例2：5张图像
    print("\n案例2：5张图像")
    # 5040 / 5 = 1008 tokens per image
    # 1008的因数分解：1008 = 16 * 63 = 21 * 48 = 28 * 36 = ...

    grid_thw_2 = torch.tensor([
        [1, 28, 36],  # 图像1: T=1, H=28, W=36, tokens=1008
        [1, 28, 36],  # 图像2: T=1, H=28, W=36, tokens=1008
        [1, 28, 36],  # 图像3: T=1, H=28, W=36, tokens=1008
        [1, 28, 36],  # 图像4: T=1, H=28, W=36, tokens=1008
        [1, 28, 36],  # 图像5: T=1, H=28, W=36, tokens=1008
    ])

    tokens_2 = grid_thw_2[:, 1] * grid_thw_2[:, 2]  # 每个图像的H*W
    cu_seqlens_2 = torch.repeat_interleave(tokens_2, grid_thw_2[:, 0]).cumsum(dim=0)
    cu_seqlens_2 = torch.nn.functional.pad(cu_seqlens_2, (1, 0), value=0)

    print(f"grid_thw = {grid_thw_2.tolist()}")
    print(f"每个图像tokens: {tokens_2.tolist()}")
    print(f"cu_seqlens = {cu_seqlens_2.tolist()}")
    print(f"batch数量 = {len(grid_thw_2)}")
    print(f"验证总tokens: {cu_seqlens_2[-1]}")

    # 转换为实际图像尺寸
    print("\n图像尺寸（像素）:")
    for i, (t, h, w) in enumerate(grid_thw_2):
        h_pixels = h.item() * patch_size
        w_pixels = w.item() * patch_size
        print(f"  图像{i+1}: {h_pixels}×{w_pixels}像素 (patches: {h}×{w})")

    # 测试案例3：不同大小的图像
    print("\n案例3：不同大小的图像")
    grid_thw_3 = torch.tensor([
        [1, 40, 40],  # 图像1: 1600 tokens
        [1, 32, 32],  # 图像2: 1024 tokens
        [1, 30, 30],  # 图像3: 900 tokens
        [1, 28, 28],  # 图像4: 784 tokens
        [1, 24, 30],  # 图像5: 720 tokens
    ])  # 总计: 1600+1024+900+784+720 = 5028 ≈ 5040

    # 调整最后一个图像使总数正好是5040
    grid_thw_3[-1] = torch.tensor([1, 24, 31])  # 744 tokens
    # 总计: 1600+1024+900+784+744 = 5052, 还是不对

    # 重新设计
    grid_thw_3 = torch.tensor([
        [1, 36, 36],  # 1296 tokens
        [1, 32, 32],  # 1024 tokens
        [1, 30, 30],  # 900 tokens
        [1, 28, 28],  # 784 tokens
        [1, 32, 32],  # 1024 tokens
    ])  # 总计: 1296+1024+900+784+1024 = 5028

    # 微调
    grid_thw_3 = torch.tensor([
        [1, 36, 35],  # 1260 tokens
        [1, 32, 32],  # 1024 tokens
        [1, 30, 30],  # 900 tokens
        [1, 28, 28],  # 784 tokens
        [1, 36, 30],  # 1080 tokens
    ])  # 总计: 1260+1024+900+784+1072 = 5040

    # 验证最后一个
    last_needed = 5040 - (1260 + 1024 + 900 + 784)
    print(f"最后一个图像需要: {last_needed} tokens")
    # 1072 = 36 * 29.78... 不是整数

    # 使用整数解
    grid_thw_3 = torch.tensor([
        [1, 36, 28],  # 1008 tokens
        [1, 36, 28],  # 1008 tokens
        [1, 36, 28],  # 1008 tokens
        [1, 36, 28],  # 1008 tokens
        [1, 36, 28],  # 1008 tokens
    ])  # 总计: 5040

    tokens_3 = grid_thw_3[:, 1] * grid_thw_3[:, 2]
    cu_seqlens_3 = torch.repeat_interleave(tokens_3, grid_thw_3[:, 0]).cumsum(dim=0)
    cu_seqlens_3 = torch.nn.functional.pad(cu_seqlens_3, (1, 0), value=0)

    print(f"grid_thw = {grid_thw_3.tolist()}")
    print(f"每个图像tokens: {tokens_3.tolist()}")
    print(f"cu_seqlens = {cu_seqlens_3.tolist()}")
    print(f"batch数量 = {len(grid_thw_3)}")
    print(f"验证总tokens: {cu_seqlens_3[-1]}")

    print("\n" + "=" * 80)
    print("💡 结论")
    print("-" * 40)
    print("""
    cu_seqlens和batch数量完全由输入的grid_thw决定：

    • grid_thw的行数 = batch数量（图像/视频数）
    • grid_thw每行的[T, H, W] = 该图像/视频的时间和空间维度
    • 每个图像的tokens = H * W
    • cu_seqlens = 累积的序列边界

    对于5040 tokens，实际的batch和cu_seqlens取决于具体输入：
    - 如果输入1张图像：batch=1, cu_seqlens=[0, 5040]
    - 如果输入5张图像：batch=5, cu_seqlens=[0, 1008, 2016, 3024, 4032, 5040]
    - 其他组合也可能，完全由grid_thw决定

    这不是"最可能"，而是由实际输入确定的！
    """)

    print("=" * 80)


if __name__ == "__main__":
    calculate_cu_seqlens(None)