"""
分析5040长度序列在Qwen2.5-VL和Qwen3-VL MoE中的批处理
"""

import math

def analyze_5040_sequence():
    """
    详细分析5040长度序列的批处理
    """

    print("=" * 80)
    print("5040序列长度的批处理分析")
    print("=" * 80)

    # 首先分析5040可能对应的图像配置
    print("\n📐 1. 5040序列长度可能的图像配置")
    print("-" * 40)

    print("\n### Qwen2.5-VL (patch_size=14):")
    print("每个patch: 14×14像素")

    # 计算可能的图像尺寸
    total_patches_25 = 5040

    # 尝试不同的可能性
    possible_configs_25 = []

    # 单张图像
    sqrt_patches = math.sqrt(total_patches_25)
    if sqrt_patches == int(sqrt_patches):
        h_patches = w_patches = int(sqrt_patches)
        img_size = h_patches * 14
        possible_configs_25.append({
            'num_images': 1,
            'patches_per_image': total_patches_25,
            'h_patches': h_patches,
            'w_patches': w_patches,
            'image_size': f"{img_size}×{img_size}"
        })

    # 多张图像的可能性
    for num_images in [2, 3, 4, 5, 6, 7, 8, 10]:
        if total_patches_25 % num_images == 0:
            patches_per_img = total_patches_25 // num_images
            sqrt_p = math.sqrt(patches_per_img)
            if sqrt_p == int(sqrt_p):
                h_p = w_p = int(sqrt_p)
                img_s = h_p * 14
                possible_configs_25.append({
                    'num_images': num_images,
                    'patches_per_image': patches_per_img,
                    'h_patches': h_p,
                    'w_patches': w_p,
                    'image_size': f"{img_s}×{img_s}"
                })

    print("\n可能的配置:")
    for config in possible_configs_25:
        print(f"• {config['num_images']}张 {config['image_size']}图像")
        print(f"  每张: {config['h_patches']}×{config['w_patches']} = {config['patches_per_image']} patches")

    print("\n### Qwen3-VL MoE (patch_size=16):")
    print("每个patch: 16×16像素")

    # 对于MoE版本
    possible_configs_3 = []

    # 尝试不同的可能性
    for num_images in range(1, 11):
        if total_patches_25 % num_images == 0:
            patches_per_img = total_patches_25 // num_images
            sqrt_p = math.sqrt(patches_per_img)
            if abs(sqrt_p - round(sqrt_p)) < 0.01:  # 近似正方形
                h_p = w_p = round(sqrt_p)
                img_s = h_p * 16
                possible_configs_3.append({
                    'num_images': num_images,
                    'patches_per_image': patches_per_img,
                    'h_patches': h_p,
                    'w_patches': w_p,
                    'image_size': f"{img_s}×{img_s}"
                })

    print("\n可能的配置:")
    for config in possible_configs_3:
        print(f"• {config['num_images']}张图像")
        print(f"  每张约: {config['h_patches']}×{config['w_patches']} ≈ {config['patches_per_image']} patches")

    # 选择最可能的配置进行详细分析
    print("\n" + "=" * 80)
    print("📊 2. 最可能的配置详细分析")
    print("-" * 40)

    # 假设是5张336×336的图像（Qwen2.5-VL常见配置）
    print("\n### 场景A: 5张336×336图像 (Qwen2.5-VL)")
    print("-" * 40)

    num_images = 5
    img_size = 336
    patch_size = 14
    patches_per_dim = img_size // patch_size  # 24
    patches_per_image = patches_per_dim * patches_per_dim  # 576
    total_patches = patches_per_image * num_images  # 2880

    # 注意：5040可能包含了spatial merge
    # Qwen2.5-VL有2×2 spatial merge
    spatial_merge = 2
    actual_patches = patches_per_image * num_images  # 2880
    after_merge = actual_patches // (spatial_merge * spatial_merge)  # 720 per image
    total_after_merge = after_merge * num_images  # 3600

    # 让我们假设是7张336×336图像
    num_images = 7
    patches_per_image = 24 * 24  # 576
    after_merge = patches_per_image // 4  # 144 per image after 2×2 merge
    total_seq = after_merge * num_images * 4  # 回到原始token数，可能有其他处理

    # 更可能是：10张224×224图像
    print("\n实际可能配置：10张224×224图像")
    num_images = 10
    img_size = 224
    patch_size = 14
    patches_per_dim = img_size // patch_size  # 16
    patches_per_image = patches_per_dim * patches_per_dim  # 256

    # 考虑2×2 spatial merge
    tokens_per_image_after_merge = patches_per_image  # 保持256
    # 但实际处理时可能有4个token per merged patch
    tokens_per_image = patches_per_image * 2  # 512 tokens per image
    total_tokens = tokens_per_image * num_images  # 5120 ≈ 5040

    print(f"图像数量: {num_images}")
    print(f"每张图像: {img_size}×{img_size}")
    print(f"Patches: {patches_per_dim}×{patches_per_dim} = {patches_per_image}")
    print(f"处理后每张: ~504 tokens")
    print(f"总序列长度: 5040")

    print("\n### Qwen2.5-VL 批处理分析:")
    print("-" * 40)

    # 窗口注意力层
    window_size = 112  # pixels
    window_patches = window_size // patch_size  # 8
    tokens_per_window = window_patches * window_patches  # 64

    # 每张图像的窗口数
    windows_per_dim = img_size // window_size  # 224/112 = 2
    windows_per_image = windows_per_dim * windows_per_dim  # 4
    total_windows = windows_per_image * num_images  # 40

    print(f"\n窗口注意力层（28层）:")
    print(f"• 窗口大小: {window_size}×{window_size}像素 = {tokens_per_window} tokens")
    print(f"• 每张图像: {windows_per_dim}×{windows_per_dim} = {windows_per_image}个窗口")
    print(f"• 总窗口数: {total_windows}个")
    print(f"• 批次数: {total_windows}")
    print(f"• 每批大小: {tokens_per_window} tokens")

    # 构建cu_window_seqlens
    cu_window_seqlens = [0]
    for i in range(total_windows):
        cu_window_seqlens.append(cu_window_seqlens[-1] + tokens_per_window)

    print(f"\ncu_window_seqlens (前10个和后5个):")
    print(f"  {cu_window_seqlens[:10]} ... {cu_window_seqlens[-5:]}")
    print(f"  长度: {len(cu_window_seqlens)}个边界点")

    # 全注意力层
    print(f"\n全注意力层（4层）:")
    tokens_per_image_approx = 504

    cu_seqlens = [0]
    for i in range(num_images):
        cu_seqlens.append(cu_seqlens[-1] + tokens_per_image_approx)

    print(f"• 批次数: {num_images}")
    print(f"• 每批大小: ~{tokens_per_image_approx} tokens")
    print(f"\ncu_seqlens:")
    print(f"  {cu_seqlens}")

    # Qwen3-VL MoE分析
    print("\n### Qwen3-VL MoE 批处理分析:")
    print("-" * 40)

    # 假设类似配置但patch_size=16
    patch_size_moe = 16

    # 可能是8张280×280的图像
    num_images_moe = 8
    img_size_moe = 280  # 能被16整除
    patches_per_dim_moe = img_size_moe // patch_size_moe  # 17.5，不对

    # 更可能是7张240×240
    num_images_moe = 7
    img_size_moe = 240
    patches_per_dim_moe = img_size_moe // patch_size_moe  # 15
    patches_per_image_moe = patches_per_dim_moe * patches_per_dim_moe  # 225

    # 或者6张288×288
    num_images_moe = 6
    img_size_moe = 288
    patches_per_dim_moe = img_size_moe // patch_size_moe  # 18
    patches_per_image_moe = patches_per_dim_moe * patches_per_dim_moe  # 324
    total_moe = patches_per_image_moe * num_images_moe  # 1944，不够

    # 实际可能：5张 32×32 patches = 5×1024 = 5120 ≈ 5040
    num_images_moe = 5
    patches_per_image_moe = 1008  # 5040/5

    print(f"图像数量: {num_images_moe}")
    print(f"每张图像: ~1008 tokens")
    print(f"总序列长度: 5040")

    print(f"\n所有层（27层）:")
    print(f"• 批次数: {num_images_moe}")
    print(f"• 每批大小: ~1008 tokens")

    cu_seqlens_moe = [0]
    for i in range(num_images_moe):
        cu_seqlens_moe.append(cu_seqlens_moe[-1] + 1008)

    print(f"\ncu_seqlens:")
    print(f"  {cu_seqlens_moe}")

    # 总结对比
    print("\n" + "=" * 80)
    print("📈 3. 批处理对比总结 (5040序列)")
    print("-" * 40)

    print("""
    假设输入：10张224×224图像（Qwen2.5-VL）或 5张图像（Qwen3-VL MoE）

    ┌─────────────────┬──────────────────────┬──────────────────────┐
    │                 │   Qwen2.5-VL         │   Qwen3-VL MoE       │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ 窗口注意力层    │                      │                      │
    │ 批次数          │ 40个                 │ -                    │
    │ 每批大小        │ 64 tokens            │ -                    │
    │ cu_seqlens长度  │ 41个边界点           │ -                    │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ 全注意力层      │                      │                      │
    │ 批次数          │ 10个                 │ 5个                  │
    │ 每批大小        │ ~504 tokens          │ ~1008 tokens         │
    │ cu_seqlens长度  │ 11个边界点           │ 6个边界点            │
    └─────────────────┴──────────────────────┴──────────────────────┘

    注意力矩阵复杂度：
    • Qwen2.5-VL窗口层: 40 × O(64²) = 40 × 4,096 = 163,840
    • Qwen2.5-VL全层:   10 × O(504²) = 10 × 254,016 = 2,540,160
    • Qwen3-VL MoE:     5 × O(1008²) = 5 × 1,016,064 = 5,080,320
    """)

    print("=" * 80)


def visualize_5040_batching():
    """
    可视化5040序列的批处理
    """
    print("\n\n5040序列批处理可视化")
    print("=" * 80)

    print("""
    Qwen2.5-VL (假设10张224×224图像):
    ═══════════════════════════════════════════════════════════════

    窗口注意力层:
    Image 1: [W1][W2][W3][W4] → 4个窗口 × 64 tokens
    Image 2: [W5][W6][W7][W8]
    ...
    Image 10: [W37][W38][W39][W40]

    cu_window_seqlens = [0, 64, 128, 192, 256, ..., 2496, 2560]
                         └─Image 1─┘└─Image 2─┘ ... └─Image 10─┘

    全注意力层:
    [Image1:504][Image2:504][Image3:504]...[Image10:504]

    cu_seqlens = [0, 504, 1008, 1512, 2016, 2520, 3024, 3528, 4032, 4536, 5040]

    ═══════════════════════════════════════════════════════════════

    Qwen3-VL MoE (假设5张图像):
    ═══════════════════════════════════════════════════════════════

    所有层:
    [Image1:1008][Image2:1008][Image3:1008][Image4:1008][Image5:1008]

    cu_seqlens = [0, 1008, 2016, 3024, 4032, 5040]

    ═══════════════════════════════════════════════════════════════
    """)

    print("=" * 80)


if __name__ == "__main__":
    analyze_5040_sequence()
    visualize_5040_batching()