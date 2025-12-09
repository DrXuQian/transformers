"""
MRoPE (Multi-Resolution Rotary Position Embedding) 对比示例
演示 Interleaved vs Standard MRoPE 的区别
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class StandardMRoPE:
    """
    Standard MRoPE (Qwen2.5-VL使用)
    分块式排列：将维度按块分配
    """
    def __init__(self, dim=128, mrope_section=[16, 24, 24]):
        """
        mrope_section: [T_dim, H_dim, W_dim] 各维度分配的维度数
        总和应该等于 dim // 2 (因为cos和sin各占一半)
        """
        self.dim = dim
        self.mrope_section = mrope_section
        assert sum(mrope_section) == dim // 2, "mrope sections must sum to dim/2"

    def apply(self, freqs_t, freqs_h, freqs_w):
        """
        Standard MRoPE: 分块式排列
        [TTTT...HHHH...WWWW]

        布局：
        - 前16维：时间信息
        - 中24维：高度信息
        - 后24维：宽度信息
        """
        # 扩展mrope_section以包含cos和sin
        mrope_section_expanded = [s * 2 for s in self.mrope_section]

        # 按块拼接
        # freqs_t[:, :, :mrope_section_expanded[0]]  # T块
        # freqs_h[:, :, mrope_section_expanded[0]:mrope_section_expanded[0]+mrope_section_expanded[1]]  # H块
        # freqs_w[:, :, mrope_section_expanded[0]+mrope_section_expanded[1]:]  # W块

        result = torch.zeros(freqs_t.shape)

        # 分配T维度 (前16*2=32维)
        t_start = 0
        t_end = mrope_section_expanded[0]
        result[..., t_start:t_end] = freqs_t[..., t_start:t_end]

        # 分配H维度 (中间24*2=48维)
        h_start = t_end
        h_end = h_start + mrope_section_expanded[1]
        result[..., h_start:h_end] = freqs_h[..., h_start:h_end]

        # 分配W维度 (后24*2=48维)
        w_start = h_end
        w_end = w_start + mrope_section_expanded[2]
        result[..., w_start:w_end] = freqs_w[..., w_start:w_end]

        return result

    def visualize_pattern(self):
        """可视化Standard MRoPE的维度分配模式"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))

        # 创建颜色映射
        pattern = []
        labels = []
        colors = []

        # T维度 - 蓝色
        for _ in range(self.mrope_section[0] * 2):
            pattern.append(0)
            labels.append('T')
            colors.append('blue')

        # H维度 - 绿色
        for _ in range(self.mrope_section[1] * 2):
            pattern.append(1)
            labels.append('H')
            colors.append('green')

        # W维度 - 红色
        for _ in range(self.mrope_section[2] * 2):
            pattern.append(2)
            labels.append('W')
            colors.append('red')

        # 绘制
        for i, (p, c) in enumerate(zip(pattern, colors)):
            ax.bar(i, 1, color=c, width=1, edgecolor='black', linewidth=0.5)

        ax.set_title('Standard MRoPE Pattern (Qwen2.5-VL)\n[TTT...HHH...WWW]', fontsize=14)
        ax.set_xlabel('Dimension Index')
        ax.set_ylim(0, 1.2)
        ax.set_xlim(-0.5, len(pattern)-0.5)

        # 添加分组标签
        ax.text(16, 1.1, 'Temporal (32)', ha='center', fontsize=10, color='blue')
        ax.text(56, 1.1, 'Height (48)', ha='center', fontsize=10, color='green')
        ax.text(104, 1.1, 'Width (48)', ha='center', fontsize=10, color='red')

        plt.tight_layout()
        return fig


class InterleavedMRoPE:
    """
    Interleaved MRoPE (Qwen3-VL使用)
    交错式排列：维度交替分配
    """
    def __init__(self, dim=128, mrope_section=[24, 20, 20]):
        """
        mrope_section: [T_dim, H_dim, W_dim] 各维度分配的维度数
        """
        self.dim = dim
        self.mrope_section = mrope_section
        assert sum(mrope_section) == dim // 2, "mrope sections must sum to dim/2"

    def apply_interleaved(self, freqs_t, freqs_h, freqs_w):
        """
        Interleaved MRoPE: 交错式排列
        [THWTHWTHW...TT]

        模式：
        - 每3个维度为一组：[T, H, W]
        - 循环交错直到维度用完
        - 最后剩余的维度给T（如果T的维度更多）
        """
        result = freqs_t.clone()  # 从T开始，后面会覆盖

        mrope_section_expanded = self.mrope_section.copy()
        for i in range(3):
            mrope_section_expanded[i] *= 2  # cos和sin

        # 交错模式实现
        for dim_idx, offset in enumerate([1, 2], start=1):  # H=1, W=2
            if dim_idx == 1:  # H维度
                source_freqs = freqs_h
            else:  # W维度
                source_freqs = freqs_w

            # 计算交错位置
            length = mrope_section_expanded[dim_idx]

            # 每3个位置中取一个，从offset开始
            indices = list(range(offset, min(length * 3, self.dim), 3))

            if len(indices) > 0:
                # 将source_freqs的对应维度复制到result的交错位置
                for i, idx in enumerate(indices[:length]):
                    if idx < result.shape[-1] and i < source_freqs.shape[-1]:
                        result[..., idx] = source_freqs[..., idx]

        return result

    def visualize_pattern(self):
        """可视化Interleaved MRoPE的维度分配模式"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))

        # 创建交错模式
        pattern = []
        labels = []
        colors = []
        color_map = {0: 'blue', 1: 'green', 2: 'red'}
        label_map = {0: 'T', 1: 'H', 2: 'W'}

        # 交错分配
        t_count = self.mrope_section[0] * 2
        h_count = self.mrope_section[1] * 2
        w_count = self.mrope_section[2] * 2

        t_used, h_used, w_used = 0, 0, 0

        for i in range(self.dim):
            # 确定当前位置应该是T、H还是W
            cycle_pos = i % 3

            if cycle_pos == 0 and t_used < t_count:
                pattern.append(0)
                t_used += 1
            elif cycle_pos == 1 and h_used < h_count:
                pattern.append(1)
                h_used += 1
            elif cycle_pos == 2 and w_used < w_count:
                pattern.append(2)
                w_used += 1
            else:
                # 如果某个维度用完了，优先分配给T
                if t_used < t_count:
                    pattern.append(0)
                    t_used += 1
                elif h_used < h_count:
                    pattern.append(1)
                    h_used += 1
                elif w_used < w_count:
                    pattern.append(2)
                    w_used += 1

        # 绘制
        for i, p in enumerate(pattern):
            ax.bar(i, 1, color=color_map[p], width=1, edgecolor='black', linewidth=0.5)

        ax.set_title('Interleaved MRoPE Pattern (Qwen3-VL)\n[THWTHWTHW...TT]', fontsize=14)
        ax.set_xlabel('Dimension Index')
        ax.set_ylim(0, 1.2)
        ax.set_xlim(-0.5, len(pattern)-0.5)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Temporal (48)'),
            Patch(facecolor='green', label='Height (40)'),
            Patch(facecolor='red', label='Width (40)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        return fig


def compare_mrope_encoding():
    """对比两种MRoPE在实际位置编码中的差异"""

    # 创建示例位置
    batch_size = 1
    seq_len = 16
    dim = 128

    # 模拟3D位置 (T, H, W)
    positions_3d = torch.tensor([
        [0, 0, 0],  # 第一个patch
        [0, 0, 1],  # 同一帧，右移
        [0, 1, 0],  # 同一帧，下移
        [0, 1, 1],  # 同一帧，右下
        [1, 0, 0],  # 下一帧，左上
        [1, 0, 1],  # 下一帧，右上
        [1, 1, 0],  # 下一帧，左下
        [1, 1, 1],  # 下一帧，右下
    ])

    # 创建频率（简化示例）
    freqs_t = torch.randn(batch_size, positions_3d.shape[0], dim)
    freqs_h = torch.randn(batch_size, positions_3d.shape[0], dim)
    freqs_w = torch.randn(batch_size, positions_3d.shape[0], dim)

    # Standard MRoPE
    standard = StandardMRoPE(dim=dim, mrope_section=[16, 24, 24])
    standard_result = standard.apply(freqs_t, freqs_h, freqs_w)

    # Interleaved MRoPE
    interleaved = InterleavedMRoPE(dim=dim, mrope_section=[24, 20, 20])
    interleaved_result = interleaved.apply_interleaved(freqs_t, freqs_h, freqs_w)

    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Standard MRoPE热力图
    axes[0, 0].imshow(standard_result[0, :8, :64].numpy(), aspect='auto', cmap='coolwarm')
    axes[0, 0].set_title('Standard MRoPE Encoding (前64维)')
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Position')

    # Interleaved MRoPE热力图
    axes[0, 1].imshow(interleaved_result[0, :8, :64].numpy(), aspect='auto', cmap='coolwarm')
    axes[0, 1].set_title('Interleaved MRoPE Encoding (前64维)')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Position')

    # 维度分配模式
    standard_pattern = StandardMRoPE(dim=128, mrope_section=[16, 24, 24])
    interleaved_pattern = InterleavedMRoPE(dim=128, mrope_section=[24, 20, 20])

    # Standard模式
    ax = axes[1, 0]
    pattern = []
    colors = []
    for _ in range(32): colors.append('blue')  # T
    for _ in range(48): colors.append('green')  # H
    for _ in range(48): colors.append('red')  # W

    for i, c in enumerate(colors[:64]):
        ax.bar(i, 1, color=c, width=1, edgecolor='black', linewidth=0.5)
    ax.set_title('Standard MRoPE Pattern')
    ax.set_ylim(0, 1.2)

    # Interleaved模式
    ax = axes[1, 1]
    colors_int = []
    for i in range(64):
        cycle = i % 3
        if cycle == 0: colors_int.append('blue')
        elif cycle == 1: colors_int.append('green')
        else: colors_int.append('red')

    for i, c in enumerate(colors_int):
        ax.bar(i, 1, color=c, width=1, edgecolor='black', linewidth=0.5)
    ax.set_title('Interleaved MRoPE Pattern')
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    return fig


def main():
    """主函数：生成对比可视化"""

    print("=" * 60)
    print("MRoPE对比分析：Interleaved vs Standard")
    print("=" * 60)

    # 1. Standard MRoPE (Qwen2.5-VL)
    print("\n1. Standard MRoPE (Qwen2.5-VL)")
    print("-" * 40)
    print("配置: [16, 24, 24] -> 分块式排列")
    print("模式: [TTT...HHH...WWW]")
    print("特点:")
    print("  - 维度按块连续分配")
    print("  - T维度集中在前32维")
    print("  - H维度集中在中间48维")
    print("  - W维度集中在后48维")
    print("  - 空间局部性好，缓存友好")

    # 2. Interleaved MRoPE (Qwen3-VL)
    print("\n2. Interleaved MRoPE (Qwen3-VL)")
    print("-" * 40)
    print("配置: [24, 20, 20] -> 交错式排列")
    print("模式: [THWTHWTHW...TT]")
    print("特点:")
    print("  - 维度交替分配")
    print("  - T、H、W信息均匀分布")
    print("  - 每个维度组包含3D完整信息")
    print("  - 更好的维度间交互")
    print("  - 适合需要综合3D理解的任务")

    # 3. 关键差异
    print("\n3. 关键差异")
    print("-" * 40)
    print("• 信息分布:")
    print("  - Standard: 分块集中，维度独立")
    print("  - Interleaved: 均匀分布，维度交织")
    print("\n• 计算效率:")
    print("  - Standard: 缓存友好，向量化计算高效")
    print("  - Interleaved: 需要更多索引操作")
    print("\n• 表达能力:")
    print("  - Standard: 适合维度独立性强的任务")
    print("  - Interleaved: 适合需要3D整体理解的任务")
    print("\n• 梯度流:")
    print("  - Standard: 维度间梯度流动受限")
    print("  - Interleaved: 维度间梯度流动更自然")

    # 生成可视化
    print("\n生成可视化对比图...")

    # 创建Standard MRoPE可视化
    standard = StandardMRoPE(dim=128, mrope_section=[16, 24, 24])
    fig1 = standard.visualize_pattern()
    plt.savefig('/home/qianxu/transformers/standard_mrope_pattern.png', dpi=150, bbox_inches='tight')

    # 创建Interleaved MRoPE可视化
    interleaved = InterleavedMRoPE(dim=128, mrope_section=[24, 20, 20])
    fig2 = interleaved.visualize_pattern()
    plt.savefig('/home/qianxu/transformers/interleaved_mrope_pattern.png', dpi=150, bbox_inches='tight')

    # 创建对比图
    fig3 = compare_mrope_encoding()
    plt.savefig('/home/qianxu/transformers/mrope_comparison.png', dpi=150, bbox_inches='tight')

    print("\n可视化图片已保存:")
    print("  - standard_mrope_pattern.png")
    print("  - interleaved_mrope_pattern.png")
    print("  - mrope_comparison.png")

    print("\n" + "=" * 60)
    print("分析完成！")


if __name__ == "__main__":
    main()