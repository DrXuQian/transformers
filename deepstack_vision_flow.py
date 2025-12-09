"""
DeepStack Vision特征提取流程详解
展示Vision Transformer如何输出多层特征
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Qwen3VLVisionEncoder(nn.Module):
    """
    Qwen3-VL的Vision Encoder
    展示如何从不同层提取特征
    """
    def __init__(self, config):
        super().__init__()

        # Vision Transformer有24层
        self.depth = 24
        self.hidden_size = 1024

        # Patch Embedding
        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=1024,
            kernel_size=(2, 16, 16),  # (temporal, height, width)
            stride=(2, 16, 16)
        )

        # 24个Vision Transformer Blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(hidden_size=1024)
            for _ in range(24)
        ])

        # DeepStack配置：在哪些层输出特征
        self.deepstack_visual_indexes = [5, 11, 17]  # 第5、11、17层

        # 为每个DeepStack层配置独立的Patch Merger
        # 将1024维的vision特征转换为2560维的LLM特征
        self.deepstack_mergers = nn.ModuleDict({
            '5': PatchMerger(1024, 2560),   # Layer 5 merger
            '11': PatchMerger(1024, 2560),  # Layer 11 merger
            '17': PatchMerger(1024, 2560),  # Layer 17 merger
        })

        # 最终输出的Patch Merger
        self.final_merger = PatchMerger(1024, 2560)

    def forward(self, pixel_values):
        """
        输入：图像像素值
        输出：
            1. final_features: 最终的视觉特征（Layer 24的输出）
            2. deepstack_features: 中间层的视觉特征字典
        """

        # Step 1: Patch Embedding
        # [B, C, T, H, W] -> [B*num_patches, hidden_size]
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        print(f"Initial shape after patch embedding: {hidden_states.shape}")
        # 例如: [batch_size, num_patches, 1024]

        # 用于存储DeepStack特征
        deepstack_features = {}

        # Step 2: 逐层处理，并在特定层提取特征
        for layer_idx, block in enumerate(self.blocks):
            # 通过当前Vision Transformer Block
            hidden_states = block(hidden_states)

            print(f"Layer {layer_idx}: shape = {hidden_states.shape}")

            # 检查是否是DeepStack层
            if layer_idx in self.deepstack_visual_indexes:
                # 提取当前层的输出
                print(f"  → DeepStack extraction at layer {layer_idx}")

                # 使用对应的merger转换维度
                merger = self.deepstack_mergers[str(layer_idx)]
                deepstack_feature = merger(hidden_states)

                # 保存到字典中
                deepstack_features[f'layer_{layer_idx}'] = deepstack_feature

                print(f"  → Extracted feature shape: {deepstack_feature.shape}")
                # 从 [batch, patches, 1024] -> [batch, patches, 2560]

        # Step 3: 最终输出处理
        final_features = self.final_merger(hidden_states)
        print(f"Final features shape: {final_features.shape}")

        return final_features, deepstack_features


class VisionTransformerBlock(nn.Module):
    """
    单个Vision Transformer Block
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        # Self-Attention with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class PatchMerger(nn.Module):
    """
    Patch Merger: 转换vision特征到LLM维度
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(self.norm(x))


def visualize_deepstack_flow():
    """
    可视化DeepStack的特征提取流程
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # 设置坐标轴
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 26)
    ax.axis('off')

    # 标题
    ax.text(7, 25, 'DeepStack Vision Feature Extraction Flow',
            fontsize=16, fontweight='bold', ha='center')

    # 输入图像
    img_rect = patches.Rectangle((1, 22), 2, 2,
                                 linewidth=2, edgecolor='green',
                                 facecolor='lightgreen')
    ax.add_patch(img_rect)
    ax.text(2, 21.5, 'Input Image', ha='center', fontsize=10)

    # Patch Embedding
    pe_rect = patches.Rectangle((0.5, 19), 3, 1.5,
                                linewidth=2, edgecolor='blue',
                                facecolor='lightblue')
    ax.add_patch(pe_rect)
    ax.text(2, 19.75, 'Patch Embed', ha='center', fontsize=10)
    ax.text(2, 19.25, '16×16 patches', ha='center', fontsize=8)

    # Vision Transformer Blocks
    layer_y_positions = []
    for i in range(24):
        y_pos = 18 - i * 0.7
        layer_y_positions.append(y_pos)

        # 判断是否是DeepStack层
        if i in [5, 11, 17]:
            color = 'red'
            facecolor = 'lightcoral'
            linewidth = 3
        else:
            color = 'gray'
            facecolor = 'lightgray'
            linewidth = 1

        # Vision Block矩形
        block_rect = patches.Rectangle((0.5, y_pos - 0.3), 3, 0.6,
                                      linewidth=linewidth,
                                      edgecolor=color,
                                      facecolor=facecolor)
        ax.add_patch(block_rect)

        # 层编号
        ax.text(0.2, y_pos, f'L{i}', ha='center', fontsize=8)
        ax.text(2, y_pos, f'ViT Block {i}', ha='center', fontsize=9)

        # 如果是DeepStack层，画出提取箭头
        if i in [5, 11, 17]:
            # 箭头指向右侧
            ax.arrow(3.5, y_pos, 2, 0,
                    head_width=0.2, head_length=0.1,
                    fc='red', ec='red')

            # Patch Merger
            merger_rect = patches.Rectangle((5.8, y_pos - 0.3), 2.5, 0.6,
                                          linewidth=2,
                                          edgecolor='purple',
                                          facecolor='plum')
            ax.add_patch(merger_rect)
            ax.text(7.05, y_pos, f'Merger {i}', ha='center', fontsize=9)

            # 箭头指向LLM
            ax.arrow(8.3, y_pos, 2, 0,
                    head_width=0.2, head_length=0.1,
                    fc='purple', ec='purple')

            # LLM层标注
            llm_rect = patches.Rectangle((10.5, y_pos - 0.3), 2.5, 0.6,
                                        linewidth=2,
                                        edgecolor='orange',
                                        facecolor='lightyellow')
            ax.add_patch(llm_rect)

            if i == 5:
                ax.text(11.75, y_pos, 'LLM L0-3', ha='center', fontsize=9)
                ax.text(11.75, y_pos - 0.15, '(低级特征)', ha='center', fontsize=7)
            elif i == 11:
                ax.text(11.75, y_pos, 'LLM L4-7', ha='center', fontsize=9)
                ax.text(11.75, y_pos - 0.15, '(中级特征)', ha='center', fontsize=7)
            elif i == 17:
                ax.text(11.75, y_pos, 'LLM L8-11', ha='center', fontsize=9)
                ax.text(11.75, y_pos - 0.15, '(高级特征)', ha='center', fontsize=7)

    # 最终输出
    final_y = layer_y_positions[-1] - 1.5
    final_rect = patches.Rectangle((0.5, final_y - 0.3), 3, 0.6,
                                  linewidth=2, edgecolor='green',
                                  facecolor='lightgreen')
    ax.add_patch(final_rect)
    ax.text(2, final_y, 'Final Merger', ha='center', fontsize=10)

    # 箭头到LLM输入
    ax.arrow(3.5, final_y, 2, 0,
            head_width=0.2, head_length=0.1,
            fc='green', ec='green')

    input_rect = patches.Rectangle((5.8, final_y - 0.3), 2.5, 0.6,
                                  linewidth=2, edgecolor='orange',
                                  facecolor='lightyellow')
    ax.add_patch(input_rect)
    ax.text(7.05, final_y, 'LLM Input', ha='center', fontsize=10)

    # 添加图例
    legend_elements = [
        patches.Patch(color='lightgray', label='Regular ViT Block'),
        patches.Patch(color='lightcoral', label='DeepStack Block'),
        patches.Patch(color='plum', label='Patch Merger'),
        patches.Patch(color='lightyellow', label='To LLM')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 添加说明文字
    ax.text(7, 0.5, 'All features come from the SAME Vision Transformer,',
            ha='center', fontsize=11, style='italic')
    ax.text(7, 0, 'just extracted at different depths (layers 5, 11, 17)',
            ha='center', fontsize=11, style='italic', color='red')

    plt.title('Qwen3-VL DeepStack: Multi-layer Vision Feature Extraction',
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/home/qianxu/transformers/deepstack_flow.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_feature_extraction():
    """
    演示实际的特征提取过程
    """
    print("=" * 70)
    print("DeepStack Vision特征提取演示")
    print("=" * 70)

    # 创建模拟配置
    class Config:
        pass

    config = Config()

    # 创建Vision Encoder
    vision_encoder = Qwen3VLVisionEncoder(config)

    # 模拟输入
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 2, 224, 224)  # [B, C, T, H, W]

    print(f"\n输入图像shape: {pixel_values.shape}")
    print("  - Batch size: 2")
    print("  - Channels: 3 (RGB)")
    print("  - Temporal: 2 frames")
    print("  - Spatial: 224×224")

    print("\n" + "-" * 70)
    print("处理过程：")
    print("-" * 70)

    # 前向传播
    with torch.no_grad():
        final_features, deepstack_features = vision_encoder(pixel_values)

    print("\n" + "-" * 70)
    print("输出结果：")
    print("-" * 70)

    print(f"\n1. 最终特征 (Layer 24输出):")
    print(f"   Shape: {final_features.shape}")
    print(f"   用途: 作为LLM的主要视觉输入")

    print(f"\n2. DeepStack中间特征:")
    for key, features in deepstack_features.items():
        print(f"   {key}:")
        print(f"     Shape: {features.shape}")
        print(f"     维度: 1024 -> 2560 (转换到LLM维度)")

    print("\n" + "=" * 70)


def explain_implementation():
    """
    解释实际实现细节
    """
    print("\n" + "=" * 70)
    print("💡 关键实现细节")
    print("=" * 70)

    details = {
        "1. 单一Vision Transformer": [
            "只有一个Vision Encoder，不是多个",
            "输入图像只处理一次",
            "通过24个ViT Block顺序处理"
        ],

        "2. 中间层特征提取": [
            "在forward过程中，保存特定层的输出",
            "Layer 5: hidden_states在第5层后的状态",
            "Layer 11: hidden_states在第11层后的状态",
            "Layer 17: hidden_states在第17层后的状态",
            "Layer 24: 最终的hidden_states"
        ],

        "3. Patch Merger的作用": [
            "Vision特征维度: 1024",
            "LLM需要维度: 2560",
            "每个DeepStack层有独立的Merger",
            "Merger = LayerNorm + Linear投影"
        ],

        "4. 为什么选择5、11、17层": [
            "Layer 5 (浅层): 捕获低级视觉特征",
            "Layer 11 (中层): 捕获中级语义特征",
            "Layer 17 (深层): 捕获高级抽象特征",
            "均匀分布在24层中，覆盖不同抽象层次"
        ],

        "5. 内存和计算开销": [
            "需要存储4份视觉特征(final + 3个中间层)",
            "每个Merger增加约2.6M参数(1024×2560)",
            "总共增加约10M参数用于DeepStack",
            "前向传播时需要额外的特征复制和转换"
        ]
    }

    for title, items in details.items():
        print(f"\n{title}:")
        for item in items:
            print(f"  • {item}")

    print("\n" + "=" * 70)


def show_actual_code():
    """
    展示实际代码片段
    """
    print("\n" + "=" * 70)
    print("📝 Qwen3-VL 实际代码")
    print("=" * 70)

    actual_code = '''
# modeling_qwen3_vl.py 中的实际实现

class Qwen3VLVisionTransformerPretrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            Qwen3VLVisionBlock(config) for _ in range(config.depth)
        ])

        # DeepStack mergers
        self.deepstack_mergers = nn.ModuleDict()
        for idx in config.deepstack_visual_indexes:
            self.deepstack_mergers[str(idx)] = Qwen3VLVisionPatchMerger(
                config, use_postshuffle_norm=True
            )

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb):
        # 收集DeepStack embeddings
        deepstack_embeds = []

        for idx, blk in enumerate(self.blocks):
            # 通过Vision Block
            hidden_states = blk(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb
            )

            # 如果是DeepStack层，提取特征
            if idx in self.config.deepstack_visual_indexes:
                merger = self.deepstack_mergers[str(idx)]
                deepstack_embeds.append(merger(hidden_states))

        # 最终输出
        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_embeds
    '''

    print(actual_code)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # 1. 演示特征提取过程
    demonstrate_feature_extraction()

    # 2. 可视化流程图
    print("\n生成DeepStack流程图...")
    visualize_deepstack_flow()
    print("流程图已保存到: deepstack_flow.png")

    # 3. 解释实现细节
    explain_implementation()

    # 4. 展示实际代码
    show_actual_code()

    print("\n" + "=" * 70)
    print("✅ 总结：DeepStack的所有视觉特征都来自同一个Vision Transformer，")
    print("   只是在不同的处理深度(层)被提取出来并注入到LLM中。")
    print("=" * 70)