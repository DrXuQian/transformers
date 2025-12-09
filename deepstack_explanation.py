"""
DeepStack Injection 机制详解
Qwen3-VL 独有的多层次视觉特征融合
"""

import torch
import torch.nn as nn

class DeepStackVisionEncoder(nn.Module):
    """
    Vision Encoder with DeepStack outputs
    在特定层输出中间特征供LLM使用
    """
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            VisionBlock(config) for _ in range(24)
        ])

        # DeepStack: 在这些层输出中间特征
        self.deepstack_indexes = [5, 11, 17]  # 配置中的 deepstack_visual_indexes

        # 每个DeepStack层有独立的PatchMerger
        self.deepstack_mergers = nn.ModuleDict({
            str(idx): PatchMerger(
                in_dim=1024,      # Vision hidden size
                out_dim=2560      # LLM hidden size
            ) for idx in self.deepstack_indexes
        })

    def forward(self, pixel_values):
        """
        返回：
        1. 最终的视觉特征 (用于输入层)
        2. DeepStack中间特征 (用于中间层注入)
        """
        hidden_states = self.patch_embed(pixel_values)

        # 收集DeepStack特征
        deepstack_features = {}

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)

            # 如果是DeepStack层，保存中间特征
            if idx in self.deepstack_indexes:
                # 使用对应的merger转换到LLM维度
                merger = self.deepstack_mergers[str(idx)]
                deepstack_features[idx] = merger(hidden_states)

        # 最终特征
        final_features = self.final_merger(hidden_states)

        return final_features, deepstack_features


class Qwen3VLModel(nn.Module):
    """
    Qwen3-VL 主模型
    展示DeepStack如何注入到LLM中
    """
    def __init__(self, config):
        super().__init__()
        self.vision_encoder = DeepStackVisionEncoder(config)
        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(36)
        ])

        # DeepStack映射：Vision层 -> LLM层
        self.deepstack_mapping = {
            5: [0, 1, 2, 3],      # Vision层5的特征注入到LLM层0-3
            11: [4, 5, 6, 7],     # Vision层11的特征注入到LLM层4-7
            17: [8, 9, 10, 11]    # Vision层17的特征注入到LLM层8-11
        }

    def forward(self, input_ids, pixel_values):
        # 1. 获取视觉特征（包括DeepStack中间特征）
        vision_features, deepstack_features = self.vision_encoder(pixel_values)

        # 2. 文本嵌入
        text_embeds = self.text_embedding(input_ids)

        # 3. 合并输入层的视觉和文本特征
        hidden_states = merge_vision_text(vision_features, text_embeds)

        # 4. 通过Decoder层，在特定层注入DeepStack特征
        for layer_idx, layer in enumerate(self.layers):
            # 检查是否需要注入DeepStack特征
            deepstack_feature = None
            for vision_idx, llm_layers in self.deepstack_mapping.items():
                if layer_idx in llm_layers:
                    deepstack_feature = deepstack_features[vision_idx]
                    break

            # 前向传播（可能包含DeepStack注入）
            hidden_states = layer(
                hidden_states,
                deepstack_feature=deepstack_feature
            )

        return hidden_states


class DecoderLayer(nn.Module):
    """
    Decoder层实现，支持DeepStack注入
    """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = FeedForward(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

        # DeepStack融合层（可选）
        self.deepstack_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, hidden_states, deepstack_feature=None):
        # 1. Self-Attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states

        # 2. DeepStack注入（如果有）
        if deepstack_feature is not None:
            hidden_states = self.inject_deepstack(hidden_states, deepstack_feature)

        # 3. FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def inject_deepstack(self, hidden_states, visual_features):
        """
        DeepStack注入的几种可能方式
        """
        # 方式1：简单相加
        # return hidden_states + visual_features

        # 方式2：门控融合（更复杂）
        combined = torch.cat([hidden_states, visual_features], dim=-1)
        gate = torch.sigmoid(self.deepstack_gate(combined))
        return hidden_states * (1 - gate) + visual_features * gate

        # 方式3：注意力融合
        # return cross_attention(hidden_states, visual_features)


# ============ 实际代码位置 ============

def show_actual_implementation():
    """
    展示Qwen3-VL实际代码中的DeepStack实现
    """

    print("=" * 60)
    print("DeepStack 在 Qwen3-VL 中的实际实现")
    print("=" * 60)

    # 1. 配置中的定义
    print("\n1. 配置文件 (config.json):")
    print("-" * 40)
    config_snippet = """
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],  // Vision层索引
        ...
    }
    """
    print(config_snippet)

    # 2. Vision Encoder输出
    print("\n2. Vision Encoder 输出多层特征:")
    print("-" * 40)
    vision_code = """
    # modeling_qwen3_vl.py:622-631
    class Qwen3VLVisionTransformerPretrainedModel:
        def forward(self, hidden_states):
            deepstack_embeds = []

            for idx, blk in enumerate(self.blocks):
                hidden_states = blk(hidden_states)

                # 收集DeepStack层的输出
                if idx in self.config.deepstack_visual_indexes:
                    merger = self.deepstack_mergers[idx]
                    deepstack_embeds.append(merger(hidden_states))

            return hidden_states, deepstack_embeds
    """
    print(vision_code)

    # 3. LLM中的注入
    print("\n3. 在LLM Decoder层中注入:")
    print("-" * 40)
    llm_code = """
    # modeling_qwen3_vl.py:893-898
    for layer_idx, decoder_layer in enumerate(self.layers):
        # 标准Decoder计算
        hidden_states = decoder_layer(hidden_states)

        # DeepStack注入（早期层）
        if deepstack_visual_embeds and layer_idx < len(deepstack_visual_embeds):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx]
            )
    """
    print(llm_code)

    print("\n" + "=" * 60)


# ============ DeepStack的优势 ============

def explain_advantages():
    """
    解释DeepStack的优势
    """

    print("\n🎯 DeepStack 的优势")
    print("=" * 60)

    advantages = {
        "1. 多层次理解": {
            "描述": "不同层次的视觉特征包含不同级别的信息",
            "细节": [
                "Layer 5: 低级视觉特征（边缘、纹理）",
                "Layer 11: 中级特征（物体部件）",
                "Layer 17: 高级特征（物体、场景）"
            ]
        },

        "2. 渐进式融合": {
            "描述": "视觉信息在LLM的不同深度逐步融合",
            "细节": [
                "早期层(0-3): 接收低级视觉特征",
                "中期层(4-7): 接收中级视觉特征",
                "后期层(8-11): 接收高级视觉特征",
                "深层(12-36): 基于已融合的特征继续推理"
            ]
        },

        "3. 信息保留": {
            "描述": "避免视觉信息在深层网络中丢失",
            "细节": [
                "传统方法：视觉特征只在输入层，可能在深层被遗忘",
                "DeepStack：在多个层次强化视觉信息",
                "类似于ResNet的跳跃连接，但是跨模态的"
            ]
        },

        "4. 细粒度控制": {
            "描述": "可以精确控制不同类型的视觉信息如何影响文本生成",
            "细节": [
                "低级特征影响：细节描述、颜色、纹理",
                "中级特征影响：物体识别、空间关系",
                "高级特征影响：场景理解、语义推理"
            ]
        }
    }

    for key, value in advantages.items():
        print(f"\n{key}: {value['描述']}")
        for detail in value['细节']:
            print(f"  • {detail}")

    print("\n" + "=" * 60)


# ============ 对比分析 ============

def compare_with_qwen25():
    """
    对比Qwen3-VL (DeepStack) vs Qwen2.5-VL (无DeepStack)
    """

    print("\n📊 DeepStack vs 传统方法对比")
    print("=" * 60)

    comparison = """
    Qwen3-VL (with DeepStack):
    ┌─────────────┐      ┌─────────────┐
    │   Vision    │ ───> │  Multiple   │
    │   Encoder   │      │  Outputs    │
    └─────────────┘      └──┬──┬──┬───┘
                            │  │  │
                  Layer 5 ──┘  │  └── Layer 17
                       Layer 11 ┘
                            ↓  ↓  ↓
    ┌─────────────────────────────────────┐
    │  LLM Decoder                        │
    │  Layer 0-3  ← Layer 5 features     │
    │  Layer 4-7  ← Layer 11 features    │
    │  Layer 8-11 ← Layer 17 features    │
    │  Layer 12-36 (standard processing) │
    └─────────────────────────────────────┘

    Qwen2.5-VL (without DeepStack):
    ┌─────────────┐      ┌─────────────┐
    │   Vision    │ ───> │   Single    │
    │   Encoder   │      │   Output    │
    └─────────────┘      └─────┬───────┘
                                │
                                ↓
    ┌─────────────────────────────────────┐
    │  LLM Decoder                        │
    │  Layer 0-36 (all layers identical) │
    └─────────────────────────────────────┘
    """

    print(comparison)

    print("\n关键差异:")
    print("-" * 40)
    print("• Qwen3-VL: 视觉信息在多个深度注入")
    print("• Qwen2.5-VL: 视觉信息仅在输入层提供")
    print("• Qwen3-VL: 需要额外的merger和存储")
    print("• Qwen2.5-VL: 架构更简单，计算更高效")


if __name__ == "__main__":
    # 展示实际实现
    show_actual_implementation()

    # 解释优势
    explain_advantages()

    # 对比分析
    compare_with_qwen25()

    print("\n" + "=" * 60)
    print("总结：DeepStack是Qwen3-VL的核心创新")
    print("通过多层次视觉注入实现更深度的多模态融合")
    print("=" * 60)