"""
DeepStack特征融合机制详解
展示LLM如何与Vision特征在不同层融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================== 融合方式实现 ==================

class Qwen3VLModel(nn.Module):
    """
    Qwen3-VL 主模型，展示特征融合的详细实现
    """
    def __init__(self, config):
        super().__init__()
        self.vision_encoder = VisionEncoder(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 36层LLM Decoder
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(36)
        ])

        # DeepStack特征映射关系
        # Vision层 -> LLM层的映射
        self.deepstack_mapping = {
            5: list(range(0, 4)),    # Vision Layer 5 -> LLM Layer 0-3
            11: list(range(4, 8)),   # Vision Layer 11 -> LLM Layer 4-7
            17: list(range(8, 12))   # Vision Layer 17 -> LLM Layer 8-11
        }

    def forward(self, input_ids, pixel_values, attention_mask=None):
        """
        完整的前向传播过程，展示融合细节
        """

        # ========== Step 1: 获取Vision特征 ==========
        vision_outputs = self.vision_encoder(pixel_values)
        final_vision_features = vision_outputs['final']  # [seq_v, 2560]
        deepstack_features = vision_outputs['deepstack']  # {5: tensor, 11: tensor, 17: tensor}

        # ========== Step 2: 准备文本输入 ==========
        text_embeds = self.embed_tokens(input_ids)  # [seq_t, 2560]

        # ========== Step 3: 输入层融合（拼接方式）==========
        # 找到特殊的<image>标记位置
        image_token_mask = (input_ids == IMAGE_TOKEN_ID)

        # 方式1：直接替换
        # 将<image>位置的embedding替换为vision features
        if image_token_mask.any():
            # 获取<image>的位置
            image_positions = torch.where(image_token_mask)[0]

            # 替换：将vision features插入到<image>位置
            inputs_embeds = text_embeds.clone()
            inputs_embeds[image_positions[0]:image_positions[0]+len(final_vision_features)] = final_vision_features
        else:
            # 方式2：序列拼接
            # [CLS] [Text Tokens] [Vision Features] [Text Tokens]
            inputs_embeds = torch.cat([text_embeds[:prefix_len],
                                      final_vision_features,
                                      text_embeds[prefix_len:]], dim=0)

        hidden_states = inputs_embeds  # [total_seq_len, 2560]

        # ========== Step 4: 通过LLM层，逐层融合DeepStack特征 ==========
        for layer_idx, decoder_layer in enumerate(self.layers):

            # 标准Transformer计算
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask
            )

            # 检查是否需要注入DeepStack特征
            deepstack_feature = self.get_deepstack_feature_for_layer(
                layer_idx,
                deepstack_features,
                hidden_states.shape
            )

            if deepstack_feature is not None:
                # 🔥 核心：DeepStack特征融合
                hidden_states = self.fuse_deepstack_features(
                    hidden_states,
                    deepstack_feature,
                    layer_idx
                )

        return hidden_states

    def get_deepstack_feature_for_layer(self, layer_idx, deepstack_features, target_shape):
        """
        获取当前层应该注入的DeepStack特征
        """
        for vision_layer, llm_layers in self.deepstack_mapping.items():
            if layer_idx in llm_layers:
                feature = deepstack_features[vision_layer]
                # 可能需要调整shape或位置
                return self.prepare_deepstack_feature(feature, target_shape)
        return None

    def fuse_deepstack_features(self, hidden_states, visual_features, layer_idx):
        """
        🔥 核心融合函数：展示不同的融合策略
        """
        # 获取视觉token的位置mask
        vision_mask = self.get_vision_positions(hidden_states)

        # ========== 融合策略1：直接相加（最简单）==========
        if self.fusion_method == 'add':
            # 只在视觉相关的位置加入visual features
            hidden_states[vision_mask] = hidden_states[vision_mask] + visual_features
            return hidden_states

        # ========== 融合策略2：门控融合（Gated Fusion）==========
        elif self.fusion_method == 'gate':
            return self.gated_fusion(hidden_states, visual_features, vision_mask)

        # ========== 融合策略3：交叉注意力（Cross-Attention）==========
        elif self.fusion_method == 'cross_attention':
            return self.cross_attention_fusion(hidden_states, visual_features)

        # ========== 融合策略4：自适应融合 ==========
        elif self.fusion_method == 'adaptive':
            return self.adaptive_fusion(hidden_states, visual_features, layer_idx)


class GatedFusion(nn.Module):
    """
    门控融合机制：学习如何混合文本和视觉特征
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 门控网络：决定保留多少原始特征vs视觉特征
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, text_hidden, vision_hidden):
        """
        text_hidden: [seq_len, hidden_size] - LLM的隐藏状态
        vision_hidden: [vision_len, hidden_size] - Vision特征
        """
        # 拼接文本和视觉特征
        combined = torch.cat([text_hidden, vision_hidden], dim=-1)

        # 计算门控值（0-1之间）
        gate = torch.sigmoid(self.gate_proj(combined))

        # 加权融合
        output = gate * text_hidden + (1 - gate) * vision_hidden

        return output


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合：文本作为Query，视觉作为Key/Value
    """
    def __init__(self, hidden_size, num_heads=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, text_hidden, vision_hidden):
        """
        使用文本query视觉信息
        """
        batch_size, text_len, _ = text_hidden.shape
        vision_len = vision_hidden.shape[1]

        # 文本作为Query
        Q = self.q_proj(text_hidden).view(batch_size, text_len, self.num_heads, self.head_dim)

        # 视觉作为Key和Value
        K = self.k_proj(vision_hidden).view(batch_size, vision_len, self.num_heads, self.head_dim)
        V = self.v_proj(vision_hidden).view(batch_size, vision_len, self.num_heads, self.head_dim)

        # 交叉注意力计算
        Q = Q.transpose(1, 2)  # [batch, heads, text_len, head_dim]
        K = K.transpose(1, 2)  # [batch, heads, vision_len, head_dim]
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 获取视觉信息
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, text_len, -1)

        # 输出投影并残差连接
        output = self.o_proj(attn_output) + text_hidden

        return output


class AdaptiveFusion(nn.Module):
    """
    自适应融合：根据层深度调整融合策略
    """
    def __init__(self, hidden_size, num_layers=36):
        super().__init__()
        self.hidden_size = hidden_size

        # 每层有不同的融合权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        # 可学习的投影矩阵
        self.vision_proj = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(12)  # 前12层
        ])

    def forward(self, text_hidden, vision_hidden, layer_idx):
        """
        根据层深度自适应调整融合方式
        """
        # 获取当前层的融合权重
        weight = torch.sigmoid(self.layer_weights[layer_idx])

        # 早期层（0-3）：更多保留视觉细节
        if layer_idx < 4:
            # 视觉特征投影
            vision_projected = self.vision_proj[layer_idx](vision_hidden)
            # 高权重的视觉特征
            output = text_hidden + weight * 1.5 * vision_projected

        # 中期层（4-7）：平衡融合
        elif layer_idx < 8:
            vision_projected = self.vision_proj[layer_idx](vision_hidden)
            # 平衡的融合
            output = text_hidden + weight * vision_projected

        # 后期层（8-11）：轻量融合
        else:
            vision_projected = self.vision_proj[layer_idx](vision_hidden)
            # 较低权重的视觉特征
            output = text_hidden + weight * 0.5 * vision_projected

        return output


# ================== 实际Qwen3-VL的融合实现 ==================

def show_actual_implementation():
    """
    展示Qwen3-VL的实际融合代码
    """
    print("=" * 70)
    print("Qwen3-VL 实际的特征融合实现")
    print("=" * 70)

    actual_code = '''
# modeling_qwen3_vl.py 中的实际实现（简化版）

class Qwen3VLForConditionalGeneration(nn.Module):
    def forward(self, input_ids, pixel_values):

        # 1. 获取vision特征
        image_embeds, deepstack_embeds = self.visual(pixel_values)

        # 2. 获取文本embedding
        inputs_embeds = self.embed_tokens(input_ids)

        # 3. 输入层融合：替换<image>位置
        image_mask = (input_ids == self.config.image_token_id)
        inputs_embeds[image_mask] = image_embeds

        # 4. 通过decoder层
        hidden_states = inputs_embeds

        for layer_idx, decoder_layer in enumerate(self.layers):
            # 标准decoder处理
            hidden_states = decoder_layer(hidden_states)

            # DeepStack融合（仅在早期层）
            if layer_idx < len(deepstack_embeds):
                # 获取视觉位置mask
                vision_mask = self._get_vision_positions(hidden_states)

                # 🔥 核心融合：直接相加
                hidden_states[vision_mask] = (
                    hidden_states[vision_mask] +
                    deepstack_embeds[layer_idx]
                )

        return hidden_states

    def _get_vision_positions(self, hidden_states):
        """
        获取序列中属于视觉的位置
        """
        # 基于position_ids或attention_mask确定
        # 哪些位置是视觉token
        return vision_positions
    '''

    print(actual_code)
    print("\n" + "=" * 70)


def visualize_fusion_process():
    """
    可视化融合过程
    """
    print("\n融合过程可视化")
    print("=" * 70)

    fusion_diagram = '''
    输入序列: [Text] [Image] [Text]
                ↓      ↓       ↓

    Layer 0-3 (接收Vision Layer 5特征):
    ------------------------------------------------
    Input:    [T T T T I I I I I T T T]  ← 输入embedding
                      ↓ ↓ ↓ ↓ ↓
    DeepStack:        [V V V V V]        ← Layer 5特征
    Fusion:   [T T T T I+V I+V I+V T T]  ← 融合后

    Layer 4-7 (接收Vision Layer 11特征):
    ------------------------------------------------
    Input:    [H H H H H H H H H H H H]  ← 上层输出
                      ↓ ↓ ↓ ↓ ↓
    DeepStack:        [V V V V V]        ← Layer 11特征
    Fusion:   [H H H H H+V H+V H+V H H]  ← 融合后

    Layer 8-11 (接收Vision Layer 17特征):
    ------------------------------------------------
    Input:    [H H H H H H H H H H H H]  ← 上层输出
                      ↓ ↓ ↓ ↓ ↓
    DeepStack:        [V V V V V]        ← Layer 17特征
    Fusion:   [H H H H H+V H+V H+V H H]  ← 融合后

    Layer 12-36 (无DeepStack):
    ------------------------------------------------
    标准Transformer处理，无额外视觉注入

    符号说明:
    T = Text embedding
    I = Image embedding (from final vision encoder)
    V = Vision features (from DeepStack)
    H = Hidden states
    + = 特征融合(相加/门控/注意力)
    '''

    print(fusion_diagram)
    print("=" * 70)


def explain_fusion_benefits():
    """
    解释融合机制的优势
    """
    print("\n🎯 DeepStack融合机制的优势")
    print("=" * 70)

    benefits = {
        "1. 多层次理解": [
            "早期层(0-3): 融合低级视觉特征，关注细节",
            "中期层(4-7): 融合中级特征，理解物体",
            "后期层(8-11): 融合高级特征，把握语义",
            "深层(12-36): 基于已融合特征做推理"
        ],

        "2. 信息保持": [
            "避免视觉信息在深层消失",
            "类似ResNet的思想，但跨模态",
            "每次注入都强化视觉信号"
        ],

        "3. 灵活融合": [
            "不同层可以有不同的融合策略",
            "可以学习最优的融合权重",
            "视觉和文本信息互补"
        ],

        "4. 位置敏感": [
            "只在视觉相关位置融合",
            "保持文本token不受干扰",
            "精确的空间对齐"
        ]
    }

    for title, items in benefits.items():
        print(f"\n{title}:")
        for item in items:
            print(f"  • {item}")

    print("\n" + "=" * 70)


def compare_fusion_methods():
    """
    对比不同的融合方法
    """
    print("\n📊 融合方法对比")
    print("=" * 70)

    comparison = """
    | 融合方法 | 实现 | 优点 | 缺点 |
    |---------|------|------|------|
    | 直接相加 | h = h + v | 简单高效 | 可能信息冲突 |
    | 门控融合 | h = g*h + (1-g)*v | 自适应权重 | 额外参数 |
    | 交叉注意力 | h = CrossAttn(h, v) | 灵活交互 | 计算量大 |
    | 投影相加 | h = h + Proj(v) | 维度对齐 | 需要训练投影 |

    Qwen3-VL选择：直接相加（简单有效）
    """

    print(comparison)
    print("=" * 70)


if __name__ == "__main__":
    # 1. 展示实际实现
    show_actual_implementation()

    # 2. 可视化融合过程
    visualize_fusion_process()

    # 3. 解释融合优势
    explain_fusion_benefits()

    # 4. 对比融合方法
    compare_fusion_methods()

    print("\n" + "=" * 70)
    print("✅ 总结：DeepStack通过在LLM的不同深度注入不同层次的视觉特征，")
    print("   实现了多层次、渐进式的视觉-语言融合。")
    print("=" * 70)