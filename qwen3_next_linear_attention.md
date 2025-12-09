# Qwen3-Next Linear Attention (Gated Delta Net) 详解

## 📚 1. 什么是Linear Attention？

### 传统注意力 vs 线性注意力

| 特性 | 传统注意力（Softmax Attention） | 线性注意力（Linear Attention） |
|------|----------------------------------|--------------------------------|
| **时间复杂度** | O(N²) - N是序列长度 | O(N) - 线性复杂度！ |
| **空间复杂度** | O(N²) | O(1) |
| **计算方式** | Attention = softmax(QK^T/√d)V | 递归/累积更新 |
| **注意力矩阵** | 需要存储完整的N×N矩阵 | 无需显式矩阵 |
| **长序列性能** | 内存和计算开销巨大 | 可处理100k+长度 |
| **实现方式** | 标准矩阵运算 | Gated Delta Net |

## 🏗️ 2. Qwen3-Next的Gated Delta Net架构

### 核心组件配置

#### 投影维度
```python
# 配置参数
linear_key_head_dim = 128      # K头维度
linear_value_head_dim = 128    # V头维度
linear_num_key_heads = 16      # K头数量
linear_num_value_heads = 32    # V头数量

# 计算总维度
key_dim = 128 × 16 = 2048
value_dim = 128 × 32 = 4096
```

#### 卷积组件
- **卷积核大小**: `linear_conv_kernel_dim = 4`
- **类型**: 1D深度卷积（depthwise）
- **特点**: 每个通道独立卷积（groups=channels）
- **作用**: 序列建模，捕获局部依赖

#### 门控机制
- **Beta门（β）**: 控制信息流动
- **Alpha门（α）**: 控制衰减率
- **Z门**: 用于归一化

## ⚙️ 3. 核心计算流程

### Step 1: 输入投影

```
hidden_states [B, L, D]
         ↓
    ┌────┴────┐
    │         │
QKVZ投影   BA投影
    │         │
    ↓         ↓
Q,K,V,Z    Beta,Alpha
```

**QKVZ投影**: `Linear(D, 2*key_dim + 2*value_dim)`
- Query: `[B, L, 16, 128]`
- Key: `[B, L, 16, 128]`
- Value: `[B, L, 32, 128]`
- Z: `[B, L, 32, 128]`

**BA投影**: `Linear(D, 2*num_v_heads)`
- Beta: `[B, L, 32]` → sigmoid激活
- Alpha: `[B, L, 32]` → 计算衰减率

### Step 2: 因果卷积

```
QKV混合 → Conv1D(kernel=4, causal) → 激活(SiLU)
```

**作用**：
- 捕获局部依赖关系
- 保持因果性（只看过去信息）
- 增强序列建模能力

### Step 3: Gated Delta Rule（核心！）

#### A. Chunk模式（训练/长序列）
```python
chunk_gated_delta_rule(Q, K, V, g, beta)
```
- 将序列分块处理
- 块内并行计算
- 块间递归传递状态

#### B. Recurrent模式（推理/单token）
```python
recurrent_gated_delta_rule(Q, K, V, g, beta, state)
```
- 逐token递归更新
- 维护累积状态
- 适合自回归生成

## 🔬 4. Gated Delta Rule数学原理

### 核心公式

#### 1. 衰减门计算
```python
g = -exp(A_log) * softplus(alpha + dt_bias)
```
- `A_log`: 可学习的衰减参数
- `alpha`: 输入相关的衰减调节
- `dt_bias`: 时间步偏置

#### 2. 信息门
```python
beta = sigmoid(b)
```
控制新信息的接受程度

#### 3. 递归更新（简化版）
```python
# 初始化
state = 0

# 对每个时间步t:
state = g[t] * state + beta[t] * (k[t] ⊗ v[t])
output[t] = q[t] · state
```

#### 4. L2归一化
- Q和K在计算前进行L2归一化
- 确保数值稳定性

### 实际实现特性
- 多头并行处理
- 块级优化
- 融合算子加速

## 🚀 5. 性能优势

### 复杂度对比

| 操作 | 传统注意力 | Linear Attention | 优势倍数 |
|------|-----------|------------------|----------|
| **时间复杂度** | O(N²) | O(N) | N倍 |
| **空间复杂度** | O(N²) | O(1) | N²倍 |
| **KV Cache** | O(N×D) | O(D) | N倍 |

### 实际性能提升

| 序列长度 | 性能提升 | 内存节省 |
|---------|---------|----------|
| 1k | ~10x | ~1000x |
| 10k | ~100x | ~100,000x |
| 100k | ~10,000x | ~10,000,000x |

### 适用场景
- ✅ 超长文档处理
- ✅ 流式推理
- ✅ 内存受限环境
- ✅ 实时生成任务

## 💡 6. 实现细节

### 类结构
```python
class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        # 替代传统的MultiHeadAttention
        # 每个decoder层可选择使用
```

### 优化实现
1. **融合算子**
   - 使用FLA库的融合算子（如果可用）
   - 回退到PyTorch纯实现
   - Causal Conv1D专用CUDA核

2. **状态缓存**
   ```python
   conv_states: [B, C, K-1]      # 卷积状态
   recurrent_states: [B, H, D, D] # 递归状态
   ```
   - 支持KV Cache兼容接口

3. **混合架构**
   - 可与传统注意力层交替使用
   - 例如：`["linear", "linear", "softmax", "linear", ...]`
   - 灵活配置每层类型

## 📊 7. 与传统注意力的对比

| 特性 | Softmax Attention | Gated Delta Net |
|-----|------------------|-----------------|
| **复杂度** | O(N²) | O(N) |
| **长程依赖** | ✅ 完美 | ⚠️ 近似 |
| **可解释性** | ✅ 注意力权重 | ❌ 隐式状态 |
| **训练稳定性** | ✅ 成熟 | ⚠️ 需要调优 |
| **推理效率** | ❌ 慢 | ✅ 快 |
| **内存效率** | ❌ 高 | ✅ 低 |
| **并行化** | ✅ 完全并行 | ⚠️ 块级并行 |

### 设计权衡
- 牺牲一定的表达能力换取效率
- 适合需要处理超长序列的场景
- 在某些任务上可能略逊于传统注意力

## 🔧 8. 配置示例

### Qwen3-Next-80B配置

```json
{
    // Linear Attention配置
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,

    // 标准Attention配置（对比）
    "num_attention_heads": 16,
    "num_key_value_heads": 2,  // GQA 8:1
    "hidden_size": 2048
}
```

### 层类型配置
可通过`layer_types`指定每层使用哪种注意力：
```python
layer_types = ["linear", "linear", "standard", "linear", ...]
```

## 📝 9. 代码实现细节

### chunk_gated_delta_rule (训练时使用)

```python
def chunk_gated_delta_rule(
    query,      # [B, L, H, D] - 查询向量
    key,        # [B, L, H, D] - 键向量
    value,      # [B, L, H, D] - 值向量
    g,          # [B, L, H] - 衰减门
    beta,       # [B, L, H] - 信息门
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=True
):
    # 1. L2归一化Q和K
    if use_qk_l2norm_in_kernel:
        query = l2norm(query)
        key = l2norm(key)

    # 2. 分块处理
    for chunk in chunks:
        # 块内并行计算
        state = update_state(chunk, g, beta)
        output = compute_output(query, state)

    return output, final_state
```

### recurrent_gated_delta_rule (推理时使用)

```python
def recurrent_gated_delta_rule(
    query, key, value, g, beta,
    initial_state, ...
):
    state = initial_state

    # 逐token递归
    for t in range(seq_len):
        # 状态更新
        state = g[t] * state + beta[t] * outer(k[t], v[t])
        # 输出计算
        output[t] = dot(q[t], state)

    return output, state
```

### 因果卷积处理

```python
# 卷积配置
self.conv1d = nn.Conv1d(
    in_channels=conv_dim,
    out_channels=conv_dim,
    kernel_size=4,          # 卷积核大小
    groups=conv_dim,        # 深度卷积
    padding=3,              # 因果padding
)

# 应用卷积
mixed_qkv = self.causal_conv1d_fn(
    x=mixed_qkv,
    weight=self.conv1d.weight,
    activation="silu"       # SiLU激活
)
```

## 🔍 10. 性能优化技巧

### 1. 融合算子
- 使用FLA库的CUDA核心
- 减少内存访问次数
- 算子级优化

### 2. 混合精度
- FP16/BF16计算
- FP32累积
- 梯度缩放

### 3. 状态管理
- 增量更新而非完全重算
- 高效的缓存机制
- 最小化内存拷贝

### 4. 并行策略
- **头并行**: 多头独立计算
- **序列并行**: 长序列分片
- **张量并行**: 模型并行

## 📌 总结

Qwen3-Next的Linear Attention (Gated Delta Net)是一个重要创新：

1. **线性复杂度**: O(N)时间和O(1)空间
2. **门控机制**: Beta门和衰减门精确控制信息流
3. **因果卷积**: 增强局部建模能力
4. **递归更新**: 高效的状态传递
5. **混合架构**: 可与传统注意力灵活组合

这使得Qwen3-Next能够高效处理超长序列（100k+），为长文本理解和生成任务提供了新的可能性。

## 参考资料

- [Qwen3-Next模型代码](./src/transformers/models/qwen3_next/modeling_qwen3_next.py)
- [FLA库](https://github.com/fla-org/flash-linear-attention)
- [Causal Conv1D](https://github.com/Dao-AILab/causal-conv1d)