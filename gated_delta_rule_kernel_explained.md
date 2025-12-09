# Gated Delta Rule Kernel 内部实现详解

## 📌 概述

`chunk_gated_delta_rule` 和 `recurrent_gated_delta_rule` 是Qwen3-Next Linear Attention的核心算法，实现了线性复杂度的注意力机制。

## 1. chunk_gated_delta_rule（块并行版本）

### 🎯 核心思想
将长序列分成固定大小的块（chunk），在块内并行计算，块间递归传递状态。

### 📊 算法流程

```python
def torch_chunk_gated_delta_rule(
    query,          # [B, L, H, D] 查询向量
    key,            # [B, L, H, D] 键向量
    value,          # [B, L, H, D] 值向量
    g,              # [B, L, H] 衰减门
    beta,           # [B, L, H] 信息门
    chunk_size=64,  # 块大小
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
```

### 🔍 详细步骤解析

#### Step 1: 预处理和归一化
```python
# L2归一化（可选）
if use_qk_l2norm_in_kernel:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

# 转置并转为float32精度
query, key, value, beta, g = [
    x.transpose(1, 2).contiguous().to(torch.float32)
    for x in (query, key, value, beta, g)
]

# 缩放query（类似传统注意力的1/√d）
scale = 1 / (query.shape[-1] ** 0.5)
query = query * scale
```

#### Step 2: Padding和重塑为块
```python
# 填充到chunk_size的整数倍
pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
query = F.pad(query, (0, 0, 0, pad_size))
key = F.pad(key, (0, 0, 0, pad_size))
value = F.pad(value, (0, 0, 0, pad_size))

# 预计算beta加权的kv
v_beta = value * beta.unsqueeze(-1)  # [B, H, L, D]
k_beta = key * beta.unsqueeze(-1)    # [B, H, L, D]

# 重塑为块: [B, H, num_chunks, chunk_size, D]
query = query.reshape(B, H, -1, chunk_size, D)
key = key.reshape(B, H, -1, chunk_size, D)
value = value.reshape(B, H, -1, chunk_size, D)
```

#### Step 3: 块内衰减计算
```python
# 累积衰减因子
g = g.cumsum(dim=-1)  # 累积和

# 计算衰减掩码矩阵
# decay_mask[i,j] = exp(g[j] - g[i]) if i <= j else 0
decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

# 块内注意力矩阵（带衰减）
attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
```

#### Step 4: 递归累积（核心创新）
```python
# 递归计算累积注意力
for i in range(1, chunk_size):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    # 递归公式：当前行 = 直接连接 + 通过之前所有状态的间接连接
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

# 添加单位矩阵（自注意力）
attn = attn + torch.eye(chunk_size)

# 计算块内输出
value = attn @ v_beta
k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
```

#### Step 5: 块间递归
```python
# 初始化递归状态 [B, H, K_dim, V_dim]
last_recurrent_state = torch.zeros(B, H, K_dim, V_dim)

# 遍历每个块
for i in range(num_chunks):
    q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]

    # 块内注意力
    attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i])

    # 从上个块继承的值
    v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
    v_new = v_i - v_prime  # 增量计算

    # 与历史状态的注意力
    attn_inter = (q_i * g[:, :, i].exp()) @ last_recurrent_state

    # 合并输出
    output[:, :, i] = attn_inter + attn @ v_new

    # 更新递归状态（带衰减）
    last_recurrent_state = (
        last_recurrent_state * g[:, :, i, -1].exp()  # 衰减旧状态
        + k_i.T @ v_new  # 添加新信息
    )
```

### 🎨 关键创新点

1. **块内并行**：chunk_size个token可以并行计算
2. **递归累积**：巧妙的递归公式实现高效累积
3. **增量计算**：`v_new = v_i - v_prime`避免重复计算
4. **衰减传递**：通过`decay_mask`精确控制信息衰减

## 2. recurrent_gated_delta_rule（逐步递归版本）

### 🎯 核心思想
逐个token处理，适合自回归生成和单token推理。

### 📊 算法流程

```python
def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False
):
```

### 🔍 详细步骤解析

#### Step 1: 初始化
```python
# 预处理（同chunk版本）
query = query * scale  # 缩放

# 初始化输出和状态
output = torch.zeros(B, H, L, V_dim)
last_recurrent_state = torch.zeros(B, H, K_dim, V_dim)  # 累积状态矩阵
```

#### Step 2: 逐token递归（核心循环）
```python
for i in range(sequence_length):
    # 获取当前时刻的输入
    q_t = query[:, :, i]      # [B, H, D]
    k_t = key[:, :, i]        # [B, H, D]
    v_t = value[:, :, i]      # [B, H, D]
    g_t = g[:, :, i].exp()    # [B, H] 衰减因子
    beta_t = beta[:, :, i]    # [B, H] 信息门

    # Step 2.1: 衰减历史状态
    last_recurrent_state = last_recurrent_state * g_t  # 应用衰减

    # Step 2.2: 计算预期值（基于历史）
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
    # kv_mem是基于当前key和历史state预测的value

    # Step 2.3: 计算增量（Delta Rule核心）
    delta = (v_t - kv_mem) * beta_t  # 实际值与预期值的差，由beta门控制

    # Step 2.4: 更新状态
    # 将当前k-v对添加到状态中（外积）
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    # 状态矩阵shape: [B, H, K_dim, V_dim]

    # Step 2.5: 生成输出
    # 用query查询累积状态
    output[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
```

### 🎨 数学原理

#### Delta Rule的核心公式

1. **状态更新方程**：
   ```
   S_t = g_t * S_{t-1} + β_t * (k_t ⊗ (v_t - k_t^T S_{t-1}))
   ```
   - `S_t`: 时刻t的状态矩阵
   - `g_t`: 衰减因子（控制历史信息保留程度）
   - `β_t`: 信息门（控制新信息接受程度）
   - `k_t ⊗ v_t`: 键值外积

2. **输出计算**：
   ```
   o_t = q_t^T S_t
   ```

3. **Delta机制**：
   ```
   delta = (v_actual - v_predicted) * beta
   ```
   只更新预测误差部分，提高效率

### 🔄 两个版本的对比

| 特性 | chunk_gated_delta_rule | recurrent_gated_delta_rule |
|------|------------------------|---------------------------|
| **处理方式** | 块并行 | 逐token串行 |
| **复杂度** | O(L/C × C²) ≈ O(L×C) | O(L) |
| **并行度** | 高（块内并行） | 低（完全串行） |
| **内存占用** | 较高（存储块矩阵） | 较低（只存状态） |
| **适用场景** | 训练、批处理 | 推理、流式生成 |
| **精度** | 完全精确 | 完全精确 |
| **实现复杂度** | 复杂（递归累积） | 简单（直接循环） |

## 3. 关键优化技巧

### 🚀 性能优化

1. **混合精度计算**
   ```python
   # 内部使用float32避免精度损失
   x.to(torch.float32)
   # 输出转回原始精度
   output.to(initial_dtype)
   ```

2. **增量计算**
   ```python
   # 不重新计算全部，只计算变化部分
   v_new = v_i - v_prime
   ```

3. **预计算优化**
   ```python
   # 提前计算beta加权
   v_beta = value * beta.unsqueeze(-1)
   k_beta = key * beta.unsqueeze(-1)
   ```

4. **内存复用**
   ```python
   # 原地操作减少内存分配
   attn.masked_fill_(mask, 0)
   ```

### 💡 数值稳定性

1. **L2归一化**：防止数值爆炸
2. **缩放因子**：`1/√d`保持梯度稳定
3. **指数衰减**：使用log空间计算避免溢出
4. **增量更新**：减少累积误差

## 4. 实际应用示例

### 训练时使用chunk版本
```python
# 长序列训练，利用并行加速
output, final_state = chunk_gated_delta_rule(
    Q, K, V, g, beta,
    chunk_size=64,  # 平衡并行度和内存
    output_final_state=True  # 保存状态用于下一层
)
```

### 推理时使用recurrent版本
```python
# 自回归生成，逐token处理
for token in generate_tokens():
    output, state = recurrent_gated_delta_rule(
        q_token, k_token, v_token, g, beta,
        initial_state=state,  # 使用上一步的状态
        output_final_state=True
    )
```

## 5. 总结

这两个kernel实现了Gated Delta Rule的核心算法：

- **chunk版本**：通过巧妙的块内递归累积和块间状态传递，实现了高效的并行计算
- **recurrent版本**：通过简洁的逐步递归，实现了低内存的流式处理

两者的核心创新在于：
1. 使用**递归状态矩阵**替代显式注意力矩阵
2. 通过**Delta机制**只更新预测误差
3. 使用**门控机制**精确控制信息流
4. 实现了**线性复杂度**的注意力计算

这使得模型能够高效处理超长序列，是Qwen3-Next的关键技术突破。