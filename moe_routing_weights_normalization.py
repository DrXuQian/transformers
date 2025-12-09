"""
MoE中的归一化权重详解
基于Qwen3-VL-MoE的实际代码
"""

import torch
import torch.nn.functional as F

def explain_moe_routing_normalization():
    """
    解释MoE路由权重归一化的过程
    """

    print("=" * 80)
    print("MoE中的归一化权重（Normalized Routing Weights）详解")
    print("=" * 80)

    print("\n📝 1. 完整的路由过程")
    print("-" * 40)
    print("""
    基于Qwen3-VL-MoE的代码（modeling_qwen3_vl_moe.py:145-148）:

    # Step 1: 计算路由logits
    router_logits = self.gate(hidden_states)  # [batch*seq_len, num_experts]

    # Step 2: 应用softmax得到概率分布
    routing_weights = F.softmax(router_logits, dim=-1)  # [batch*seq_len, num_experts]

    # Step 3: 选择top-k专家
    routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
    # routing_weights: [batch*seq_len, top_k] - top-k个专家的分数
    # router_indices: [batch*seq_len, top_k] - top-k个专家的索引

    # Step 4: 归一化top-k的权重（关键步骤！）
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    """)

    print("\n🔍 2. 为什么需要归一化？")
    print("-" * 40)
    print("""
    原因分析：

    1. Softmax后的完整分布：
       - 所有128个专家的概率和 = 1.0
       - 例如：[0.15, 0.12, 0.08, 0.07, 0.06, ...] 总和 = 1.0

    2. 选择top-8后：
       - 只保留8个最高分数：[0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
       - 这8个分数的和 < 1.0 (例如：0.60)
       - 丢失了其他120个专家的40%概率质量

    3. 重新归一化的必要性：
       - 确保选中专家的权重和 = 1.0
       - 保持输出的数值稳定性
       - 避免信息损失
    """)

    print("\n💻 3. 具体示例")
    print("-" * 40)

    # 模拟一个例子
    torch.manual_seed(42)
    batch_seq_len = 2  # 2个token
    num_experts = 128  # 128个专家
    top_k = 8  # 选择top-8

    # Step 1: 模拟router logits
    router_logits = torch.randn(batch_seq_len, num_experts)
    print(f"Router logits shape: {router_logits.shape}")

    # Step 2: Softmax
    routing_weights_full = F.softmax(router_logits, dim=-1)
    print(f"\nSoftmax后（全部专家）:")
    print(f"• Shape: {routing_weights_full.shape}")
    print(f"• 每个token的概率和: {routing_weights_full.sum(dim=-1).tolist()}")

    # Step 3: Top-k selection
    routing_weights_topk, router_indices = torch.topk(routing_weights_full, top_k, dim=-1)
    print(f"\nTop-{top_k}选择后（归一化前）:")
    print(f"• Shape: {routing_weights_topk.shape}")
    print(f"• Top-{top_k}权重和: {routing_weights_topk.sum(dim=-1).tolist()}")
    print(f"• 第一个token的top-{top_k}权重: {routing_weights_topk[0].tolist()[:5]}... (显示前5个)")

    # Step 4: Normalization
    routing_weights_normalized = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
    print(f"\n归一化后:")
    print(f"• Shape: {routing_weights_normalized.shape}")
    print(f"• 归一化后权重和: {routing_weights_normalized.sum(dim=-1).tolist()}")
    print(f"• 第一个token的归一化权重: {routing_weights_normalized[0].tolist()[:5]}... (显示前5个)")

    # 对比归一化前后
    print(f"\n归一化效果对比（第一个token的前3个专家）:")
    for i in range(3):
        before = routing_weights_topk[0, i].item()
        after = routing_weights_normalized[0, i].item()
        scale = after / before
        print(f"• 专家{i}: {before:.4f} → {after:.4f} (放大{scale:.2f}倍)")

    print("\n" + "=" * 80)
    print("📊 4. 数学公式")
    print("-" * 40)
    print("""
    设：
    • S = softmax(router_logits) ∈ R^{N×E}  (N=tokens, E=experts)
    • W_topk, I_topk = topk(S, k)  (选择top-k)
    • W_topk ∈ R^{N×k}: top-k专家的原始softmax分数
    • I_topk ∈ R^{N×k}: top-k专家的索引

    归一化公式：
    W_normalized[i,j] = W_topk[i,j] / Σ(W_topk[i,:])

    确保：
    Σ(W_normalized[i,:]) = 1.0  ∀i ∈ [1,N]
    """)

    print("\n🎯 5. 实际影响")
    print("-" * 40)
    print("""
    归一化权重的作用：

    1. **数值稳定性**：
       - 确保加权求和时输出幅度正确
       - 避免输出值过小（未归一化时可能只有0.6倍）

    2. **梯度流**：
       - 保持梯度的合理范围
       - 避免梯度消失

    3. **专家负载均衡**：
       - 归一化后的权重更准确反映相对重要性
       - 有助于load balancing loss的计算

    4. **输出一致性**：
       - 无论选择多少专家，输出scale保持一致
       - output = Σ(W_normalized[i] * Expert_i(x))
    """)

    print("\n💡 6. Qwen3-VL-MoE的具体实现")
    print("-" * 40)
    print("""
    在Qwen3-VL-30B-A3B-Instruct中：
    • 总专家数：128
    • 激活专家数：8
    • 归一化确保这8个专家的权重和 = 1.0

    代码位置：modeling_qwen3_vl_moe.py
    第148行：routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    这行代码就是对top-8专家的权重进行重新归一化！
    """)

    print("=" * 80)


def compare_with_without_normalization():
    """
    对比有无归一化的差异
    """
    print("\n\n对比实验：有无归一化的差异")
    print("=" * 80)

    torch.manual_seed(42)

    # 模拟专家输出
    num_tokens = 1
    hidden_dim = 256
    num_experts = 128
    top_k = 8

    # Router logits和选择
    router_logits = torch.randn(num_tokens, num_experts)
    routing_weights = F.softmax(router_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)

    # 模拟每个专家的输出
    x = torch.randn(num_tokens, hidden_dim)
    expert_outputs = torch.randn(top_k, num_tokens, hidden_dim) * 2  # 专家输出

    print("实验设置：")
    print(f"• Top-{top_k}权重和（归一化前）: {topk_weights.sum().item():.4f}")
    print(f"• 输入x的L2范数: {x.norm().item():.4f}")

    # 无归一化的输出
    output_no_norm = torch.zeros_like(x)
    for i in range(top_k):
        output_no_norm += topk_weights[0, i] * expert_outputs[i, 0]

    # 有归一化的输出
    normalized_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    output_with_norm = torch.zeros_like(x)
    for i in range(top_k):
        output_with_norm += normalized_weights[0, i] * expert_outputs[i, 0]

    print(f"\n输出对比：")
    print(f"• 无归一化输出的L2范数: {output_no_norm.norm().item():.4f}")
    print(f"• 有归一化输出的L2范数: {output_with_norm.norm().item():.4f}")
    print(f"• 范数比例: {output_with_norm.norm().item() / output_no_norm.norm().item():.4f}")

    print(f"\n结论：")
    print(f"归一化使输出幅度增大约 {1/topk_weights.sum().item():.2f} 倍")
    print("这保证了模型各层的激活值保持在合理范围内")

    print("=" * 80)


if __name__ == "__main__":
    explain_moe_routing_normalization()
    compare_with_without_normalization()