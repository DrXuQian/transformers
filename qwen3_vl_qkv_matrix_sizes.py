"""
Qwen3-VL (Dense版本) QKV和Output Projection矩阵大小分析
基于代码和配置文件
"""

def analyze_qwen3_vl_matrix_sizes():
    """
    分析Qwen3-VL的QKV和Output projection矩阵大小
    """

    print("=" * 80)
    print("Qwen3-VL QKV和Output Projection矩阵大小")
    print("=" * 80)

    # 配置参数（基于Qwen3-VL-7B-Instruct）
    vision_config = {
        'hidden_size': 1024,
        'num_heads': 16,
        'head_dim': 64,  # hidden_size // num_heads
    }

    text_config = {
        'hidden_size': 2560,
        'num_attention_heads': 32,
        'num_key_value_heads': 8,  # GQA
        'head_dim': 80,  # hidden_size // num_attention_heads
    }

    print("\n📊 1. Vision Encoder的注意力矩阵")
    print("-" * 40)
    print(f"配置：")
    print(f"• hidden_size = {vision_config['hidden_size']}")
    print(f"• num_heads = {vision_config['num_heads']}")
    print(f"• head_dim = {vision_config['head_dim']}")

    print(f"\n矩阵大小：")

    # Vision使用合并的QKV
    print(f"\n合并的QKV矩阵 (self.qkv):")
    qkv_in = vision_config['hidden_size']
    qkv_out = vision_config['hidden_size'] * 3
    print(f"• nn.Linear({qkv_in}, {qkv_out}, bias=True)")
    print(f"• 参数量: {qkv_in * qkv_out} + {qkv_out} (bias) = {qkv_in * qkv_out + qkv_out:,}")

    print(f"\n分解后：")
    print(f"• Q: [{qkv_in}, {vision_config['hidden_size']}]")
    print(f"• K: [{qkv_in}, {vision_config['hidden_size']}]")
    print(f"• V: [{qkv_in}, {vision_config['hidden_size']}]")

    print(f"\nOutput Projection (self.proj):")
    o_in = vision_config['hidden_size']
    o_out = vision_config['hidden_size']
    print(f"• nn.Linear({o_in}, {o_out}, bias=False)")
    print(f"• 参数量: {o_in * o_out:,}")

    print(f"\n每个Vision Block的注意力参数总量：")
    vision_attn_params = qkv_in * qkv_out + qkv_out + o_in * o_out
    print(f"• {vision_attn_params:,} 参数")
    print(f"• 24层总计: {vision_attn_params * 24:,} 参数")

    print("\n" + "=" * 80)
    print("📊 2. LLM Decoder的注意力矩阵")
    print("-" * 40)
    print(f"配置：")
    print(f"• hidden_size = {text_config['hidden_size']}")
    print(f"• num_attention_heads = {text_config['num_attention_heads']}")
    print(f"• num_key_value_heads = {text_config['num_key_value_heads']} (GQA 4:1)")
    print(f"• head_dim = {text_config['head_dim']}")

    print(f"\n矩阵大小：")

    # LLM使用分离的Q、K、V
    print(f"\nQ Projection (self.q_proj):")
    q_in = text_config['hidden_size']
    q_out = text_config['num_attention_heads'] * text_config['head_dim']
    print(f"• nn.Linear({q_in}, {q_out}, bias=False)")
    print(f"• 矩阵形状: [{q_in}, {q_out}]")
    print(f"• 参数量: {q_in * q_out:,}")

    print(f"\nK Projection (self.k_proj):")
    k_in = text_config['hidden_size']
    k_out = text_config['num_key_value_heads'] * text_config['head_dim']
    print(f"• nn.Linear({k_in}, {k_out}, bias=False)")
    print(f"• 矩阵形状: [{k_in}, {k_out}]")
    print(f"• 参数量: {k_in * k_out:,}")

    print(f"\nV Projection (self.v_proj):")
    v_in = text_config['hidden_size']
    v_out = text_config['num_key_value_heads'] * text_config['head_dim']
    print(f"• nn.Linear({v_in}, {v_out}, bias=False)")
    print(f"• 矩阵形状: [{v_in}, {v_out}]")
    print(f"• 参数量: {v_in * v_out:,}")

    print(f"\nOutput Projection (self.o_proj):")
    o_in = text_config['num_attention_heads'] * text_config['head_dim']
    o_out = text_config['hidden_size']
    print(f"• nn.Linear({o_in}, {o_out}, bias=False)")
    print(f"• 矩阵形状: [{o_in}, {o_out}]")
    print(f"• 参数量: {o_in * o_out:,}")

    print(f"\n每个Decoder Block的注意力参数总量：")
    text_attn_params = q_in * q_out + k_in * k_out + v_in * v_out + o_in * o_out
    print(f"• {text_attn_params:,} 参数")
    print(f"• 36层总计: {text_attn_params * 36:,} 参数")

    print("\n" + "=" * 80)
    print("📊 3. 对比总结")
    print("-" * 40)

    print("\nVision Encoder (每层):")
    print(f"• QKV: 1024 → 3072 (合并)")
    print(f"• O:   1024 → 1024")
    print(f"• 总参数: {vision_attn_params:,}")

    print("\nLLM Decoder (每层):")
    print(f"• Q: 2560 → 2560")
    print(f"• K: 2560 → 640 (GQA)")
    print(f"• V: 2560 → 640 (GQA)")
    print(f"• O: 2560 → 2560")
    print(f"• 总参数: {text_attn_params:,}")

    print("\n关键差异:")
    print("1. Vision使用合并的QKV矩阵，LLM使用分离的Q、K、V")
    print("2. Vision没有GQA，LLM使用4:1 GQA")
    print("3. Vision有bias，LLM没有bias")
    print("4. LLM的注意力参数量更大（约2.8倍）")

    print("\n" + "=" * 80)
    print("📐 4. 实际计算示例")
    print("-" * 40)

    print("\n假设输入序列长度为1000:")

    print("\nVision Encoder:")
    print("• 输入: [1000, 1024]")
    print("• QKV输出: [1000, 3072]")
    print("• 拆分后: Q[1000, 1024], K[1000, 1024], V[1000, 1024]")
    print("• 重塑为多头: Q[1000, 16, 64], K[1000, 16, 64], V[1000, 16, 64]")
    print("• 注意力输出: [1000, 1024]")
    print("• 经过O_proj: [1000, 1024]")

    print("\nLLM Decoder:")
    print("• 输入: [1000, 2560]")
    print("• Q输出: [1000, 2560]")
    print("• K输出: [1000, 640]")
    print("• V输出: [1000, 640]")
    print("• 重塑为多头: Q[1000, 32, 80], K[1000, 8, 80], V[1000, 8, 80]")
    print("• K/V通过repeat扩展到32头")
    print("• 注意力输出: [1000, 2560]")
    print("• 经过O_proj: [1000, 2560]")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_qwen3_vl_matrix_sizes()