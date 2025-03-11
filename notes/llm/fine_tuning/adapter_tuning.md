# Adapter Tuning
在微调时将模型主体冻结，仅训练特定于任务的参数.

1. adapter tuning
   设计Adapter结构，并将其嵌入Transformer的结构里面，针对每一个Transformer层，增加了两个Adapter结构(分别是多头注意力的投影之后和第二个feed-forward层之后)，
   在训练时，固定住原来预训练模型的参数不变，只对新增的 Adapter 结构和 Layer Norm 层进行微调，从而保证了训练的高效性。

   结构：input(transformer输出) -> Feedforward(d, m) -> activation -> Feedforward(m, d)，同时引入残差机制：
      $h \leftarrow h + f(hW_{down})W_{up}$; 通过控制 $m$ 的大小来限制 Adapter 的参数量。

2. lora
   llm在预训练后，越大的模型权重矩阵的秩越小，于是可以通过低秩分解来减少参数量。
   操作：将fine-tune的参数矩阵W变成两个小矩阵的乘积 W=AB，即：$W_0+\Delta W=W_0 +AB$

   流程：
   初始化：A（高斯分布），B（初始化为0）
   插入位置：Q,K,V,O等矩阵上， 

`lora_config = LoraConfig(target_modules=['up_proj', 'gate_proj', 'q_proj', 'o_proj', 'down_proj', 'v_proj', 'k_proj'])`
```text 
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(102400, 4096)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-06)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-06)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-06)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=102400, bias=False)
)
```