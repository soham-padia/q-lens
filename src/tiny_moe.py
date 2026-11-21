"""Build a tiny random-weight model with the Qwen3.5-122B-A10B architecture (hybrid Gated-DeltaNet + MoE,
VLM wrapper) and save it in bf16 with the real tokenizer: a cheap end-to-end test of the multi-GPU /
mixed-precision / MoE-jvp path in second_order.py.  Usage: python src/tiny_moe.py OUT_DIR
"""
import sys

import torch
import transformers

src, out = "Qwen/Qwen3.5-122B-A10B", sys.argv[1]
cfg = transformers.AutoConfig.from_pretrained(src)
t = cfg.text_config
for k, v in dict(hidden_size=256, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                 linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                 num_experts=16, num_experts_per_tok=4, moe_intermediate_size=64,
                 shared_expert_intermediate_size=64).items():
    setattr(t, k, v)
t.layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(t.num_hidden_layers)]
cfg.vision_config.depth = 1
cfg.vision_config.out_hidden_size = t.hidden_size
torch.manual_seed(0)
model = transformers.AutoModelForImageTextToText.from_config(cfg).to(torch.bfloat16)
model.save_pretrained(out)
transformers.AutoTokenizer.from_pretrained(src).save_pretrained(out)
print("saved", out, sum(p.numel() for p in model.parameters()) / 1e6, "M params")
