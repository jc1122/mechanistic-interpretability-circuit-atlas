"""Minimal GPT-NeoX forward pass in plain torch (parameter names follow Hugging Face's GPTNeoXForCausalLM, so a state dict built
here also loads there). Supported config: no rotary embedding, sequential residual (x + attn(LN x); then + mlp(LN x)), exact GELU,
LayerNorm eps 1e-5, untied output head. Causal attention, scores divided by sqrt(d_head)."""
import math
import torch
import torch.nn.functional as F


def names(cfg):
    """{parameter name: shape}."""
    D, DM, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    s = {"gpt_neox.embed_in.weight": (V, D)}
    for l in range(cfg["num_hidden_layers"]):
        p = f"gpt_neox.layers.{l}."
        for ln in ("input_layernorm", "post_attention_layernorm"):
            s[p + ln + ".weight"] = s[p + ln + ".bias"] = (D,)
        for n, o, i in (("attention.query_key_value", 3 * D, D), ("attention.dense", D, D), ("mlp.dense_h_to_4h", DM, D), ("mlp.dense_4h_to_h", D, DM)):
            s[p + n + ".weight"] = (o, i); s[p + n + ".bias"] = (o,)
    s["gpt_neox.final_layer_norm.weight"] = s["gpt_neox.final_layer_norm.bias"] = (D,)
    s["lm_head.weight"] = (V, D)
    return s


def zeros(cfg):
    return {k: torch.zeros(v) for k, v in names(cfg).items()}


def random_init(cfg, seed):
    """Random-weight control: weights N(0, 0.02), biases 0, LayerNorm gain 1 (the Hugging Face initialiser's distribution)."""
    g = torch.Generator().manual_seed(seed); sd = {}
    for k, shp in names(cfg).items():
        if k.endswith(".bias"): sd[k] = torch.zeros(shp)
        elif "layernorm" in k or "layer_norm" in k: sd[k] = torch.ones(shp)
        else: sd[k] = torch.randn(shp, generator=g) * 0.02
    return sd


@torch.no_grad()
def forward(sd, cfg, ids):
    """ids [B, S] -> (residual after the last block [B, S, D], logits [B, S, V])."""
    D, H, S = cfg["hidden_size"], cfg["num_attention_heads"], ids.shape[1]; dh = D // H
    ln = lambda x, p: F.layer_norm(x, (D,), sd[p + ".weight"], sd[p + ".bias"], cfg.get("layer_norm_eps", 1e-5))
    lin = lambda x, p: F.linear(x, sd[p + ".weight"], sd[p + ".bias"])
    x = sd["gpt_neox.embed_in.weight"][ids]; causal = torch.ones(S, S, dtype=torch.bool).tril()
    for l in range(cfg["num_hidden_layers"]):
        p = f"gpt_neox.layers.{l}."
        qkv = lin(ln(x, p + "input_layernorm"), p + "attention.query_key_value").view(*ids.shape, H, 3 * dh).transpose(1, 2)  # per head [q | k | v]
        q, k, v = qkv.chunk(3, -1)
        att = ((q @ k.transpose(-1, -2)) / math.sqrt(dh)).masked_fill(~causal, float("-inf")).softmax(-1)
        x = x + lin((att @ v).transpose(1, 2).reshape(*ids.shape, D), p + "attention.dense")
        x = x + lin(F.gelu(lin(ln(x, p + "post_attention_layernorm"), p + "mlp.dense_h_to_4h")), p + "mlp.dense_4h_to_h")
    return x, F.linear(ln(x, "gpt_neox.final_layer_norm"), sd["lm_head.weight"])
