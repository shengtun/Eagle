import math
import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

class AttentionPatcher:
    """
    Monkey-patch manager for Qwen2_5_VLAttention.forward.
    Use:
      p = AttentionPatcher(cfg=None)
      p.enable()
      ... run model ...
      p.disable()
    """
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg
        self._orig_forward = None

    def enable(self):
        if self._orig_forward is not None:
            return
        self._orig_forward = Qwen2_5_VLAttention.forward

        def _patched_forward(
            layer,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            if past_key_value is None and past_key_values is not None:
                past_key_value = past_key_values

            bsz, q_len, _ = hidden_states.size()

            query_states = layer.q_proj(hidden_states)
            key_states = layer.k_proj(hidden_states)
            value_states = layer.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, -1, layer.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, -1, layer.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, -1, layer.head_dim).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, layer.rope_scaling["mrope_section"]
            )

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, layer.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, layer.num_key_value_groups)
            value_states = repeat_kv(value_states, layer.num_key_value_groups)

            attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(layer.head_dim)

            cfg = self.cfg if self.cfg is not None else getattr(layer, "_attn_edit_cfg", None)
            if cfg is not None and cfg.get("enabled", False):
                if (not cfg.get("decode_only", True)) or (q_len == 1):
                    layer_ok = (cfg.get("layers") is None) or (layer.layer_idx in cfg["layers"])
                    if layer_ok:
                        heads_mask = cfg.get("heads_mask", None)
                        img_slice = cfg.get("img_slice", None)
                        text_slice = cfg.get("text_slice", None)

                        w_img = float(cfg.get("weight_img", 1.0))
                        w_txt = float(cfg.get("weight_txt", 1.0))
                        boost = float(cfg.get("boost_img", 0.0))
                        suppr = float(cfg.get("suppr_txt", 0.0))

                        hs = slice(None) if heads_mask is None else heads_mask

                        if img_slice is not None and w_img != 1.0:
                            attn_logits[:, :, :, img_slice] *= w_img
                        if text_slice is not None and w_txt != 1.0:
                            attn_logits[:, :, :, text_slice] *= w_txt

                        if img_slice is not None and boost != 0.0:
                            attn_logits[:, hs, -1, img_slice] += boost
                        if text_slice is not None and suppr != 0.0:
                            attn_logits[:, hs, -1, text_slice] -= suppr

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_logits = attn_logits + causal_mask

            if query_states.dtype == torch.float16:
                attn_logits = torch.where(torch.isinf(attn_logits), torch.zeros_like(attn_logits), attn_logits)

            attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=layer.attention_dropout, training=layer.training)

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = layer.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        Qwen2_5_VLAttention.forward = _patched_forward

    def disable(self):
        if self._orig_forward is None:
            return
        Qwen2_5_VLAttention.forward = self._orig_forward
        self._orig_forward = None

    def set_cfg(self, cfg: dict | None):
        self.cfg = cfg

import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
import torch
import torch.nn.functional as F

class EagerAttnPatcher:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._orig = None

    def enable(self):
        if self._orig is not None:
            return
        self._orig = qwen2_mod.eager_attention_forward
        outer = self

        def patched_eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=0.0,
            **kwargs,
        ):
            key_states = qwen2_mod.repeat_kv(key, module.num_key_value_groups)
            value_states = qwen2_mod.repeat_kv(value, module.num_key_value_groups)

            attn_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_logits = attn_logits + causal_mask

            # ====== ✅ 在 softmax 前改 logits ======
            cfg = outer.cfg if outer.cfg is not None else getattr(module, "_attn_edit_cfg", None)
            if cfg is not None and cfg.get("enabled", False):
                # attn_logits: [B, H, Q, K]
                B, H, Q, K = attn_logits.shape
                decode_only = cfg.get("decode_only", True)
                if (not decode_only) or (Q == 1):
                    layer_ok = (cfg.get("layers") is None) or (getattr(module, "layer_idx", None) in cfg["layers"])
                    if layer_ok:
                        hs = slice(None) if cfg.get("heads_mask") is None else cfg["heads_mask"]
                        img_slice = cfg.get("img_slice", None)
                        txt_slice = cfg.get("text_slice", None)

                        w_img = float(cfg.get("weight_img", 1.0))
                        w_txt = float(cfg.get("weight_txt", 1.0))
                        boost = float(cfg.get("boost_img", 0.0))
                        suppr = float(cfg.get("suppr_txt", 0.0))

                        if img_slice is not None and w_img != 1.0:
                            attn_logits[:, hs, :, img_slice] *= w_img
                        if txt_slice is not None and w_txt != 1.0:
                            attn_logits[:, hs, :, txt_slice] *= w_txt

                        # last query token bias
                        if img_slice is not None and boost != 0.0:
                            attn_logits[:, hs, -1, img_slice] += boost
                        if txt_slice is not None and suppr != 0.0:
                            attn_logits[:, hs, -1, txt_slice] -= suppr

                        # 可选：debug 一次确认命中
                        if cfg.get("debug", False) and not hasattr(module, "_patch_printed"):
                            module._patch_printed = True
                            print(f"[PATCH HIT] layer={getattr(module,'layer_idx',None)} Q={Q} K={K}")

            attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        qwen2_mod.eager_attention_forward = patched_eager_attention_forward

    def disable(self):
        if self._orig is None:
            return
        qwen2_mod.eager_attention_forward = self._orig
        self._orig = None

    def set_cfg(self, cfg):
        self.cfg = cfg
