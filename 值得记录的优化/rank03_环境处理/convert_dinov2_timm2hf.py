#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 timm 训练的 DINOv2-B/14 checkpoint 转成 transformers.Dinov2Model 命名权重, 供 submit_dinov2.py 使用.

timm (vit_base_patch14_dinov2) key -> transformers Dinov2Model key 映射:
  cls_token                       -> embeddings.cls_token
  pos_embed                       -> embeddings.position_embeddings
  patch_embed.proj.weight/bias    -> embeddings.patch_embeddings.projection.weight/bias
  norm.weight/bias                -> layernorm.weight/bias
  blocks.{i}.norm1.*              -> encoder.layer.{i}.norm1.*
  blocks.{i}.attn.qkv.weight/bias -> 拆成 query/key/value 各自 weight/bias
  blocks.{i}.attn.proj.*          -> encoder.layer.{i}.attention.output.dense.*
  blocks.{i}.ls1.gamma            -> encoder.layer.{i}.layer_scale1.lambda1
  blocks.{i}.norm2.*              -> encoder.layer.{i}.norm2.*
  blocks.{i}.mlp.fc1.*            -> encoder.layer.{i}.mlp.fc1.*
  blocks.{i}.mlp.fc2.*            -> encoder.layer.{i}.mlp.fc2.*
  blocks.{i}.ls2.gamma            -> encoder.layer.{i}.layer_scale2.lambda1
  head.weight/bias                -> head.weight/bias  (保留, submit 模型自写 head)

注意:
- timm DINOv2 attn 用融合 qkv (768->2304), transformers 用分离 q/k/v (各 768->768). 需沿 dim0 切3段.
- pos_embed: timm (1,257,768) (1 cls + 256 patch, 224/14=16, 16*16=256), transformers (1,257,768) 一致, 不需改.
"""
import os
import sys
import torch


def convert(timm_sd):
    hf_sd = {}
    for k, v in timm_sd.items():
        if k == "cls_token":
            hf_sd["embeddings.cls_token"] = v
        elif k == "pos_embed":
            hf_sd["embeddings.position_embeddings"] = v
        elif k == "patch_embed.proj.weight":
            hf_sd["embeddings.patch_embeddings.projection.weight"] = v
        elif k == "patch_embed.proj.bias":
            hf_sd["embeddings.patch_embeddings.projection.bias"] = v
        elif k == "norm.weight":
            hf_sd["layernorm.weight"] = v
        elif k == "norm.bias":
            hf_sd["layernorm.bias"] = v
        elif k.startswith("head."):
            hf_sd[k] = v  # head.weight / head.bias 保留
        elif k.startswith("blocks."):
            # blocks.{i}.{rest}
            _, i, rest = k.split(".", 2)
            li = f"encoder.layer.{i}."
            if rest == "norm1.weight":
                hf_sd[li + "norm1.weight"] = v
            elif rest == "norm1.bias":
                hf_sd[li + "norm1.bias"] = v
            elif rest == "norm2.weight":
                hf_sd[li + "norm2.weight"] = v
            elif rest == "norm2.bias":
                hf_sd[li + "norm2.bias"] = v
            elif rest == "ls1.gamma":
                hf_sd[li + "layer_scale1.lambda1"] = v
            elif rest == "ls2.gamma":
                hf_sd[li + "layer_scale2.lambda1"] = v
            elif rest == "attn.proj.weight":
                hf_sd[li + "attention.output.dense.weight"] = v
            elif rest == "attn.proj.bias":
                hf_sd[li + "attention.output.dense.bias"] = v
            elif rest == "mlp.fc1.weight":
                hf_sd[li + "mlp.fc1.weight"] = v
            elif rest == "mlp.fc1.bias":
                hf_sd[li + "mlp.fc1.bias"] = v
            elif rest == "mlp.fc2.weight":
                hf_sd[li + "mlp.fc2.weight"] = v
            elif rest == "mlp.fc2.bias":
                hf_sd[li + "mlp.fc2.bias"] = v
            elif rest == "attn.qkv.weight":
                # (2304, 768) -> q/k/v 各 (768,768)
                q, k, val = torch.chunk(v, 3, dim=0)
                hf_sd[li + "attention.attention.query.weight"] = q
                hf_sd[li + "attention.attention.key.weight"] = k
                hf_sd[li + "attention.attention.value.weight"] = val
            elif rest == "attn.qkv.bias":
                q, k, val = torch.chunk(v, 3, dim=0)
                hf_sd[li + "attention.attention.query.bias"] = q
                hf_sd[li + "attention.attention.key.bias"] = k
                hf_sd[li + "attention.attention.value.bias"] = val
            else:
                raise KeyError(f"unmapped block key: {k}")
        else:
            raise KeyError(f"unmapped key: {k}")
    return hf_sd


def main():
    if len(sys.argv) < 3:
        print("usage: convert_dinov2_timm2hf.py <timm_ckpt.pth> <out_hf.pt>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    timm_sd = ck["model"]
    hf_sd = convert(timm_sd)
    # submit_dinov2.WeatherDino 把 Dinov2Model 挂在 self.backbone, 需加前缀;
    # head 不加前缀 (自写 head 直接在顶层).
    out = {}
    for k, v in hf_sd.items():
        if k.startswith("head."):
            out[k] = v
        else:
            out["backbone." + k] = v
    torch.save(out, dst)
    print(f"converted {len(out)} keys -> {dst}")
    # 抽样打印
    for k in sorted(out)[:8]:
        print("  ", k, tuple(out[k].shape))
    print("  ...")
    for k in sorted(out)[-6:]:
        print("  ", k, tuple(out[k].shape))


if __name__ == "__main__":
    main()
