#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv2 提交模型 (transformers 版, 官方环境无 timm 用此包).

设计:
- 官方环境: 纯 CPU, torch 2.3.1+cpu, 无 timm, 有 transformers 4.49.0.
- 用 transformers.Dinov2Model 作 backbone + 自写 nn.Linear 分类头.
- 权重: 把 timm 训练出的 checkpoint (vit_base_patch14_dinov2 + head) 转成 transformers 命名,
  作为包内常量序列化 (dinov2_hf.pt), 运行时本地加载, 不联网不下 hub.
- predict(X): cv2 BGR ndarray (H,W,3) → RGB → resize 224 → ImageNet normalize → forward → argmax → label 字符串.

类别顺序: ['cloudy','rainy','snowy','sunny'] (字母序, 与官方 ImageFolder 一致).
"""
import os
import numpy as np
import torch
import torch.nn as nn

# transformers 仅用于 backbone 结构 (Dinov2Model/Dinov2Config); 不联网加载权重.
from transformers import Dinov2Model, Dinov2Config

# ===== 路径与常量 =====
_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS = os.path.join(_HERE, "dinov2_hf.pt")  # 转换后的 transformers 命名权重
CLASSES = ["cloudy", "rainy", "snowy", "sunny"]  # 字母序, 与官方 ImageFolder sorted 一致
INPUT_SIZE = 224

# ImageNet 统计 (训练时所用)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class WeatherDino(nn.Module):
    """transformers.Dinov2Model backbone + 线性分类头.

    backbone forward 取 last_hidden_state 的 CLS token (index 0) → head → logits.
    与 timm vit_base_patch14_dinov2 forward_pooled(取 cls) 一致.
    """

    def __init__(self, num_classes=4):
        super().__init__()
        cfg = Dinov2Config(
            image_size=INPUT_SIZE,
            patch_size=14,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            use_mask_token=False,
            use_swiglu_ffn=False,
        )
        self.backbone = Dinov2Model(cfg)
        # 新版 transformers 移除了 add_pooling_layer; pooler 可能存在, 不影响取 CLS.
        try:
            # 若存在 pooler 且未实例化权重, 用 strict=False 忽略.
            pass
        except Exception:
            pass
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]  # CLS token
        return self.head(cls)


_model = None


def _load_model():
    global _model
    if _model is None:
        m = WeatherDino(num_classes=len(CLASSES))
        sd = torch.load(_WEIGHTS, map_location="cpu", weights_only=False)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        # 严格检查: 不应有 backbone 结构性缺失/多余 (只允许极少数无关字段)
        real_missing = [k for k in missing if not k.startswith("backbone.embeddings.mask_token")]
        if real_missing:
            raise RuntimeError(f"missing keys (non-trivial): {real_missing}")
        if unexpected:
            raise RuntimeError(f"unexpected keys: {unexpected}")
        m.eval()
        _model = m
    return _model


def _preprocess(X):
    """X: np.ndarray, BGR (cv2.imread) or RGB, (H,W,3) uint8/float → tensor (1,3,224,224) ImageNet norm."""
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    arr = np.asarray(X)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"predict expects (H,W,3) image, got {arr.shape}")
    # cv2 读出来是 BGR; 若已 RGB 也无害 (色调不影响结构), 但为与训练一致统一按 BGR→RGB 处理.
    # 训练数据来自 PIL (RGB), 官方 predict 收 cv2 BGR, 故需翻转.
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    rgb = arr[:, :, ::-1].copy()  # BGR -> RGB
    # resize 224 (cv2 INTER_LINEAR 与 PIL BILINEAR 接近; 训练用 PIL.Image.resize 默认 BILINEAR)
    try:
        import cv2
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image
        rgb = np.asarray(Image.fromarray(rgb).resize((INPUT_SIZE, INPUT_SIZE)))
    arr_f = rgb.astype(np.float32) / 255.0
    arr_f = (arr_f - IMAGENET_MEAN) / IMAGENET_STD
    arr_f = np.transpose(arr_f, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(arr_f)).unsqueeze(0)


def predict(X):
    """官方接口: 接收 np.ndarray (cv2 读取, (224,224,3)), 返回 label 字符串."""
    m = _load_model()
    x = _preprocess(X)
    with torch.no_grad():
        logits = m(x)
    idx = int(logits.argmax(1).item())
    return CLASSES[idx]


# ===== 评测入口 (H100 验证用; 提交时官方不调用) =====
def _eval_cli():
    import argparse, glob, sys
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    m = _load_model()
    # 类别目录 (平铺或 test 子目录)
    entries = [e for e in glob.glob(os.path.join(args.data_dir, "*")) if os.path.isdir(e)]
    names = {os.path.basename(e) for e in entries}
    class_dirs = {}
    if names & {"train", "test", "val"}:
        sp = os.path.join(args.data_dir, "test")
        if not os.path.isdir(sp):
            sp = os.path.join(args.data_dir, "val")
        for c in glob.glob(os.path.join(sp, "*")):
            if os.path.isdir(c):
                class_dirs[os.path.basename(c)] = c
    else:
        for e in entries:
            cn = os.path.basename(e)
            sub = [s for s in glob.glob(os.path.join(e, "*")) if os.path.isdir(s)]
            sn = {os.path.basename(s) for s in sub}
            if "test" in sn:
                class_dirs[cn] = os.path.join(e, "test")
            elif sn & {"train", "val"}:
                class_dirs[cn] = os.path.join(e, "train")
            else:
                class_dirs[cn] = e

    y_true, y_pred = [], []
    EVAL = ["sunny", "cloudy", "rainy", "snowy"]
    for ci, c in enumerate(CLASSES):
        d = class_dirs.get(c)
        if not d:
            continue
        for f in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isdir(f):
                continue
            img = np.asarray(Image.open(f).convert("RGB"))
            # 注意: 训练图来自 PIL RGB, 这里直接喂 RGB (不翻转) 模拟训练分布.
            arr_f = img.astype(np.float32) / 255.0
            arr_f = (arr_f - IMAGENET_MEAN) / IMAGENET_STD
            arr_f = np.transpose(arr_f, (2, 0, 1))
            x = torch.from_numpy(np.ascontiguousarray(arr_f)).unsqueeze(0)
            with torch.no_grad():
                logits = m(x)
            y_true.append(ci)
            y_pred.append(int(logits.argmax(1).item()))
    yt = np.array(y_true); yp = np.array(y_pred)
    mp = np.array([EVAL.index(CLASSES[i]) for i in range(len(CLASSES))])
    yt2 = mp[yt]; yp2 = mp[yp]
    n = len(yt2)
    acc = (yt2 == yp2).mean()
    # F1 macro (不依赖 sklearn)
    f1s = []
    for c in range(len(EVAL)):
        tp = int(((yp2 == c) & (yt2 == c)).sum())
        fp = int(((yp2 == c) & (yt2 != c)).sum())
        fn = int(((yp2 != c) & (yt2 == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    print("n samples:", n)
    print("F1 macro:", round(float(np.mean(f1s)), 4))
    print("Acc:", round(float(acc), 4))
    print("per-class F1:", {EVAL[i]: round(f1s[i], 4) for i in range(len(EVAL))})
    # 混淆矩阵
    print("confusion (rows=true, cols=pred) order=%s" % EVAL)
    print("         " + "  ".join("%-8s" % c for c in EVAL))
    cm = np.zeros((len(EVAL), len(EVAL)), dtype=int)
    for t, p in zip(yt2, yp2):
        cm[t][p] += 1
    for i, row in enumerate(cm):
        print("  %-7s " % EVAL[i] + "  ".join("%-8d" % v for v in row))


if __name__ == "__main__":
    _eval_cli()
