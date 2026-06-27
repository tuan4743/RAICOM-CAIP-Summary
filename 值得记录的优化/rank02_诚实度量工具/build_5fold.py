#!/usr/bin/env python3
"""
构建 5 折 CV 数据集: 合并 dataset_split_av 全量(train+test, 已删6张噪声+补122张av2),
按类分层 80/20 划 5 折, 每折生成结构A目录 dataset_split_av_kfold/fold{0-4}/<cls>/{train,test}/.
- 每折的 test 是该折独立验证集(模型训练时没见过), 用于诚实泛化指标.
- 5折并集 = 全量, 每张图恰好做1次test.
- 硬链接省空间, 失败回退复制. 文件名保留原名(含av2_前缀可追溯).
注意: 这是CV, 训练/验证仍同源(dataset_split_av), 诚实指标但分布gap对官方仍在.
"""
import os, shutil, random
from pathlib import Path
from PIL import Image

KFOLD = 5
SEED = 2026
CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
SRC_ROOT = Path("/gaojiawei/zya/project/EfficientNetV2-B2/dataset_split_av")
DST_ROOT = Path("/gaojiawei/zya/project/EfficientNetV2-B2/dataset_split_av_kfold")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

def main():
    random.seed(SEED)
    if DST_ROOT.exists():
        print(f"⚠️  {DST_ROOT} 已存在, 删除重建")
        shutil.rmtree(DST_ROOT)

    # 收集每类全量(去重按文件名)
    pool = {}
    for cls in CLASSES:
        files = []
        for sp in ["train", "test"]:
            d = SRC_ROOT / cls / sp
            if d.is_dir():
                for f in d.iterdir():
                    if f.suffix.lower() in IMG_EXT:
                        files.append(f)
        # 去重(同名可能在train和test都有, 不应发生, 防御)
        seen = set(); uniq = []
        for f in files:
            if f.name not in seen:
                seen.add(f.name); uniq.append(f)
        random.shuffle(uniq)
        pool[cls] = uniq
        print(f"  {cls}: 全量 {len(uniq)}")

    # 分层 K 折: 每类切成 K 份, 第 i 折 test = 各类第 i 份, train = 其余
    folds = [ {c: [] for c in CLASSES} for _ in range(KFOLD) ]
    for cls in CLASSES:
        items = pool[cls]
        k = len(items) // KFOLD
        for i in range(KFOLD):
            start = i * k
            end = (i + 1) * k if i < KFOLD - 1 else len(items)
            folds[i][cls] = items[start:end]

    # 写入结构A
    for fi in range(KFOLD):
        fold_dir = DST_ROOT / f"fold{fi}"
        test_names = {c: set(f.name for f in folds[fi][c]) for c in CLASSES}
        for cls in CLASSES:
            test_items = folds[fi][cls]
            train_items = [f for f in pool[cls] if f.name not in test_names[cls]]
            for split, items in [("train", train_items), ("test", test_items)]:
                dd = fold_dir / cls / split
                dd.mkdir(parents=True, exist_ok=True)
                n = 0
                for f in items:
                    try:
                        im = Image.open(f); im.verify()
                    except Exception:
                        continue
                    out = dd / f.name
                    try:
                        os.link(f, out)
                    except OSError:
                        shutil.copy(f, out)
                    n += 1
                if split == "test":
                    print(f"  fold{fi} {cls}/test={n} train={len(train_items)}")
        print(f"✅ fold{fi} done")

if __name__ == "__main__":
    main()
