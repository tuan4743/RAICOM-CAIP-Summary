# EfficientNetV2-B2 天气分类任务文档

> 竞赛：有限资源下优化模型训练及推理效率
> 目标：**精度越高越好**（F1 macro）；EfficientNet-V2-B2 精度封顶再考虑换模型
> 负责人：tuan｜工作目录：服务器 `s1:~/project/EfficientNetV2-B2`

---

> 📌 **本文档是总览/导航**。流水账细节见 [实验记录.md](./实验记录.md)，训练技术方案见 [训练方案与改造.md](./训练方案与改造.md)，14→4 类映射见 [14类映射规则.md](./14类映射规则.md)。

## 一、验收硬约束（决定整体方案）

| 约束 | 值 | 说明 |
|------|----|----|
| **训练** | V100（32G×2），无时间硬限 | 放开冲精度 |
| **验收** | 2核8G **纯 CPU** 推理 | **70 分钟内**完成 |
| **指标** | **F1 macro**（非 accuracy） | 少数类 rainy 权重高 |
| **验收类别顺序** | `['sunny','cloudy','rainy','snowy']` | 非 sorted 字母序，须 `--classes-override` 对齐 |
| **验收测试集** | 组委会独立持有，与 `dataset/`（224 原版）同分布 | **不来自** `dataset/`，无泄露；与 `dataset_aug3`（HD）**不同分布** |
| **提交物** | 训练权重 `best.pth` + CPU 推理脚本 | — |

- 训练阶段：V100 大 batch / 384 输入 / 渐进式解冻 / 长 epochs，追求 F1 macro 最大化。
- 推理阶段：单独的 CPU 推理脚本，流式读取，适配 2核8G，70 分钟内跑完。

## 二、当前状态（截至实验 F / H100 复现）

- **当前最佳（验收同分布）**：实验 E（v6 split + Focal）→ F1 macro **0.8422**，Acc 87.68%（`checkpoints_4cls_v6_split/best.pth`）
- **进行中**：实验 F — H100 上跑**验证实验**（判官方验收集是否与 dataset/ 同分布）+ E 对照(224+字母序)
  - **实验一 极致过拟合**：全关正则+全量 dataset_split/train 训到 train acc≈100%, 拿过拟合权重跑官方测试. F1高→同分布; F1暴跌→分布不同须重做泛化
  - **实验二 E 对照**：224+字母序正常泛化训练, 作过拟合 F1 的对照基线
  - 状态：首次启动卡 HF 超时（权重已缓存但 timm 仍发 HEAD）, 已 kill 待修复离线加载后重启
- **当前瓶颈**：rainy precision 仅 0.65（recall 0.94），模型过度判 rainy，把 cloudy/sunny 误判为 rainy
- **距目标 93%**：差约 9 点，全卡在 rainy precision
- 已排除的死路：
  - 方案 B（14类→4类 argmax）自身 test 95.75%，但跨集泛化到 `dataset/` 仅 F1 0.51 → **同分布虚高，不可提交**
  - 方案 D（aug3 HD）自身 test 95.42%，跨分布测 `dataset/` 仅 F1 0.5125 → **分布不一致，不可提交**
  - 实验 E.2（去 Focal 改 CE）：F1 0.8346，比 Focal 还低 0.008，rainy precision 反而更差 → **loss 调参无效**

### 关键认知（中段修正，多次踩坑得出）
1. 验收测试集 ≠ 任何训练 test 集，与 `dataset/` 同分布、与 aug3 不同分布
2. 比的是 **F1 macro**，不是 accuracy
3. 验收类别顺序是 `['sunny','cloudy','rainy','snowy']`
4. 实验 E.2 结论：cloudy↔rainy 在 224px 低对比度数据上**本质可混淆**，去 Focal 不解决——真正杠杆是**数据/分辨率/模型容量**，而非 loss 调参

## 三、实验进度总表

| 实验 | 方案 | 数据 | F1 macro / Acc | 状态 |
|------|------|------|:---:|:---:|
| 0 基线 | 冻结 head | dataset_augmented (4类, 224) | Acc 85.54% | 旧 |
| A | 4类 渐进解冻 | dataset_augmented | Acc 89.29% | 完成 |
| B | 14类→4类 argmax | dataset_augmented_old | 95.75% (虚高) | ❌ 跨集仅0.51 |
| C | 4类 v5 EMA/TTA | (224 预resize) | Acc 89.96% | 完成 |
| D | 4类 v6 aug3 HD | dataset_aug3 | 95.42% (虚高) | ❌ 跨分布0.5125 |
| **E** | **v6 split + Focal** | **dataset_split (dataset/ 85/15)** | **F1 0.8422 / Acc 87.68%** | **当前最佳** |
| E.2 | v6 split + CE (去Focal) | dataset_split | F1 0.8346 | ❌ 更差 |
| F-过拟合 | 极致过拟合(全关正则) | dataset_split/train | 待官方F1判分布 | 🔄 H100 GPU0 |
| F-E对照 | E同款224+字母序 | dataset_split | 待测 | 🔄 H100 GPU1 |

> 详情见 [实验记录.md](./实验记录.md)。

## 四、数据集对比

| 数据集 | 类别 | 用途 | 分布 | 状态 |
|--------|------|------|------|:---:|
| `dataset/` | 4类 平铺(224原版) | 验收同分布基准 | ✅ 验收同分布 | 主攻 |
| `dataset_split/` | 4类 train4252/test747 | 当前主训集(85/15, seed42, 硬链接) | ✅ 同分布 | 主训 |
| `dataset_augmented/` | 4类 增强(train/test已划分) | 方案A | 同分布 | 备选 |
| `dataset_augmented_old/` | 14类 增强 | 方案B(映射回4类) | 14类均衡但跨分布虚高 | ❌ 弃 |
| `dataset_aug3/` | 4类 HD原图(不预resize) | 方案D | ❌ HD分布≠验收 | ❌ 弃 |
| `dataset_old/` | 14类 原版HD | aug3 来源 | — | 参考 |

**不均衡问题**：`dataset/` cloudy 2184 / rainy 446 / snowy 403 / sunny 1966（约 5:1），rainy/snowy 少数类是 F1 瓶颈。

## 五、checkpoint 索引

| 目录 | 对应实验 | best.pth | 说明 |
|------|----------|:---:|------|
| `checkpoints_v3/` | 0 基线 | ✓ | 85.54% 旧冻结版 |
| `checkpoints_4cls/` | A | ✓ | 4类 89.29% |
| `checkpoints_14cls/` | B | ✓ | 14类(虚高,弃) |
| `checkpoints_4cls_v5/` | C | ✓ | 4类 v5 89.96% |
| `checkpoints_4cls_v5_aug2/` | C变体 | ✓ | — |
| `checkpoints_4cls_v6_aug3/` | D | ✓ | aug3 HD(弃) |
| **`checkpoints_4cls_v6_split/`** | **E** | ✓ | **当前最佳 F1 0.8422** |
| `checkpoints_4cls_v6_split_ce/` | E.2 | ✓ | CE(更差) |

## 六、文件索引

```
project/EfficientNetV2-B2/
├── dataset/                  # 原始4类(224平铺) ← 验收同分布
├── dataset_split/            # 4类 85/15 划分 ← 当前主训集
├── dataset_augmented/        # 4类增强 ← 方案A
├── dataset_augmented_old/    # 14类增强 ← 方案B(弃)
├── dataset_aug3/             # 4类HD原图 ← 方案D(弃)
├── models/
│   ├── main_v6.py            # 当前主训脚本(RRC/RandAug/RE/EMA/TTA/classes-override)
│   ├── eval_f1.py            # F1 macro 评测(自适应类别顺序)
│   ├── predict.py            # CPU 推理脚本(流式)
│   ├── predict_cross.py      # 跨集泛化检验脚本
│   ├── main_v4.py / main_v5.py  # 历史训练脚本
│   └── checkpoints_*/        # 见上表
├── tasks/
│   ├── README.md             # 本文件(总览)
│   ├── 实验记录.md           # 流水账
│   ├── 训练方案与改造.md     # 技术方案
│   └── 14类映射规则.md       # 14→4映射
├── errors_review/            # 误判样本导出(按 真类→预测类)
├── review_cloudy_misrain/    # cloudy被误判rainy专项审查
└── run_v6_split*.sh          # 训练启动脚本
```

## 七、决策记录

- [x] 训练策略：渐进式解冻
- [x] 数据集：4类主攻(`dataset_split`)，14类方案弃(跨分布虚高)
- [x] 训练在 V100，验收在 2核8G CPU 推理(70分钟)
- [x] 14类→4类映射规则已定(见映射规则.md)，但方案B已弃
- [x] main.py 改造完成(main_v6.py)
- [x] CPU 推理脚本 predict.py 完成
- [x] 方案A(4类)与方案B(14类)对比完成
- [x] 确认验收同分布 = `dataset/`(224)，aug3/augmented_old 分布不符
- [x] 实验 E.2 确认：loss 调参(去Focal改CE)无效，瓶颈是数据/分辨率
- [ ] **下一步**：见下方"待决策"

## 八、下一步候选方向（待讨论）

实验 E.2 的结论是**真正杠杆在数据/分辨率/模型容量**，而非 loss 调参。候选：

1. **升输入分辨率**：224→384 已做(b2设计尺寸)，是否进一步用更大 crop / 更高清源
2. **rainy 硬负样本**：定向增强 cloudy↔rainy 边界样本
3. **模型容量**：b2 封顶前是否已榨干，或换 b3(需重估 CPU 70分钟约束)
4. **重新审视"本质可混淆"结论**：是否过早放弃 loss/阈值调节

> ⏸️ **暂不动手**：先讨论以上方向再定。
