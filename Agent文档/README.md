# 天气分类任务文档（总索引）

> 竞赛：有限资源下优化模型训练及推理效率
> 目标：**F1 macro ≥ 0.95** 进国赛；4 类 `cloudy/rainy/snowy/sunny`（字母序 = 官方 label 顺序）
> 负责人：tuan

---

> 📌 **本文档是总览/导航**。流水账细节见 [实验记录.md](./实验记录.md)，训练技术方案见 [训练方案与改造.md](./训练方案与改造.md)，14→4 类映射见 [14类映射规则.md](./14类映射规则.md)。
> 本地工作区 `tasks_tmp/` 是服务器 `models/` + `tasks/` 的镜像（脚本/记录），权重与大数据集在服务器。

## 一、验收硬约束（决定整体方案）

| 约束 | 值 | 说明 |
|------|----|----|
| **训练** | s1 V100(32G×2) / 1docker 多卡 / H100 | b2 早期在 s1；DINOv2 系列在 1docker/H100 |
| **官方验收** | 2核8G **纯 CPU**，torch 2.3.1+cpu，**无 timm**，有 transformers 4.49.0 | 70 分钟内完成推理 |
| **指标** | **F1 macro**（非 accuracy） | 少数类 rainy/snowy 权重高 |
| **类别顺序** | `['cloudy','rainy','snowy','sunny']`（字母序） | 与官方 ImageFolder sorted 一致，提交零风险 |
| **predict 接口** | `predict(X)` 收 cv2 BGR ndarray `(224,224,3)` → 返回 label 字符串 | 提交只认此接口，不能改签名 |
| **提交物** | 提交包（`submit_dinov2.py` + 本地化权重 `dinov2_hf.pt`） | 官方无 timm，用 transformers.Dinov2Model 重写 |

## 二、当前状态（截至 G.19，已收尾）

- **当前最优权重**：s123（DINOv2-ViT-B/14，s123 管线）
  - **本地干净 test (741)**：F1 macro **0.9592**，Acc 0.9582
  - **官方 F1**：**0.9375**（迁移损失 ~0.022）
  - **5折 OOB 诚实度量**：均值 **0.9497**（静态 test 乐观 ~+0.01）
- **fold3 提交包**：已打包（`submit_fold3.tar.gz`，OOB F1 0.9628，cv2 路径无损复现）→ **用户决定不提交**（时间紧 + 单折方差风险）
- **距 0.95 国赛门槛**：本地已超（0.9592>0.95），但官方迁移 -0.02 后 0.9375，仍差 0.0125。**临界，未达线。**
- **路线已收尾**：本地调参杠杆穷尽，突破需靠缩小迁移损失（非提本地 F1）。

### 核心瓶颈与结论
- **sunny↔cloudy 是绝对瓶颈**（22 张互错占 71% 错误），属 224px 低分辨率本质可混淆 + 部分标签噪声。
- **rainy/snowy 已解决**（DINOv2 把 b2 的 0.64/0.71 拉到 0.945/0.975）。
- 数据集 Bayes 上限 ~0.97–0.98，剩 ~2 点是无解边界样本。

## 三、关键认知与教训（最值钱的沉淀）

> 详细推演见对话复盘；此处为索引。比任何 F1 数字都可复用。

1. **诚实度量须 OOB**：静态 test 泄露训练分布、乐观 ~+0.01；真实泛化看 5折 OOB。实测 0.9592 vs 0.9497。
2. **ensemble 有效性的唯一判据 = 错误独立性**，非模型数量。同 seed/同家族三次 ensemble 全失败（错误相关）；正确做法是异构 + 按类加权。
3. **边界分两类**：规则性（标签不一致，可统一规则处理）vs 本质性（特征真实重叠，硬扣分，靠加输入维度降）。
4. **补数据有效性判据**：判错 test 图的 train 近邻是否同类。同类=本质性（补了无效），异类=覆盖性（补了有效）。av3 失败正因没做此分析、盲补同分布图。
5. **不同 backbone 短板不同→边界不同**：b2 短在局部纹理→卡 rainy/snowy；DINOv2 短在全局布局→卡 sunny/cloudy。
6. **真实信号（官方）拿太少、噪声信号（本地 test）看太重**——决策配比倒挂是结构性错误。换 DINOv2 验证有效后，最该立刻全量训+官方测拿真实基准。
7. **领域约束**："不调对比度，否则 rainy 崩"——去颜色增强后 rainy 0.91→0.95，用户判断正确。
8. **不信 ckpt 元数据**：`architecture` 字段恒为 b2 不可靠，从 state_dict 推断 arch（`head.weight.shape[1]==1024 → dinov2_l, 否则 dinov2`）。

## 四、实验进度总表

> b2 系列（实验 0–F）为**历史路线**，天花板 ~0.77；DINOv2 系列（G.x）是**主线**，突破至 0.95+。

| 实验 | 方案 | 数据 | F1 macro | 状态 |
|------|------|------|:---:|:---:|
| 0 基线 | b2 冻结 head | dataset_augmented | Acc 85.54% | 历史 |
| A | b2 4类 渐进解冻 | dataset_augmented | Acc 89.29% | 历史 |
| B | 14类→4类 argmax | dataset_augmented_old | 95.75%(虚高) | ❌ 跨集0.51 |
| C | b2 v5 EMA/TTA | 224预resize | Acc 89.96% | 历史 |
| D | b2 v6 aug3 HD | dataset_aug3 | 95.42%(虚高) | ❌ 跨分布0.51 |
| E | b2 v6 split+Focal | dataset_split | 0.8422 | 历史(b2到顶) |
| F | b2 过拟合判据/E对照 | dataset_split | 本地0.766→官方0.754 | 判据成立 |
| **G.5** | **换 DINOv2-ViT-B/14** | dataset_split_av | **0.9397** | 🎯 +17点突破 |
| **G.7** | **80ep+纯几何增强** | dataset_split_av | **0.9537** | 🎯 破0.95 |
| **G.10 s123** | **seed=123** | dataset_split_av | **0.9546**(747)/0.9592(741) | 🥇 本地最优 |
| G.10 | 多seed/focal/高分辨率 消融 | — | ≤0.9546 | 噪声内无超 |
| G.12 | ViT-L/14 | dataset_split_av | 0.9491 | ❌ 过拟合,架构到顶 |
| G.13 | B+L ensemble | — | 0.9532 | ❌ 无增益 |
| G.14 | drop_path 0.1/0.2 | — | 0.9215/0.9150 | ❌ 负收益 |
| G.16 | test噪声清理+干净基线 | — | 0.9592(741) | ✅ 基线上调 |
| G.18 | av3 补数据重训 | dataset_split_av+av2 | 0.9584 | ❌ 补数据无收益 |
| **G.19** | **5折CV + fold3打包** | kfold | OOB均值0.9497/fold3 0.9628 | 收尾,未提交 |

> 官方 F1 仅 s123 测过一次（0.9375）。详情见 [实验记录.md](./实验记录.md)。

## 五、数据集对比

| 数据集 | 类别 | 用途 | 分布 | 状态 |
|--------|------|------|------|:---:|
| `dataset/` | 4类 平铺(224原版) | 验收同分布基准 | ✅ 验收同分布 | 基准 |
| `dataset_split/` | 4类 85/15 | b2 主训集 | ✅ 同分布 | 历史 |
| `dataset_split_av/` | 4类 + 补数据(pexels) | **DINOv2 主训集** | ✅ 同分布 | 主训 |
| `dataset_split_av_kfold/fold{0-4}/` | 5折CV | 诚实度量+fold3 | 互斥OOB | G.19 |
| `dataset_augmented/` | 4类增强 | 方案A | 同分布 | 备选 |
| `dataset_aug3/` | 4类 HD原图 | 方案D | ❌ HD≠验收 | ❌ 弃 |

**不均衡**：cloudy ~1941 / rainy ~484 / snowy ~414 / sunny ~1710（train），rainy/snowy 少数类曾是 F1 瓶颈（DINOv2 后已解决）。

## 六、checkpoint 索引（服务器 `models/checkpoints_*`）

| 目录 | 实验 | 架构 | 说明 |
|------|------|------|------|
| `checkpoints_4cls_v6_split/` | E | b2 | b2到顶 0.8422(历史) |
| `checkpoints_dinov2_av/` | G.5 | DINOv2-B | +17点 0.9397 |
| `checkpoints_dinov2_geom/` | G.7 | DINOv2-B | 破0.95 0.9537 |
| **`checkpoints_dinov2_s123/`** | **G.10** | **DINOv2-B** | **🥇 本地最优 0.9592** |
| `checkpoints_dinov2_vitl/` | G.12 | DINOv2-L | 过拟合 0.9491(弃) |
| `checkpoints_dinov2_av3/` | G.18 | DINOv2-B | 补数据 0.9584(无效) |
| **`checkpoints_dinov2_kfold/fold{0-4}/`** | **G.19** | **DINOv2-B** | **5折, fold3 OOB 0.9628** |

> s123 干净 test 0.9592 / 官方 0.9375。fold3 已转 transformers 权重并打包，未提交。

## 七、文件索引

```
EfficientNetV2-B2/                       # 服务器; tasks_tmp/ 为本地镜像
├── models/
│   ├── main_v6.py              # 主训脚本(DINOv2/b2分支, --geom-only, --select-f1, --drop-path等)
│   ├── eval_timm.py            # timm 模型 F1 评测
│   ├── eval_f1.py / eval_puretorch.py / eval_ensemble.py  # 各路评测
│   ├── audit_confidence.py / audit_confidence_v2.py  # ★置信度分档审计+判错图导出
│   ├── build_split.py / build_5fold.py  # 数据划分/5折CV构建(★OOB诚实度量)
│   ├── merge_av_to_split.py    # 补数据合并(预resize,仅入train)
│   ├── convert_dinov2_timm2hf.py  # ★timm→transformers 权重转换(qkv拆分/key映射)
│   ├── submit_dinov2.py        # ★提交主文件(predict接口, transformers.Dinov2Model+head)
│   ├── predict.py / submit_predict.py  # b2 时代 CPU 推理脚本(历史)
│   ├── effnetv2_b2_puretorch.py # b2 纯torch重写(官方无timm时用,历史)
│   └── checkpoints_*/          # 见上表
├── run_dinov2_*.sh             # DINOv2 训练启动脚本(s123/geom/kfold/vitl/reg等)
├── tasks/
│   ├── README.md               # 本文件(总览)
│   ├── 实验记录.md             # 流水账(0→G.19, 最详细)
│   ├── 训练方案与改造.md       # b2 技术方案(历史)
│   └── 14类映射规则.md         # 14→4映射(方案B已弃)
├── submit_fold3/               # ★G.19 fold3 提交包(submit_dinov2.py + dinov2_hf.pt)
├── hard_examples_s123/         # s123 判错图(按 真类_预测类)
└── errors_review/              # b2 误判样本导出(历史)
```

★ = 本次最有复用价值的产出（见第三节认知）。

## 八、提交物

- **s123 提交包**（已交官方，F1=0.9375）：`submit_dinov2.py` + `dinov2_hf.pt`，transformers 版，cv2 BGR→RGB→resize224+ImageNet normalize，无损复现 timm 0.9546。
- **fold3 提交包**（未提交）：`submit_fold3.tar.gz`（303MB，本地 `tasks_tmp/`），OOB 0.9628，cv2 路径无损复现。用户决定不赌单折方差，收尾。

## 九、决策记录

- [x] b2 路线天花板 ~0.77，**换 DINOv2-ViT-B/14**（+17 点，正确杠杆）
- [x] 80ep + 纯几何增强（去颜色变换）破 0.95（"不调对比度否则 rainy 崩"成立）
- [x] drop_path/TTA/高分辨率/补数据/同配置 ensemble **全负收益**，s123 管线是局部最优
- [x] ViT-L 架构到顶（数据量撑不住，B 更优）
- [x] 提交方案：timm→transformers 权重转换 + 无损验证，适配官方无 timm 环境
- [x] 5折 CV 给诚实度量（OOB 0.9497 vs 静态 0.9592）
- [x] fold3 打包完成，**不提交**，收尾

## 十、若继续：下一步方向（未执行）

1. **缩小迁移损失（主矛盾）**：补覆盖官方分布的真实数据；多拿官方 F1 反馈定位真实瓶颈。
2. **错误独立 ensemble（未验证假设）**：异构模型（CLIP 视觉端 / Swin 攻全局布局）补 DINOv2 的 sunny↔cloudy 短板，按类加权，OOB 验证。预期 +0.005~0.01。
3. **5折 OOB 调参（方法论修正）**：小量级消融用 OOB 均值±方差判断，静态 test 上 ±0.001 全是噪声。
4. **加输入维度降本质性边界**：时序/元数据/多模态，让 sunny↔cloudy 重叠区在新维度可分。
