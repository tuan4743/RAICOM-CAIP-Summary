#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
置信度审计 (难例挖掘 / hard example mining) —— 用纯 torch b2 模型, 零 timm.

对 dataset_split/test 全部四类做审计, 每张图输出完整 softmax 分布 + 置信度档位.
判错图自动复制到 hard_examples/<true>_<pred>/<filename>, 供人工看图找系统性错误模式.

用途: 给"手动补数据"提供精准方向 —— 不是泛泛补 rainy/snowy, 而是补模型系统性判错的子类型.
判据: 用 E 对照权重 (test F1 0.7662, 正常泛化), 暴露的弱点比过拟合版真实.

输出:
  1. 终端: 混淆矩阵 + 错误模式统计 + 各置信度档位计数 + top-N 高置信错误
  2. audit_<ckpt名>.csv: 每张图 path, true, pred, correct, max_prob, 各类 prob, 档位
  3. hard_examples/<true>_<pred>/<filename>: 判错图 (按错误方向分目录)

用法 (H100, efficientnet env):
  python audit_confidence.py --checkpoint checkpoints_e_224/best.pth --data-dir ../dataset_split
"""
import os, sys, argparse, glob, csv, shutil
import numpy as np, torch
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location('m6', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_v6.py'))
m6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m6)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 置信度档位 (基于预测类的 softmax 概率)
#   high   >=0.9  : 模型很确定 (无论对错)
#   mid    0.5~0.9: 边界模糊, 模型靠猜
#   low    <0.5   : 模型没主见 (max prob 不到一半, 几乎在乱猜)
def bucket(p):
    if p >= 0.9:
        return 'high'
    if p >= 0.5:
        return 'mid'
    return 'low'


def find_class_dirs(data_dir):
    """复用 eval_puretorch 同款结构探测, 保证一致.
    支持: {train,test}/{cls}/*  |  {cls}/{train,test}/*  |  {cls}/*(flat)
    优先取 test split.
    """
    entries = [e for e in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(e)]
    names = {os.path.basename(e) for e in entries}
    if names & {'train', 'test', 'val'}:
        for split in ('test', 'val', 'train'):
            sp = os.path.join(data_dir, split)
            if os.path.isdir(sp):
                return [(os.path.basename(c), c) for c in glob.glob(os.path.join(sp, '*')) if os.path.isdir(c)]
    out = []
    for e in entries:
        cn = os.path.basename(e)
        sub = [s for s in glob.glob(os.path.join(e, '*')) if os.path.isdir(s)]
        sname = {os.path.basename(s) for s in sub}
        if 'test' in sname:
            out.append((cn, os.path.join(e, 'test')))
        elif sname & {'train', 'val'}:
            out.append((cn, os.path.join(e, 'train')))
        else:
            out.append((cn, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--hard-dir', default='hard_examples',
                    help='判错图复制到此 (默认 hard_examples/)')
    ap.add_argument('--top', type=int, default=30, help='终端打印 top-N 高置信错误')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    sd = ck['model']
    classes = list(ck['classes'])   # 字母序 (checkpoint 内记录)
    print('ckpt classes:', classes)
    # 从 state_dict 推断架构 (ckpt 的 arch 字段不可靠, 恒为 b2)
    arch = 'dinov2_l' if (sd.get('head.weight') is not None and sd['head.weight'].shape[1] == 1024) else 'dinov2'
    isize = ck.get('input_size', 224)
    print('inferred arch:', arch, ' input_size:', isize)
    pm, _ = m6.build_model(arch, len(classes), pretrained=False)
    pm.load_state_dict(sd, strict=True)
    pm = pm.to(dev).eval()

    class_dirs = dict(find_class_dirs(args.data_dir))
    print('found class dirs:', {k: v for k, v in class_dirs.items()})

    # ===== 推理全部 test 样本, 收集完整分布 =====
    records = []   # (path, true_idx, pred_idx, max_prob, probs_vec)
    for ci, c in enumerate(classes):
        d = class_dirs.get(c)
        if not d:
            print('  WARN no dir for class', c)
            continue
        files = sorted(f for f in glob.glob(os.path.join(d, '*')) if not os.path.isdir(f))
        for f in files:
            img = Image.open(f).convert('RGB').resize((isize, isize))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            arr = np.transpose(arr, (2, 0, 1))
            x = torch.from_numpy(arr).unsqueeze(0).to(dev)
            with torch.no_grad():
                logits = pm(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())
            records.append((f, ci, pred, float(probs[pred]), probs))

    n = len(records)
    yt = np.array([r[1] for r in records])
    yp = np.array([r[2] for r in records])
    print('\n==================== 审计汇总 ====================')
    print('n samples:', n)
    print('Acc:', round(accuracy_score(yt, yp), 4))

    # 重排到验收顺序算 F1 (与 eval_puretorch 一致, 便于对照)
    EVAL = ['sunny', 'cloudy', 'rainy', 'snowy']
    m = np.array([EVAL.index(classes[i]) for i in range(len(classes))])
    yt_e = m[yt]; yp_e = m[yp]
    rep = classification_report(yt_e, yp_e, target_names=EVAL, zero_division=0, output_dict=True)
    print('F1 macro (验收顺序):', round(rep['macro avg']['f1-score'], 4))
    print(classification_report(yt_e, yp_e, target_names=EVAL, zero_division=0))

    # 混淆矩阵 (checkpoint 类别顺序)
    cm = confusion_matrix(yt, yp, labels=list(range(len(classes))))
    print('\nconfusion (rows=true, cols=pred)  classes=%s' % classes)
    hdr = '         ' + '  '.join('%-8s' % c for c in classes)
    print(hdr)
    for i, row in enumerate(cm):
        print('  %-7s ' % classes[i] + '  '.join('%-8d' % v for v in row))

    # ===== 错误模式统计 =====
    errors = [r for r in records if r[1] != r[2]]
    print('\n==================== 错误模式 ====================')
    print('total errors:', len(errors))
    from collections import Counter
    pattern = Counter((classes[r[1]], classes[r[2]]) for r in errors)
    print('error pattern (true->pred : count):')
    for (t, p), cnt in pattern.most_common():
        bar = '█' * cnt
        print('  %-8s -> %-8s : %2d  %s' % (t, p, cnt, bar))

    # ===== 置信度档位 (判对 vs 判错分别统计) =====
    print('\n==================== 置信度档位 ====================')
    print('档位定义: high>=0.9 | mid 0.5~0.9 | low<0.5  (max softmax prob)')
    print('  档位     判对    判错    判错率')
    for bk in ['high', 'mid', 'low']:
        in_b = [r for r in records if bucket(r[3]) == bk]
        corr = sum(1 for r in in_b if r[1] == r[2])
        wrong = len(in_b) - corr
        rate = wrong / len(in_b) if in_b else 0
        print('  %-6s   %4d   %4d    %.3f' % (bk, corr, wrong, rate))

    # 重点: mid/low 档的判对样本 = 边界模糊样本 (补数据优先级中)
    #       所有判错样本 = 硬负样本 (补数据优先级高)
    print('\n  边界模糊判对 (mid档判对): %d 张' % sum(
        1 for r in records if bucket(r[3]) == 'mid' and r[1] == r[2]))
    print('  硬负样本 (判错): %d 张' % len(errors))

    # ===== top-N 高置信错误 (模型很确信却错了 = 最该补) =====
    print('\n==================== top-%d 高置信错误 ====================' % args.top)
    sorted_err = sorted(errors, key=lambda r: -r[3])
    for r in sorted_err[:args.top]:
        probs_str = '  '.join('%s=%.2f' % (classes[i], r[4][i]) for i in range(len(classes)))
        print('  %s->%s  conf=%.2f | %s | %s' % (
            classes[r[1]], classes[r[2]], r[3], probs_str, os.path.basename(r[0])))

    # ===== 写 CSV (每张图完整分布) =====
    ckname = os.path.splitext(os.path.basename(args.checkpoint))[0]
    csv_path = 'audit_%s.csv' % ckname
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['path', 'true', 'pred', 'correct', 'max_prob', 'bucket']
                   + ['prob_%s' % c for c in classes])
        for r in records:
            w.writerow([r[0], classes[r[1]], classes[r[2]],
                        int(r[1] == r[2]), '%.4f' % r[3], bucket(r[3])]
                       + ['%.4f' % v for v in r[4]])
    print('\nCSV written:', csv_path, '(%d rows)' % n)

    # ===== 复制判错图到 hard_examples/<true>_<pred>/ =====
    hd = args.hard_dir
    if os.path.isdir(hd):
        shutil.rmtree(hd)
    for r in errors:
        subdir = os.path.join(hd, '%s_%s' % (classes[r[1]], classes[r[2]]))
        os.makedirs(subdir, exist_ok=True)
        try:
            shutil.copy2(r[0], os.path.join(subdir, os.path.basename(r[0])))
        except Exception as e:
            print('  copy fail %s: %s' % (r[0], e))
    print('判错图已复制到:', hd, '(%d 张, 按 <true>_<pred> 分目录)' % len(errors))
    print('\n人工看图建议: 先看错误数最多的 <true>_<pred> 目录, 找该子类型的共同视觉特征,')
    print('针对性补该子类型数据 (而非泛泛补整个类).')


if __name__ == '__main__':
    main()
