from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmseg.apis import inference_model, init_model
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='MTD: DAPR-FRFN vs Naive correction vs PCGRC mechanism analysis.'
    )


    parser.add_argument(
        '--dapr-config',
        type=Path,
        default=ROOT
        / 'projects/industrial_defect_seg/configs/mtd/'
          'segformer_mit-b0_dapr_frfn_mtd_binary_20k.py'
    )

    parser.add_argument(
        '--dapr-checkpoint',
        type=Path,
        default=ROOT
        / 'work_dirs/ablation/mtd/'
          'segformer_b0_dapr_frfn_mtd_binary_20k_bs4/'
          'best_mIoU_iter_7000.pth'
    )


    parser.add_argument(
        '--naive-config',
        type=Path,
        default=ROOT
        / 'projects/industrial_defect_seg/configs/mtd/'
          'segformer_mit-b0_dapr_frfn_naive_correction_mtd_binary_20k_bs4.py'
    )

    parser.add_argument(
        '--naive-checkpoint',
        type=Path,
        default=ROOT
        / 'work_dirs/ablation/mtd/'
          'segformer_b0_dapr_frfn_naive_correction_mtd_binary_20k_bs4/'
          'best_mIoU_iter_9500.pth'
    )


    parser.add_argument(
        '--gated-config',
        type=Path,
        default=ROOT
        / 'projects/industrial_defect_seg/configs/mtd/'
          'segformer_mit-b0_dapr_frfn_pcgrc_g001_mtd_binary_20k_bs4.py'
    )

    parser.add_argument(
        '--gated-checkpoint',
        type=Path,
        default=ROOT
        / 'work_dirs/comparison/mtd/'
          'segformer_b0_dapr_frfn_pcgrc_g001_mtd_binary_20k_bs4/'
          'best_mIoU_iter_9500.pth'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT
        / 'paper_analysis/mechanism/naive_vs_pcgrc/mtd'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=6,
        help='Maximum candidates retained for each candidate category.'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )

    return parser.parse_args()


def require_file(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f'{name} does not exist:\n{path}'
        )
    return path


def load_cfg(config_path: Path) -> Config:
    cfg = Config.fromfile(str(config_path))

    custom_imports = cfg.get('custom_imports', None)

    if custom_imports:
        import_modules_from_strings(**custom_imports)

    return cfg


def build_model(
    config_path: Path,
    checkpoint_path: Path,
    device: str,
):
    cfg = load_cfg(config_path)

    model = init_model(
        cfg,
        str(checkpoint_path),
        device=device
    )

    model.eval()

    return model


def build_test_dataset(config_path: Path):
    cfg = load_cfg(config_path)

    dataset = DATASETS.build(
        cfg.test_dataloader.dataset
    )

    dataset.full_init()

    return dataset


def read_binary_gt(mask_path: Path) -> np.ndarray:
    gt = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED
    )

    if gt is None:
        raise RuntimeError(
            f'Cannot read annotation:\n{mask_path}'
        )

    if gt.ndim == 3:
        gt = gt[..., 0]


    return (gt > 0).astype(np.uint8)


@torch.inference_mode()
def predict_binary(
    model,
    image_path: Path,
    target_shape=None,
) -> np.ndarray:
    result = inference_model(
        model,
        str(image_path)
    )

    pred = (
        result.pred_sem_seg.data
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    if (
        target_shape is not None
        and pred.shape != tuple(target_shape)
    ):
        pred = cv2.resize(
            pred,
            (
                int(target_shape[1]),
                int(target_shape[0])
            ),
            interpolation=cv2.INTER_NEAREST
        )

    return pred


def binary_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    gt_fg = gt == 1
    pred_fg = pred == 1

    tp = int(
        np.logical_and(gt_fg, pred_fg).sum()
    )

    fp = int(
        np.logical_and(~gt_fg, pred_fg).sum()
    )

    fn = int(
        np.logical_and(gt_fg, ~pred_fg).sum()
    )

    tn = int(
        np.logical_and(~gt_fg, ~pred_fg).sum()
    )

    union = tp + fp + fn

    if union > 0:
        iou = tp / union
    else:
        iou = 1.0

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 1.0
    )

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
    }


def release_model(model):
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_base_records(dataset) -> List[dict]:
    records = []

    dataset_size = len(dataset)

    if dataset_size == 0:
        raise RuntimeError(
            'MTD test dataset contains 0 samples. '
            'Check data_root / data_prefix / test split.'
        )

    print(f'Dataset length from MMEngine: {dataset_size}')

    for index in range(dataset_size):
        data_info = dataset.get_data_info(index)

        if 'img_path' not in data_info:
            raise RuntimeError(
                f'img_path missing from data_info[{index}]: '
                f'{data_info.keys()}'
            )

        if 'seg_map_path' not in data_info:
            raise RuntimeError(
                f'seg_map_path missing from data_info[{index}]: '
                f'{data_info.keys()}'
            )

        image_path = Path(
            data_info['img_path']
        )

        mask_path = Path(
            data_info['seg_map_path']
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f'Test image not found:\n{image_path}'
            )

        if not mask_path.is_file():
            raise FileNotFoundError(
                f'Test mask not found:\n{mask_path}'
            )

        gt = read_binary_gt(
            mask_path
        )

        gt_pixels = int(
            (gt == 1).sum()
        )

        records.append({
            'image_id':
                image_path.stem,

            'image_path':
                str(image_path),

            'mask_path':
                str(mask_path),

            'gt_pixels':
                gt_pixels,

            'total_pixels':
                int(gt.size),

            'is_positive':
                int(gt_pixels > 0),
        })

    if not records:
        raise RuntimeError(
            'Failed to build any MTD test records.'
        )

    return records


def scan_one_model(
    model_name: str,
    config_path: Path,
    checkpoint_path: Path,
    records: List[dict],
    device: str,
):
    print('\n' + '=' * 70)
    print(f'Loading {model_name}')
    print(f'Config:     {config_path}')
    print(f'Checkpoint: {checkpoint_path}')
    print('=' * 70)

    model = build_model(
        config_path,
        checkpoint_path,
        device
    )

    output = {}

    for index, row in enumerate(
        records,
        start=1
    ):
        image_path = Path(
            row['image_path']
        )

        mask_path = Path(
            row['mask_path']
        )

        gt = read_binary_gt(
            mask_path
        )

        pred = predict_binary(
            model,
            image_path,
            target_shape=gt.shape
        )

        metrics = binary_metrics(
            gt,
            pred
        )

        output[row['image_id']] = metrics

        if (
            index % 50 == 0
            or index == len(records)
        ):
            print(
                f'{model_name}: '
                f'{index}/{len(records)}'
            )

    release_model(model)

    return output


def add_model_metrics(
    rows: List[dict],
    prefix: str,
    metrics_dict: Dict[str, Dict[str, float]],
):
    for row in rows:
        m = metrics_dict[
            row['image_id']
        ]

        row[f'{prefix}_iou'] = m['iou']
        row[f'{prefix}_precision'] = m[
            'precision'
        ]
        row[f'{prefix}_recall'] = m[
            'recall'
        ]

        row[f'{prefix}_tp'] = m['tp']
        row[f'{prefix}_fp'] = m['fp']
        row[f'{prefix}_fn'] = m['fn']
        row[f'{prefix}_tn'] = m['tn']


def global_defect_iou(
    df: pd.DataFrame,
    prefix: str,
):
    tp = float(
        df[f'{prefix}_tp'].sum()
    )

    fp = float(
        df[f'{prefix}_fp'].sum()
    )

    fn = float(
        df[f'{prefix}_fn'].sum()
    )

    denom = tp + fp + fn

    if denom == 0:
        return 1.0

    return tp / denom


def select_candidates(
    df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    pos = df[
        df['gt_pixels'] > 0
    ].copy()

    pos['naive_minus_dapr'] = (
        pos['naive_iou']
        - pos['dapr_iou']
    )

    pos['gated_minus_dapr'] = (
        pos['gated_iou']
        - pos['dapr_iou']
    )

    pos['gated_minus_naive'] = (
        pos['gated_iou']
        - pos['naive_iou']
    )

    selected = []


    group = pos[
        (pos['naive_minus_dapr'] <= -0.05)
        &
        (pos['gated_minus_naive'] >= 0.10)
    ].sort_values(
        'gated_minus_naive',
        ascending=False
    ).head(top_k)

    for _, row in group.iterrows():
        item = row.to_dict()
        item['reason'] = (
            'naive_damage_gated_recovery'
        )
        selected.append(item)


    group = pos[
        (pos['gated_minus_dapr'] >= 0.05)
        &
        (pos['gated_minus_naive'] >= 0.05)
    ].sort_values(
        'gated_minus_dapr',
        ascending=False
    ).head(top_k)

    for _, row in group.iterrows():
        item = row.to_dict()
        item['reason'] = (
            'gated_extra_gain'
        )
        selected.append(item)


    group = pos[
        pos['gated_minus_dapr'] <= -0.05
    ].sort_values(
        'gated_minus_dapr',
        ascending=True
    ).head(top_k)

    for _, row in group.iterrows():
        item = row.to_dict()
        item['reason'] = (
            'gated_regression'
        )
        selected.append(item)

    out = pd.DataFrame(
        selected
    )

    if not out.empty:
        out = out.drop_duplicates(
            subset='image_id',
            keep='first'
        )

    return out


def get_selected_predictions(
    name: str,
    config_path: Path,
    checkpoint_path: Path,
    candidates: pd.DataFrame,
    device: str,
):
    print(
        f'\nRendering selected predictions: '
        f'{name}'
    )

    model = build_model(
        config_path,
        checkpoint_path,
        device
    )

    predictions = {}

    for _, row in candidates.iterrows():
        image_path = Path(
            row['image_path']
        )

        gt = read_binary_gt(
            Path(row['mask_path'])
        )

        pred = predict_binary(
            model,
            image_path,
            target_shape=gt.shape
        )

        predictions[
            row['image_id']
        ] = pred

    release_model(model)

    return predictions


def error_map(
    gt: np.ndarray,
    pred: np.ndarray,
):
    """
    RGB error visualization:
      TP -> green
      FP -> red
      FN -> blue
      TN -> black
    """
    rgb = np.zeros(
        (
            gt.shape[0],
            gt.shape[1],
            3
        ),
        dtype=np.uint8
    )

    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)

    rgb[tp] = (0, 200, 0)
    rgb[fp] = (220, 0, 0)
    rgb[fn] = (0, 80, 230)

    return rgb


def save_binary_mask(
    path: Path,
    mask: np.ndarray,
):
    cv2.imwrite(
        str(path),
        (
            (mask > 0)
            .astype(np.uint8)
            * 255
        )
    )


def render_candidates(
    candidates: pd.DataFrame,
    dapr_preds,
    naive_preds,
    gated_preds,
    selected_root: Path,
    figure_root: Path,
):
    for _, row in candidates.iterrows():
        sample_id = row['image_id']

        image_path = Path(
            row['image_path']
        )

        mask_path = Path(
            row['mask_path']
        )

        gt = read_binary_gt(
            mask_path
        )

        image_bgr = cv2.imread(
            str(image_path)
        )

        if image_bgr is None:
            raise RuntimeError(
                f'Cannot read image: {image_path}'
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        dapr = dapr_preds[sample_id]
        naive = naive_preds[sample_id]
        gated = gated_preds[sample_id]

        sample_dir = (
            selected_root
            / sample_id
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        save_binary_mask(
            sample_dir / 'gt.png',
            gt
        )

        save_binary_mask(
            sample_dir / 'dapr.png',
            dapr
        )

        save_binary_mask(
            sample_dir / 'naive_correction.png',
            naive
        )

        save_binary_mask(
            sample_dir / 'pcgrc.png',
            gated
        )


        fig, axes = plt.subplots(
            1,
            5,
            figsize=(17, 4)
        )

        axes[0].imshow(
            image_rgb
        )
        axes[0].set_title('Input')

        axes[1].imshow(
            gt,
            vmin=0,
            vmax=1
        )
        axes[1].set_title('GT')

        axes[2].imshow(
            dapr,
            vmin=0,
            vmax=1
        )
        axes[2].set_title(
            'DAPR-FRFN\n'
            f"IoU={row['dapr_iou']:.3f}"
        )

        axes[3].imshow(
            naive,
            vmin=0,
            vmax=1
        )
        axes[3].set_title(
            'Naive correction\n'
            f"IoU={row['naive_iou']:.3f}"
        )

        axes[4].imshow(
            gated,
            vmin=0,
            vmax=1
        )
        axes[4].set_title(
            'PCGRC\n'
            f"IoU={row['gated_iou']:.3f}"
        )

        for ax in axes:
            ax.axis('off')

        fig.suptitle(
            f"{sample_id} | "
            f"{row['reason']}\n"
            f"Naive-DAPR="
            f"{100*row['naive_minus_dapr']:+.2f} pp, "
            f"Gated-Naive="
            f"{100*row['gated_minus_naive']:+.2f} pp"
        )

        fig.tight_layout()

        fig.savefig(
            figure_root
            / f'prediction__{row["reason"]}__{sample_id}.png',
            dpi=200,
            bbox_inches='tight'
        )

        plt.close(fig)


        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4)
        )

        axes[0].imshow(
            error_map(gt, dapr)
        )
        axes[0].set_title(
            'DAPR-FRFN error'
        )

        axes[1].imshow(
            error_map(gt, naive)
        )
        axes[1].set_title(
            'Naive correction error'
        )

        axes[2].imshow(
            error_map(gt, gated)
        )
        axes[2].set_title(
            'PCGRC error'
        )

        for ax in axes:
            ax.axis('off')

        fig.suptitle(
            'TP=green, FP=red, FN=blue'
        )

        fig.tight_layout()

        fig.savefig(
            figure_root
            / f'error__{row["reason"]}__{sample_id}.png',
            dpi=200,
            bbox_inches='tight'
        )

        plt.close(fig)


def build_summary(
    df: pd.DataFrame,
):
    dapr = global_defect_iou(
        df,
        'dapr'
    )

    naive = global_defect_iou(
        df,
        'naive'
    )

    gated = global_defect_iou(
        df,
        'gated'
    )

    pos = df[
        df['gt_pixels'] > 0
    ].copy()

    naive_delta = (
        pos['naive_iou']
        - pos['dapr_iou']
    )

    gated_delta = (
        pos['gated_iou']
        - pos['dapr_iou']
    )

    recovery = (
        pos['gated_iou']
        - pos['naive_iou']
    )

    eps = 1e-12

    improved = int(
        (recovery > eps).sum()
    )

    degraded = int(
        (recovery < -eps).sum()
    )

    equal = int(
        (recovery.abs() <= eps).sum()
    )

    expected = {
        'DAPR-FRFN': 75.64,
        'Naive correction': 72.51,
        'PCGRC': 77.53,
    }

    measured = {
        'DAPR-FRFN': 100 * dapr,
        'Naive correction': 100 * naive,
        'PCGRC': 100 * gated,
    }

    lines = [
        '===== MTD: Naive correction vs PCGRC =====',
        '',
        'Global Defect IoU:',
    ]

    for key in expected:
        diff = (
            measured[key]
            - expected[key]
        )

        lines.append(
            f'{key:<14}: '
            f'{measured[key]:.4f}% '
            f'(official {expected[key]:.2f}%, '
            f'diff {diff:+.4f} pp)'
        )

    lines.extend([
        '',
        f'Positive test samples: {len(pos)}',
        '',
        'PCGRC vs Naive correction on positive samples:',
        f'Improved / Degraded / Equal = '
        f'{improved} / {degraded} / {equal}',
        (
            'Mean Δ(Gated-Naive)   = '
            f'{100*recovery.mean():+.4f} pp'
        ),
        (
            'Median Δ(Gated-Naive) = '
            f'{100*recovery.median():+.4f} pp'
        ),
        '',
        (
            'Mean Δ(Naive-DAPR)    = '
            f'{100*naive_delta.mean():+.4f} pp'
        ),
        (
            'Median Δ(Naive-DAPR)  = '
            f'{100*naive_delta.median():+.4f} pp'
        ),
        '',
        (
            'Mean Δ(Gated-DAPR)    = '
            f'{100*gated_delta.mean():+.4f} pp'
        ),
        (
            'Median Δ(Gated-DAPR)  = '
            f'{100*gated_delta.median():+.4f} pp'
        ),
        '',
        'Interpretation target:',
        (
            'Naive correction may introduce destructive structural correction, '
            'whereas PCGRC selectively controls correction injection.'
        ),
    ])

    max_diff = max(
        abs(
            measured[k]
            - expected[k]
        )
        for k in expected
    )

    if max_diff > 0.50:
        lines.extend([
            '',
            'WARNING:',
            (
                'At least one global IoU differs from the official result '
                'by more than 0.50 pp. Do not interpret mechanism figures '
                'until config/checkpoint/test protocol is checked.'
            )
        ])

    return '\n'.join(lines)


def main():
    args = parse_args()

    register_all_modules(
        init_default_scope=True
    )

    files = {
        'DAPR config':
            args.dapr_config,
        'DAPR checkpoint':
            args.dapr_checkpoint,

        'Naive config':
            args.naive_config,
        'Naive checkpoint':
            args.naive_checkpoint,

        'Gated config':
            args.gated_config,
        'Gated checkpoint':
            args.gated_checkpoint,
    }

    for name, path in files.items():
        require_file(
            path,
            name
        )

    if (
        args.device.startswith('cuda')
        and not torch.cuda.is_available()
    ):
        print(
            'CUDA is unavailable. '
            'Falling back to CPU.'
        )
        args.device = 'cpu'

    metrics_dir = (
        args.output_dir
        / 'metrics'
    )

    selected_dir = (
        args.output_dir
        / 'selected'
    )

    figures_dir = (
        args.output_dir
        / 'figures'
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    selected_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f'Using device: {args.device}'
    )


    dataset = build_test_dataset(
        args.dapr_config
    )

    records = make_base_records(
        dataset
    )
    if len(records) == 0:
        raise RuntimeError(
            'records is empty before inference.'
        )

    print(
        'First test sample:'
    )
    print(
        f'  image = {records[0]["image_path"]}'
    )
    print(
        f'  mask  = {records[0]["mask_path"]}'
    )
    print(
        f'MTD test samples: {len(records)}'
    )

    print(
        'Positive samples: '
        f'{sum(r["is_positive"] for r in records)}'
    )


    dapr_metrics = scan_one_model(
        'DAPR-FRFN',
        args.dapr_config,
        args.dapr_checkpoint,
        records,
        args.device
    )

    add_model_metrics(
        records,
        'dapr',
        dapr_metrics
    )

    del dapr_metrics

    naive_metrics = scan_one_model(
        'Naive correction',
        args.naive_config,
        args.naive_checkpoint,
        records,
        args.device
    )

    add_model_metrics(
        records,
        'naive',
        naive_metrics
    )

    del naive_metrics

    gated_metrics = scan_one_model(
        'PCGRC',
        args.gated_config,
        args.gated_checkpoint,
        records,
        args.device
    )

    add_model_metrics(
        records,
        'gated',
        gated_metrics
    )

    del gated_metrics


    df = pd.DataFrame(
        records
    )
    required_columns = [
        'dapr_iou',
        'naive_iou',
        'gated_iou',
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            'Inference results are incomplete. '
            f'Missing columns: {missing_columns}\n'
            f'Available columns: {list(df.columns)}'
        )
    df['naive_minus_dapr'] = (
        df['naive_iou']
        - df['dapr_iou']
    )

    df['gated_minus_dapr'] = (
        df['gated_iou']
        - df['dapr_iou']
    )

    df['gated_minus_naive'] = (
        df['gated_iou']
        - df['naive_iou']
    )

    per_image_path = (
        metrics_dir
        / 'per_image_metrics.csv'
    )

    df.to_csv(
        per_image_path,
        index=False,
        encoding='utf-8-sig'
    )

    summary = build_summary(
        df
    )

    print('\n' + summary)

    (
        metrics_dir
        / 'summary.txt'
    ).write_text(
        summary + '\n',
        encoding='utf-8'
    )


    candidates = select_candidates(
        df,
        args.top_k
    )

    candidate_path = (
        metrics_dir
        / 'candidate_selection.csv'
    )

    candidates.to_csv(
        candidate_path,
        index=False,
        encoding='utf-8-sig'
    )

    print(
        f'\nSelected candidate count: '
        f'{len(candidates)}'
    )

    if candidates.empty:
        print(
            'No candidates satisfied current thresholds. '
            'Metrics were saved; thresholds can be relaxed later.'
        )
        return


    dapr_preds = get_selected_predictions(
        'DAPR-FRFN',
        args.dapr_config,
        args.dapr_checkpoint,
        candidates,
        args.device
    )

    naive_preds = get_selected_predictions(
        'Naive correction',
        args.naive_config,
        args.naive_checkpoint,
        candidates,
        args.device
    )

    gated_preds = get_selected_predictions(
        'PCGRC',
        args.gated_config,
        args.gated_checkpoint,
        candidates,
        args.device
    )

    render_candidates(
        candidates,
        dapr_preds,
        naive_preds,
        gated_preds,
        selected_dir,
        figures_dir
    )

    print('\n===== Finished =====')
    print(f'Per-image metrics: {per_image_path}')
    print(f'Summary:           {metrics_dir / "summary.txt"}')
    print(f'Candidates:        {candidate_path}')
    print(f'Figures:           {figures_dir}')


if __name__ == '__main__':
    main()
