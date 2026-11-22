from __future__ import annotations

import argparse
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
        description=(
            'KSDD2 DAPR internal visualization: '
            'channel statistics and adaptive routing weights.'
        )
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=ROOT
        / 'projects/industrial_defect_seg/configs/ksdd2/'
          'segformer_mit-b0_dapr_frfn_ksdd2_20k.py'
    )

    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=ROOT
        / 'work_dirs/ablation/ksdd2/'
          'segformer_b0_dapr_frfn_ksdd2_20k/'
          'best_mIoU_iter_18000.pth'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT
        / 'paper_analysis/mechanism/dapr/ksdd2'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )

    parser.add_argument(
        '--heatmap-samples',
        type=int,
        default=60,
        help='Maximum number of positive samples shown in the heatmap.'
    )

    return parser.parse_args()


def require_file(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(
            f'{name} does not exist:\n{path}'
        )


def load_cfg(path: Path):
    cfg = Config.fromfile(
        str(path)
    )

    custom_imports = cfg.get(
        'custom_imports',
        None
    )

    if custom_imports:
        import_modules_from_strings(
            **custom_imports
        )

    return cfg


def build_model(
    config_path: Path,
    checkpoint_path: Path,
    device: str,
):
    cfg = load_cfg(
        config_path
    )

    model = init_model(
        cfg,
        str(checkpoint_path),
        device=device
    )

    model.eval()

    return model


def build_dataset(
    config_path: Path,
):
    cfg = load_cfg(
        config_path
    )

    dataset = DATASETS.build(
        cfg.test_dataloader.dataset
    )

    dataset.full_init()

    return dataset


def read_binary_gt(
    mask_path: Path,
):
    gt = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED
    )

    if gt is None:
        raise RuntimeError(
            f'Cannot read GT:\n{mask_path}'
        )

    if gt.ndim == 3:
        gt = gt[..., 0]

    return (
        (gt > 0)
        .astype(np.uint8)
    )


class DAPRInternalCollector:
    """
    Exact hooks based on the real module tree:

        frfn_refine.frfn
        frfn_refine.frfn.channel_router

    We do NOT modify model source code.

    X:
        actual input tensor entering DAPRFRFN

    channel statistics:
        spatial mean/std computed directly from X

    route weights:
        direct output of channel_router, whose final layer is Sigmoid
    """

    def __init__(self, decode_head):

        named_modules = dict(
            decode_head.named_modules()
        )

        self.frfn_name = (
            'frfn_refine.frfn'
        )

        self.router_name = (
            'frfn_refine.frfn.channel_router'
        )

        if (
            self.frfn_name
            not in named_modules
        ):
            raise RuntimeError(
                f'Cannot find module: '
                f'{self.frfn_name}'
            )

        if (
            self.router_name
            not in named_modules
        ):
            raise RuntimeError(
                f'Cannot find module: '
                f'{self.router_name}'
            )

        self.frfn = named_modules[
            self.frfn_name
        ]

        self.router = named_modules[
            self.router_name
        ]

        self.current: Dict[
            str,
            np.ndarray
        ] = {}

        self.frfn_handle = (
            self.frfn
            .register_forward_pre_hook(
                self._frfn_pre_hook
            )
        )

        self.router_handle = (
            self.router
            .register_forward_hook(
                self._router_hook
            )
        )

    def clear(self):
        self.current = {}

    def close(self):
        self.frfn_handle.remove()
        self.router_handle.remove()

    def _frfn_pre_hook(
            self,
            module,
            inputs,
    ):
        del module

        if not inputs:
            return

        x = inputs[0]

        if not isinstance(
                x,
                torch.Tensor
        ):
            raise RuntimeError(
                'DAPRFRFN first input is not a Tensor.'
            )

        x = x.detach()


        if x.ndim == 3:

            B, N, C = x.shape


            mean = x.mean(
                dim=1
            )

            std = x.std(
                dim=1,
                unbiased=False
            )

            self.current[
                'feature_shape'
            ] = np.asarray(
                x.shape,
                dtype=np.int64
            )

            self.current[
                'token_count'
            ] = np.asarray(
                [N],
                dtype=np.int64
            )

            self.current[
                'channels'
            ] = np.asarray(
                [C],
                dtype=np.int64
            )


        elif x.ndim == 4:

            B, C, H, W = x.shape

            mean = x.mean(
                dim=(2, 3)
            )

            std = x.std(
                dim=(2, 3),
                unbiased=False
            )

            self.current[
                'feature_shape'
            ] = np.asarray(
                x.shape,
                dtype=np.int64
            )

            self.current[
                'token_count'
            ] = np.asarray(
                [H * W],
                dtype=np.int64
            )

            self.current[
                'channels'
            ] = np.asarray(
                [C],
                dtype=np.int64
            )

        else:
            raise RuntimeError(
                'Unexpected DAPRFRFN input shape: '
                f'{tuple(x.shape)}. '
                'Expected [B,N,C] or [B,C,H,W].'
            )

        self.current['mean'] = (
            mean
            .cpu()
            .numpy()
        )

        self.current['std'] = (
            std
            .cpu()
            .numpy()
        )

        self.current['std'] = (
            std
            .cpu()
            .numpy()
        )

    def _router_hook(
        self,
        module,
        inputs,
        output,
    ):
        del module, inputs

        if not isinstance(
            output,
            torch.Tensor
        ):
            raise RuntimeError(
                'channel_router output '
                'is not a Tensor.'
            )

        weight = (
            output
            .detach()
        )


        if weight.ndim == 4:
            if (
                weight.shape[2] != 1
                or weight.shape[3] != 1
            ):
                raise RuntimeError(
                    'Unexpected router output: '
                    f'{tuple(weight.shape)}'
                )

            weight = (
                weight[:, :, 0, 0]
            )

        if weight.ndim != 2:
            raise RuntimeError(
                'Expected routing weight '
                f'[B,C], got {tuple(weight.shape)}'
            )


        minimum = float(
            weight.min()
        )

        maximum = float(
            weight.max()
        )

        if (
            minimum < -1e-5
            or maximum > 1.0 + 1e-5
        ):
            raise RuntimeError(
                'Router output is outside '
                '[0,1], which is inconsistent '
                'with the confirmed Sigmoid output.\n'
                f'min={minimum}, max={maximum}'
            )

        self.current['weight'] = (
            weight
            .cpu()
            .numpy()
        )


def pearson(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64
    ).reshape(-1)

    y = np.asarray(
        y,
        dtype=np.float64
    ).reshape(-1)

    if (
        x.size == 0
        or y.size == 0
        or x.std() == 0
        or y.std() == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


def spearman(
    x,
    y,
):
    x = pd.Series(
        np.asarray(x).reshape(-1)
    ).rank()

    y = pd.Series(
        np.asarray(y).reshape(-1)
    ).rank()

    return float(
        x.corr(y)
    )


def cosine_similarity_matrix(
    x: np.ndarray,
):
    norm = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

    normalized = (
        x
        / (norm + 1e-12)
    )

    return (
        normalized
        @ normalized.T
    )


def assign_scale_groups(
    gt_pixels: np.ndarray,
):
    """
    Positive KSDD2 samples only.

    Q1 / Q2 follow the previous positive-sample
    scale-analysis convention:
       SMALL  <= Q1
       MEDIUM <= Q2
       LARGE  > Q2
    """

    q1 = float(
        np.quantile(
            gt_pixels,
            1.0 / 3.0
        )
    )

    q2 = float(
        np.quantile(
            gt_pixels,
            2.0 / 3.0
        )
    )

    groups = []

    for value in gt_pixels:
        if value <= q1:
            groups.append(
                'SMALL'
            )

        elif value <= q2:
            groups.append(
                'MEDIUM'
            )

        else:
            groups.append(
                'LARGE'
            )

    return (
        np.asarray(groups),
        q1,
        q2
    )


def plot_heatmap(
    weights,
    gt_pixels,
    output_path: Path,
    max_samples: int,
):
    order = np.argsort(
        gt_pixels
    )

    if (
        len(order)
        > max_samples
    ):
        indices = np.linspace(
            0,
            len(order) - 1,
            max_samples
        )

        indices = np.round(
            indices
        ).astype(int)

        order = order[
            indices
        ]

    matrix = weights[
        order
    ]

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    im = ax.imshow(
        matrix,
        aspect='auto',
        interpolation='nearest'
    )

    ax.set_xlabel(
        'Channel index'
    )

    ax.set_ylabel(
        'Positive samples '
        '(sorted by GT defect area)'
    )

    ax.set_title(
        'DAPR routing weights across KSDD2 samples'
    )

    fig.colorbar(
        im,
        ax=ax,
        label='Routing weight'
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight'
    )

    plt.close(fig)


def choose_representatives(
    gt_pixels,
):
    """
    Representative samples near the median
    of each equal-frequency scale group.
    """

    order = np.argsort(
        gt_pixels
    )

    thirds = np.array_split(
        order,
        3
    )

    selected = []

    for group in thirds:
        selected.append(
            int(
                group[
                    len(group) // 2
                ]
            )
        )

    return selected


def plot_profiles(
    weights,
    gt_pixels,
    sample_ids,
    output_path: Path,
):
    indices = (
        choose_representatives(
            gt_pixels
        )
    )

    labels = [
        'Small',
        'Medium',
        'Large',
    ]

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    x = np.arange(
        weights.shape[1]
    )

    for idx, label in zip(
        indices,
        labels
    ):
        ax.plot(
            x,
            weights[idx],
            linewidth=1.2,
            label=(
                f'{label}: '
                f'{sample_ids[idx]} '
                f'(GT={gt_pixels[idx]})'
            )
        )

    ax.set_xlabel(
        'Channel index'
    )

    ax.set_ylabel(
        'Routing weight'
    )

    ax.set_title(
        'Representative sample-specific DAPR routing profiles'
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight'
    )

    plt.close(fig)


def plot_relation(
    x,
    weight,
    xlabel,
    output_path: Path,
):
    p = pearson(
        x,
        weight
    )

    s = spearman(
        x,
        weight
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.scatter(
        np.asarray(
            x
        ).reshape(-1),
        np.asarray(
            weight
        ).reshape(-1),
        s=6,
        alpha=0.15
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        'Routing weight'
    )

    ax.set_title(
        f'{xlabel} vs routing weight\n'
        f'Pearson={p:.3f}, '
        f'Spearman={s:.3f}'
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight'
    )

    plt.close(fig)

    return p, s


def plot_cosine(
    weights,
    gt_pixels,
    output_path: Path,
):
    order = np.argsort(
        gt_pixels
    )

    ordered = weights[
        order
    ]

    similarity = (
        cosine_similarity_matrix(
            ordered
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    im = ax.imshow(
        similarity,
        aspect='auto',
        vmin=0,
        vmax=1,
        interpolation='nearest'
    )

    ax.set_xlabel(
        'Positive sample index '
        '(sorted by defect area)'
    )

    ax.set_ylabel(
        'Positive sample index '
        '(sorted by defect area)'
    )

    ax.set_title(
        'Pairwise cosine similarity of DAPR routing profiles'
    )

    fig.colorbar(
        im,
        ax=ax,
        label='Cosine similarity'
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight'
    )

    plt.close(fig)

    return similarity


def plot_channel_variation(
    weights,
    output_path: Path,
):
    variation = weights.std(
        axis=0
    )

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        np.arange(
            len(variation)
        ),
        variation
    )

    ax.set_xlabel(
        'Channel index'
    )

    ax.set_ylabel(
        'Std of routing weight across samples'
    )

    ax.set_title(
        'Channel-wise sample variation of DAPR routing weights'
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight'
    )

    plt.close(fig)


def plot_group_mean_profiles(
    weights,
    groups,
    output_path: Path,
):
    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    x = np.arange(
        weights.shape[1]
    )

    for group_name in (
        'SMALL',
        'MEDIUM',
        'LARGE',
    ):
        subset = weights[
            groups == group_name
        ]

        profile = subset.mean(
            axis=0
        )

        ax.plot(
            x,
            profile,
            linewidth=1.2,
            label=(
                f'{group_name} '
                f'(n={len(subset)})'
            )
        )

    ax.set_xlabel(
        'Channel index'
    )

    ax.set_ylabel(
        'Mean routing weight'
    )

    ax.set_title(
        'Mean DAPR routing profiles by defect-size group'
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight'
    )

    plt.close(fig)


def main():
    args = parse_args()

    require_file(
        args.config,
        'Config'
    )

    require_file(
        args.checkpoint,
        'Checkpoint'
    )

    register_all_modules(
        init_default_scope=True
    )

    if (
        args.device.startswith(
            'cuda'
        )
        and not torch.cuda.is_available()
    ):
        print(
            'CUDA unavailable; '
            'falling back to CPU.'
        )

        args.device = 'cpu'

    raw_dir = (
        args.output_dir
        / 'raw'
    )

    csv_dir = (
        args.output_dir
        / 'csv'
    )

    figure_dir = (
        args.output_dir
        / 'figures'
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f'Using device: {args.device}'
    )

    print(
        f'Config:\n{args.config}'
    )

    print(
        f'Checkpoint:\n{args.checkpoint}'
    )

    model = build_model(
        args.config,
        args.checkpoint,
        args.device
    )

    dataset = build_dataset(
        args.config
    )

    dataset_size = len(
        dataset
    )

    print(
        f'\nKSDD2 test samples: '
        f'{dataset_size}'
    )

    if dataset_size == 0:
        raise RuntimeError(
            'KSDD2 test dataset is empty.'
        )


    collector = (
        DAPRInternalCollector(
            model.decode_head
        )
    )

    print(
        '\nExact hooks:'
    )

    print(
        '  feature input : '
        'frfn_refine.frfn'
    )

    print(
        '  route weights : '
        'frfn_refine.frfn.channel_router'
    )


    positive_infos = []

    for dataset_index in range(
        dataset_size
    ):
        data_info = (
            dataset.get_data_info(
                dataset_index
            )
        )

        image_path = Path(
            data_info[
                'img_path'
            ]
        )

        mask_path = Path(
            data_info[
                'seg_map_path'
            ]
        )

        gt = read_binary_gt(
            mask_path
        )

        gt_pixels = int(
            (gt == 1).sum()
        )

        if gt_pixels <= 0:
            continue

        positive_infos.append({
            'image_path':
                image_path,

            'mask_path':
                mask_path,

            'sample_id':
                image_path.stem,

            'gt_pixels':
                gt_pixels,
        })

    print(
        'Positive samples: '
        f'{len(positive_infos)}'
    )

    if len(positive_infos) != 110:
        print(
            'WARNING: expected 110 positive '
            'KSDD2 test samples from the '
            'established split.'
        )


    first = positive_infos[0]

    collector.clear()

    print(
        '\nRunning one-sample dry-run...'
    )

    with torch.inference_mode():
        _ = inference_model(
            model,
            str(
                first[
                    'image_path'
                ]
            )
        )

    for required in (
        'mean',
        'std',
        'weight',
    ):
        if (
            required
            not in collector.current
        ):
            collector.close()

            raise RuntimeError(
                f'Dry-run failed: '
                f'{required} not captured.'
            )

    dry_mean = (
        collector.current[
            'mean'
        ]
    )

    dry_std = (
        collector.current[
            'std'
        ]
    )

    dry_weight = (
        collector.current[
            'weight'
        ]
    )

    print(
        'Dry-run captured:'
    )

    print(
        f'  DAPR input shape = '
        f'{tuple(collector.current["feature_shape"])}'
    )

    print(
        f'  Mean   {dry_mean.shape}'
    )

    print(
        f'  Std    {dry_std.shape}'
    )

    print(
        f'  Weight {dry_weight.shape}'
    )

    print(
        '  Weight range: '
        f'[{dry_weight.min():.6f}, '
        f'{dry_weight.max():.6f}]'
    )

    print(
        f'  Mean   {dry_mean.shape}'
    )

    print(
        f'  Std    {dry_std.shape}'
    )

    print(
        f'  Weight {dry_weight.shape}'
    )

    print(
        '  Weight range: '
        f'[{dry_weight.min():.6f}, '
        f'{dry_weight.max():.6f}]'
    )

    if (
        dry_mean.shape
        != dry_std.shape
        or dry_mean.shape
        != dry_weight.shape
    ):
        collector.close()

        raise RuntimeError(
            'Dry-run shape mismatch.'
        )

    channels = int(
        dry_weight.shape[1]
    )

    print(
        f'  Channels C = {channels}'
    )


    sample_ids: List[str] = []
    gt_pixels_list: List[int] = []

    means = []
    stds = []
    weights = []

    try:
        for index, info in enumerate(
            positive_infos,
            start=1
        ):
            collector.clear()

            with torch.inference_mode():
                _ = inference_model(
                    model,
                    str(
                        info[
                            'image_path'
                        ]
                    )
                )

            missing = [
                key
                for key in (
                    'mean',
                    'std',
                    'weight',
                )
                if (
                    key
                    not in collector.current
                )
            ]

            if missing:
                raise RuntimeError(
                    f'Missing internal tensors '
                    f'for {info["sample_id"]}: '
                    f'{missing}'
                )

            mean = (
                collector.current[
                    'mean'
                ][0]
            )

            std = (
                collector.current[
                    'std'
                ][0]
            )

            weight = (
                collector.current[
                    'weight'
                ][0]
            )

            if (
                len(mean) != channels
                or len(std) != channels
                or len(weight) != channels
            ):
                raise RuntimeError(
                    f'Unexpected channel count '
                    f'for {info["sample_id"]}.'
                )

            sample_ids.append(
                info[
                    'sample_id'
                ]
            )

            gt_pixels_list.append(
                info[
                    'gt_pixels'
                ]
            )

            means.append(
                mean
            )

            stds.append(
                std
            )

            weights.append(
                weight
            )

            if (
                index % 20 == 0
                or index
                == len(
                    positive_infos
                )
            ):
                print(
                    f'Collected '
                    f'{index}/'
                    f'{len(positive_infos)}'
                )

    finally:
        collector.close()


    sample_ids = np.asarray(
        sample_ids
    )

    gt_pixels = np.asarray(
        gt_pixels_list,
        dtype=np.int64
    )

    means = np.asarray(
        means,
        dtype=np.float32
    )

    stds = np.asarray(
        stds,
        dtype=np.float32
    )

    weights = np.asarray(
        weights,
        dtype=np.float32
    )

    print(
        '\n===== Final shapes ====='
    )

    print(
        f'Mean:    {means.shape}'
    )

    print(
        f'Std:     {stds.shape}'
    )

    print(
        f'Weights: {weights.shape}'
    )

    expected_shape = (
        len(
            positive_infos
        ),
        channels
    )

    if (
        means.shape
        != expected_shape
        or stds.shape
        != expected_shape
        or weights.shape
        != expected_shape
    ):
        raise RuntimeError(
            'Final captured tensor shape '
            'is inconsistent.'
        )


    (
        groups,
        q1,
        q2
    ) = assign_scale_groups(
        gt_pixels
    )


    np.savez_compressed(
        raw_dir
        / 'dapr_internal_features.npz',

        sample_ids=
            sample_ids,

        gt_pixels=
            gt_pixels,

        groups=
            groups,

        means=
            means,

        stds=
            stds,

        weights=
            weights,
    )


    sample_rows = []

    for i in range(
        len(sample_ids)
    ):
        sample_rows.append({
            'sample_id':
                sample_ids[i],

            'gt_pixels':
                int(
                    gt_pixels[i]
                ),

            'scale_group':
                groups[i],

            'channel_mean_avg':
                float(
                    means[i].mean()
                ),

            'channel_mean_std':
                float(
                    means[i].std()
                ),

            'channel_spatial_std_avg':
                float(
                    stds[i].mean()
                ),

            'channel_spatial_std_std':
                float(
                    stds[i].std()
                ),

            'route_weight_mean':
                float(
                    weights[i].mean()
                ),

            'route_weight_std':
                float(
                    weights[i].std()
                ),

            'route_weight_min':
                float(
                    weights[i].min()
                ),

            'route_weight_max':
                float(
                    weights[i].max()
                ),
        })

    pd.DataFrame(
        sample_rows
    ).to_csv(
        csv_dir
        / 'sample_level_stats.csv',
        index=False,
        encoding='utf-8-sig'
    )


    channel_rows = []

    channel_weight_std = (
        weights.std(
            axis=0
        )
    )

    for channel in range(
        channels
    ):
        channel_rows.append({
            'channel':
                channel,

            'mean_activation_mean':
                float(
                    means[
                        :,
                        channel
                    ].mean()
                ),

            'mean_activation_std':
                float(
                    means[
                        :,
                        channel
                    ].std()
                ),

            'spatial_std_mean':
                float(
                    stds[
                        :,
                        channel
                    ].mean()
                ),

            'spatial_std_std':
                float(
                    stds[
                        :,
                        channel
                    ].std()
                ),

            'route_weight_mean':
                float(
                    weights[
                        :,
                        channel
                    ].mean()
                ),

            'route_weight_std':
                float(
                    channel_weight_std[
                        channel
                    ]
                ),

            'route_weight_min':
                float(
                    weights[
                        :,
                        channel
                    ].min()
                ),

            'route_weight_max':
                float(
                    weights[
                        :,
                        channel
                    ].max()
                ),
        })

    pd.DataFrame(
        channel_rows
    ).to_csv(
        csv_dir
        / 'channel_level_stats.csv',
        index=False,
        encoding='utf-8-sig'
    )


    group_rows = []

    for group_name in (
        'SMALL',
        'MEDIUM',
        'LARGE',
    ):
        subset = weights[
            groups == group_name
        ]

        group_rows.append({
            'group':
                group_name,

            'n':
                int(
                    len(subset)
                ),

            'mean_route_weight':
                float(
                    subset.mean()
                ),

            'std_route_weight':
                float(
                    subset.std()
                ),

            'mean_within_sample_weight_std':
                float(
                    subset.std(
                        axis=1
                    ).mean()
                ),
        })

    pd.DataFrame(
        group_rows
    ).to_csv(
        csv_dir
        / 'scale_group_stats.csv',
        index=False,
        encoding='utf-8-sig'
    )


    plot_heatmap(
        weights,
        gt_pixels,
        figure_dir
        / 'route_weight_heatmap.png',
        args.heatmap_samples
    )

    plot_profiles(
        weights,
        gt_pixels,
        sample_ids,
        figure_dir
        / 'route_weight_profiles.png'
    )

    plot_group_mean_profiles(
        weights,
        groups,
        figure_dir
        / 'route_weight_group_profiles.png'
    )

    mean_p, mean_s = (
        plot_relation(
            means,
            weights,
            'Channel spatial mean',
            figure_dir
            / 'mean_vs_route_weight.png'
        )
    )

    std_p, std_s = (
        plot_relation(
            stds,
            weights,
            'Channel spatial std',
            figure_dir
            / 'std_vs_route_weight.png'
        )
    )

    similarity = (
        plot_cosine(
            weights,
            gt_pixels,
            figure_dir
            / 'routing_cosine_similarity.png'
        )
    )

    plot_channel_variation(
        weights,
        figure_dir
        / 'route_weight_channel_variation.png'
    )


    n = similarity.shape[0]

    if n > 1:
        off_diag = (
            ~np.eye(
                n,
                dtype=bool
            )
        )

        pairwise_values = (
            similarity[
                off_diag
            ]
        )

        mean_pairwise_cosine = float(
            pairwise_values.mean()
        )

        std_pairwise_cosine = float(
            pairwise_values.std()
        )

        min_pairwise_cosine = float(
            pairwise_values.min()
        )

    else:
        mean_pairwise_cosine = np.nan
        std_pairwise_cosine = np.nan
        min_pairwise_cosine = np.nan

    mean_channel_weight_std = float(
        channel_weight_std.mean()
    )

    max_channel_weight_std = float(
        channel_weight_std.max()
    )

    most_variable_channel = int(
        np.argmax(
            channel_weight_std
        )
    )

    overall_weight_min = float(
        weights.min()
    )

    overall_weight_max = float(
        weights.max()
    )

    overall_weight_mean = float(
        weights.mean()
    )

    overall_weight_std = float(
        weights.std()
    )


    summary = '\n'.join([
        '===== KSDD2 DAPR internal visualization =====',
        '',
        'Model:',
        'DAPR-FRFN seed42',
        'Official Defect IoU = 73.21',
        '',
        'Exact hooked modules:',
        'Feature input  = frfn_refine.frfn',
        'Route weights = frfn_refine.frfn.channel_router',
        '',
        f'Positive samples = {len(sample_ids)}',
        f'Channels = {channels}',
        '',
        f'Mean shape    = {means.shape}',
        f'Std shape     = {stds.shape}',
        f'Weights shape = {weights.shape}',
        '',
        'Defect-area grouping:',
        f'Q1 = {q1:.2f} pixels',
        f'Q2 = {q2:.2f} pixels',
        '',
        'Route-weight range:',
        f'Min  = {overall_weight_min:.6f}',
        f'Max  = {overall_weight_max:.6f}',
        f'Mean = {overall_weight_mean:.6f}',
        f'Std  = {overall_weight_std:.6f}',
        '',
        'Statistic-weight relationships:',
        (
            'Mean vs Weight: '
            f'Pearson={mean_p:.6f}, '
            f'Spearman={mean_s:.6f}'
        ),
        (
            'Std  vs Weight: '
            f'Pearson={std_p:.6f}, '
            f'Spearman={std_s:.6f}'
        ),
        '',
        'Sample-adaptive routing variation:',
        (
            'Mean pairwise cosine similarity = '
            f'{mean_pairwise_cosine:.6f}'
        ),
        (
            'Std pairwise cosine similarity  = '
            f'{std_pairwise_cosine:.6f}'
        ),
        (
            'Min pairwise cosine similarity  = '
            f'{min_pairwise_cosine:.6f}'
        ),
        (
            'Mean channel-wise weight std    = '
            f'{mean_channel_weight_std:.6f}'
        ),
        (
            'Max channel-wise weight std     = '
            f'{max_channel_weight_std:.6f}'
        ),
        (
            'Most variable channel           = '
            f'{most_variable_channel}'
        ),
        '',
        'Interpretation rules:',
        (
            '1. Weak simple correlation between a single statistic '
            'and route weight is NOT evidence that statistics are unused, '
            'because routing is jointly generated by a nonlinear Mean+Std '
            'mapping.'
        ),
        (
            '2. Sample-adaptive routing should primarily be judged from '
            'sample-to-sample route-profile variation, not from a single '
            'channel correlation.'
        ),
        (
            '3. If pairwise cosine similarity is extremely close to 1 '
            'and channel-wise route-weight std is near zero, avoid strong '
            '"sample-adaptive routing" claims in the paper.'
        ),
    ])

    print(
        '\n' + summary
    )

    (
        args.output_dir
        / 'summary.txt'
    ).write_text(
        summary + '\n',
        encoding='utf-8'
    )

    print(
        '\n===== Saved ====='
    )

    print(
        raw_dir
        / 'dapr_internal_features.npz'
    )

    print(
        csv_dir
        / 'sample_level_stats.csv'
    )

    print(
        csv_dir
        / 'channel_level_stats.csv'
    )

    print(
        csv_dir
        / 'scale_group_stats.csv'
    )

    print(
        figure_dir
    )

    print(
        args.output_dir
        / 'summary.txt'
    )


if __name__ == '__main__':
    main()
