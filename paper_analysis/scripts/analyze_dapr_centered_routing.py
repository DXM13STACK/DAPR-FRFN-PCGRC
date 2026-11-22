from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Centered DAPR routing analysis using '
            'previously extracted KSDD2 internal tensors.'
        )
    )

    parser.add_argument(
        '--npz',
        type=Path,
        default=ROOT
        / 'paper_analysis/mechanism/dapr/ksdd2/raw/'
          'dapr_internal_features.npz'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT
        / 'paper_analysis/mechanism/dapr/ksdd2'
    )

    parser.add_argument(
        '--heatmap-samples',
        type=int,
        default=60
    )

    return parser.parse_args()


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if (
        len(x) == 0
        or len(y) == 0
        or x.std() == 0
        or y.std() == 0
    ):
        return np.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def spearman(x, y):
    x = pd.Series(
        np.asarray(x).reshape(-1)
    ).rank()

    y = pd.Series(
        np.asarray(y).reshape(-1)
    ).rank()

    return float(
        x.corr(y)
    )


def cosine_similarity_matrix(x):
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


def choose_representatives(gt_pixels):
    """
    Representative positive samples from:
        small / medium / large thirds
    """

    order = np.argsort(
        gt_pixels
    )

    groups = np.array_split(
        order,
        3
    )

    return [
        int(
            group[
                len(group) // 2
            ]
        )
        for group
        in groups
    ]


def plot_centered_heatmap(
    delta,
    gt_pixels,
    output_path,
    max_samples,
):
    order = np.argsort(
        gt_pixels
    )

    if len(order) > max_samples:
        positions = np.linspace(
            0,
            len(order) - 1,
            max_samples
        )

        positions = np.round(
            positions
        ).astype(int)

        order = order[
            positions
        ]

    matrix = delta[
        order
    ]


    vmax = float(
        np.quantile(
            np.abs(matrix),
            0.995
        )
    )

    if vmax <= 0:
        vmax = 1e-6

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    im = ax.imshow(
        matrix,
        aspect='auto',
        interpolation='nearest',
        cmap='coolwarm',
        vmin=-vmax,
        vmax=vmax
    )

    ax.set_xlabel(
        'Channel index'
    )

    ax.set_ylabel(
        'Positive samples '
        '(sorted by GT defect area)'
    )

    ax.set_title(
        'Centered DAPR routing modulation '
        r'$\Delta A = A - \bar{A}$'
    )

    fig.colorbar(
        im,
        ax=ax,
        label='Routing-weight deviation'
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight'
    )

    plt.close(fig)


def plot_centered_profiles(
    delta,
    gt_pixels,
    sample_ids,
    output_path,
):
    indices = choose_representatives(
        gt_pixels
    )

    labels = [
        'Small',
        'Medium',
        'Large',
    ]

    channels = np.arange(
        delta.shape[1]
    )

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    for index, label in zip(
        indices,
        labels
    ):
        ax.plot(
            channels,
            delta[index],
            linewidth=1.1,
            label=(
                f'{label}: '
                f'{sample_ids[index]} '
                f'(GT={gt_pixels[index]})'
            )
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle='--'
    )

    ax.set_xlabel(
        'Channel index'
    )

    ax.set_ylabel(
        r'Centered routing modulation $\Delta A$'
    )

    ax.set_title(
        'Sample-specific deviations from '
        'the mean DAPR routing template'
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight'
    )

    plt.close(fig)


def plot_mean_template(
    mean_template,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        np.arange(
            len(mean_template)
        ),
        mean_template
    )

    ax.set_xlabel(
        'Channel index'
    )

    ax.set_ylabel(
        'Mean routing weight'
    )

    ax.set_title(
        'Stable mean DAPR routing template'
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight'
    )

    plt.close(fig)


def plot_modulation_magnitude(
    modulation_rms,
    gt_pixels,
    groups,
    output_path,
):
    order = np.argsort(
        gt_pixels
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        np.arange(
            len(order)
        ),
        modulation_rms[
            order
        ],
        marker='o',
        markersize=3,
        linewidth=1.0
    )

    ax.set_xlabel(
        'Positive sample index '
        '(sorted by GT defect area)'
    )

    ax.set_ylabel(
        'RMS routing modulation'
    )

    ax.set_title(
        'Magnitude of sample-specific DAPR routing modulation'
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight'
    )

    plt.close(fig)


def plot_centered_cosine(
    delta,
    gt_pixels,
    output_path,
):
    order = np.argsort(
        gt_pixels
    )

    ordered = delta[
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
        interpolation='nearest',
        cmap='coolwarm',
        vmin=-1,
        vmax=1
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
        'Cosine similarity of centered '
        'DAPR routing modulations'
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


def main():
    args = parse_args()

    if not args.npz.is_file():
        raise FileNotFoundError(
            f'NPZ not found:\n{args.npz}'
        )

    csv_dir = (
        args.output_dir
        / 'csv'
    )

    figure_dir = (
        args.output_dir
        / 'figures'
    )

    raw_dir = (
        args.output_dir
        / 'raw'
    )

    csv_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    data = np.load(
        args.npz,
        allow_pickle=True
    )

    required = [
        'sample_ids',
        'gt_pixels',
        'weights',
    ]

    for key in required:
        if key not in data:
            raise RuntimeError(
                f'Missing NPZ key: {key}'
            )

    sample_ids = data[
        'sample_ids'
    ]

    gt_pixels = data[
        'gt_pixels'
    ].astype(np.int64)

    weights = data[
        'weights'
    ].astype(np.float64)

    if weights.ndim != 2:
        raise RuntimeError(
            f'Expected [N,C] weights, '
            f'got {weights.shape}'
        )

    n_samples, channels = (
        weights.shape
    )

    print(
        f'Samples  = {n_samples}'
    )

    print(
        f'Channels = {channels}'
    )


    if 'groups' in data:
        groups = data[
            'groups'
        ].astype(str)

    else:
        q1 = np.quantile(
            gt_pixels,
            1.0 / 3.0
        )

        q2 = np.quantile(
            gt_pixels,
            2.0 / 3.0
        )

        groups = np.asarray([
            (
                'SMALL'
                if value <= q1
                else
                'MEDIUM'
                if value <= q2
                else
                'LARGE'
            )
            for value
            in gt_pixels
        ])


    mean_template = (
        weights.mean(
            axis=0
        )
    )


    delta = (
        weights
        - mean_template[
            None,
            :
        ]
    )


    centered_channel_mean = (
        delta.mean(
            axis=0
        )
    )

    max_centering_error = float(
        np.abs(
            centered_channel_mean
        ).max()
    )


    modulation_rms = np.sqrt(
        np.mean(
            delta ** 2,
            axis=1
        )
    )

    modulation_l1 = np.mean(
        np.abs(delta),
        axis=1
    )

    modulation_max = np.max(
        np.abs(delta),
        axis=1
    )


    total_var = float(
        np.var(
            weights
        )
    )

    static_var = float(
        np.var(
            mean_template
        )
    )

    adaptive_var = float(
        np.mean(
            np.var(
                weights,
                axis=0
            )
        )
    )

    decomposed_var = (
        static_var
        + adaptive_var
    )

    if total_var > 0:
        static_fraction = (
            static_var
            / total_var
        )

        adaptive_fraction = (
            adaptive_var
            / total_var
        )

    else:
        static_fraction = np.nan
        adaptive_fraction = np.nan


    area_pearson = pearson(
        gt_pixels,
        modulation_rms
    )

    area_spearman = spearman(
        gt_pixels,
        modulation_rms
    )


    centered_similarity = (
        plot_centered_cosine(
            delta,
            gt_pixels,
            figure_dir
            / 'centered_routing_cosine_similarity.png'
        )
    )

    if n_samples > 1:
        mask = ~np.eye(
            n_samples,
            dtype=bool
        )

        centered_pairwise = (
            centered_similarity[
                mask
            ]
        )

        centered_cos_mean = float(
            centered_pairwise.mean()
        )

        centered_cos_std = float(
            centered_pairwise.std()
        )

        centered_cos_min = float(
            centered_pairwise.min()
        )

        centered_cos_max = float(
            centered_pairwise.max()
        )

    else:
        centered_cos_mean = np.nan
        centered_cos_std = np.nan
        centered_cos_min = np.nan
        centered_cos_max = np.nan


    plot_centered_heatmap(
        delta,
        gt_pixels,
        figure_dir
        / 'centered_route_weight_heatmap.png',
        args.heatmap_samples
    )

    plot_centered_profiles(
        delta,
        gt_pixels,
        sample_ids,
        figure_dir
        / 'centered_route_weight_profiles.png'
    )

    plot_mean_template(
        mean_template,
        figure_dir
        / 'mean_route_weight_template.png'
    )

    plot_modulation_magnitude(
        modulation_rms,
        gt_pixels,
        groups,
        figure_dir
        / 'routing_modulation_magnitude.png'
    )


    np.savez_compressed(
        raw_dir
        / 'dapr_centered_routing.npz',

        sample_ids=
            sample_ids,

        gt_pixels=
            gt_pixels,

        groups=
            groups,

        mean_template=
            mean_template,

        centered_weights=
            delta,

        modulation_rms=
            modulation_rms,

        modulation_l1=
            modulation_l1,

        modulation_max=
            modulation_max,
    )


    sample_rows = []

    for index in range(
        n_samples
    ):
        sample_rows.append({
            'sample_id':
                sample_ids[index],

            'gt_pixels':
                int(
                    gt_pixels[index]
                ),

            'scale_group':
                groups[index],

            'modulation_rms':
                float(
                    modulation_rms[index]
                ),

            'modulation_l1':
                float(
                    modulation_l1[index]
                ),

            'modulation_max_abs':
                float(
                    modulation_max[index]
                ),
        })

    sample_df = pd.DataFrame(
        sample_rows
    )

    sample_df.to_csv(
        csv_dir
        / 'centered_routing_sample_stats.csv',
        index=False,
        encoding='utf-8-sig'
    )


    group_rows = []

    for name in (
        'SMALL',
        'MEDIUM',
        'LARGE',
    ):
        mask = (
            groups == name
        )

        values = (
            modulation_rms[
                mask
            ]
        )

        group_rows.append({
            'group':
                name,

            'n':
                int(
                    mask.sum()
                ),

            'mean_modulation_rms':
                float(
                    values.mean()
                ),

            'std_modulation_rms':
                float(
                    values.std()
                ),

            'median_modulation_rms':
                float(
                    np.median(
                        values
                    )
                ),

            'min_modulation_rms':
                float(
                    values.min()
                ),

            'max_modulation_rms':
                float(
                    values.max()
                ),
        })

    pd.DataFrame(
        group_rows
    ).to_csv(
        csv_dir
        / 'centered_routing_group_stats.csv',
        index=False,
        encoding='utf-8-sig'
    )


    ranking = np.argsort(
        modulation_rms
    )[::-1]

    top_rows = []

    for rank, index in enumerate(
        ranking[:20],
        start=1
    ):
        top_rows.append({
            'rank':
                rank,

            'sample_id':
                sample_ids[index],

            'gt_pixels':
                int(
                    gt_pixels[index]
                ),

            'scale_group':
                groups[index],

            'modulation_rms':
                float(
                    modulation_rms[index]
                ),

            'modulation_l1':
                float(
                    modulation_l1[index]
                ),

            'modulation_max_abs':
                float(
                    modulation_max[index]
                ),
        })

    pd.DataFrame(
        top_rows
    ).to_csv(
        csv_dir
        / 'centered_routing_top_modulated_samples.csv',
        index=False,
        encoding='utf-8-sig'
    )


    summary = '\n'.join([
        '===== KSDD2 DAPR centered routing analysis =====',
        '',
        f'Samples  = {n_samples}',
        f'Channels = {channels}',
        '',
        'Centered definition:',
        'A_bar[c] = mean_i A[i,c]',
        'DeltaA[i,c] = A[i,c] - A_bar[c]',
        '',
        (
            'Maximum channel-centering error = '
            f'{max_centering_error:.10f}'
        ),
        '',
        'Variance decomposition:',
        (
            'Total routing variance           = '
            f'{total_var:.10f}'
        ),
        (
            'Stable channel-template variance = '
            f'{static_var:.10f}'
        ),
        (
            'Adaptive within-channel variance = '
            f'{adaptive_var:.10f}'
        ),
        (
            'Static + adaptive variance       = '
            f'{decomposed_var:.10f}'
        ),
        (
            'Static-template fraction         = '
            f'{100*static_fraction:.4f}%'
        ),
        (
            'Sample-modulation fraction       = '
            f'{100*adaptive_fraction:.4f}%'
        ),
        '',
        'Sample modulation magnitude:',
        (
            'Mean RMS modulation   = '
            f'{modulation_rms.mean():.6f}'
        ),
        (
            'Median RMS modulation = '
            f'{np.median(modulation_rms):.6f}'
        ),
        (
            'Min RMS modulation    = '
            f'{modulation_rms.min():.6f}'
        ),
        (
            'Max RMS modulation    = '
            f'{modulation_rms.max():.6f}'
        ),
        '',
        'Defect area vs modulation RMS:',
        (
            'Pearson  = '
            f'{area_pearson:.6f}'
        ),
        (
            'Spearman = '
            f'{area_spearman:.6f}'
        ),
        '',
        'Centered modulation pairwise cosine:',
        (
            'Mean = '
            f'{centered_cos_mean:.6f}'
        ),
        (
            'Std  = '
            f'{centered_cos_std:.6f}'
        ),
        (
            'Min  = '
            f'{centered_cos_min:.6f}'
        ),
        (
            'Max  = '
            f'{centered_cos_max:.6f}'
        ),
        '',
        'Interpretation guide:',
        (
            'A high static-template fraction means DAPR primarily '
            'learns a stable channel preference.'
        ),
        (
            'A non-zero modulation fraction and heterogeneous '
            'centered profiles indicate additional '
            'input-conditioned adjustment around that template.'
        ),
        (
            'Do not call the routing strongly sample-adaptive '
            'unless the modulation component is quantitatively meaningful.'
        ),
    ])

    print(
        '\n' + summary
    )

    summary_path = (
        args.output_dir
        / 'centered_routing_summary.txt'
    )

    summary_path.write_text(
        summary + '\n',
        encoding='utf-8'
    )

    print(
        '\n===== Saved ====='
    )

    print(
        summary_path
    )

    print(
        figure_dir
        / 'centered_route_weight_heatmap.png'
    )

    print(
        figure_dir
        / 'centered_route_weight_profiles.png'
    )

    print(
        figure_dir
        / 'mean_route_weight_template.png'
    )

    print(
        figure_dir
        / 'routing_modulation_magnitude.png'
    )

    print(
        figure_dir
        / 'centered_routing_cosine_similarity.png'
    )


if __name__ == '__main__':
    main()
