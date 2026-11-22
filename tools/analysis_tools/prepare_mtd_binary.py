import argparse
from pathlib import Path
import csv
import random
import shutil

import cv2
import numpy as np


SOURCE_ROOT = Path('data/Magnetic-tile-defect-datasets.-master')
OUTPUT_ROOT = Path('data/MTD_binary')


CATEGORIES = [
    'MT_Blowhole',
    'MT_Break',
    'MT_Crack',
    'MT_Fray',
    'MT_Free',
    'MT_Uneven',
]


SPLIT_SEED = 42

TRAIN_RATIO = 0.64
VAL_RATIO = 0.16


MASK_THRESHOLD = 127


def load_mask(path: Path):

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise RuntimeError(
            f'Cannot read mask: {path}'
        )

    return mask


def safe_name(category, stem):

    return (
        f'{category}__{stem}'
    )


def split_items(items, seed):

    items = list(items)

    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)

    n_train = round(
        n * TRAIN_RATIO
    )

    n_val = round(
        n * VAL_RATIO
    )

    train = items[
        :n_train
    ]

    val = items[
        n_train:
        n_train + n_val
    ]

    test = items[
        n_train + n_val:
    ]

    return train, val, test


def main():
    global SOURCE_ROOT, OUTPUT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    SOURCE_ROOT = args.source_root
    OUTPUT_ROOT = args.output_root

    print('=' * 90)
    print('Prepare MTD binary segmentation dataset')
    print('=' * 90)

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            SOURCE_ROOT
        )


    if OUTPUT_ROOT.exists():

        print(
            f'Removing old output: '
            f'{OUTPUT_ROOT}'
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    for split in [
        'train',
        'val',
        'test',
    ]:

        (
            OUTPUT_ROOT
            / 'images'
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT_ROOT
            / 'annotations'
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )


    samples_by_category = {}

    excluded = []

    for category in CATEGORIES:

        folder = (
            SOURCE_ROOT
            / category
            / 'Imgs'
        )

        jpgs = {
            p.stem: p
            for p in folder.glob(
                '*.jpg'
            )
        }

        pngs = {
            p.stem: p
            for p in folder.glob(
                '*.png'
            )
        }

        common = sorted(
            set(jpgs)
            &
            set(pngs)
        )

        samples = []

        for stem in common:

            image_path = (
                jpgs[stem]
            )

            mask_path = (
                pngs[stem]
            )

            raw_mask = load_mask(
                mask_path
            )

            raw_nonzero = int(
                np.count_nonzero(
                    raw_mask
                )
            )


            if category == 'MT_Free':

                samples.append({
                    'category':
                        category,

                    'stem':
                        stem,

                    'image':
                        image_path,

                    'mask':
                        mask_path,
                })

                continue


            if raw_nonzero == 0:

                excluded.append({
                    'category':
                        category,

                    'stem':
                        stem,

                    'reason':
                        'defect_category_but_empty_mask',
                })

                continue

            samples.append({
                'category':
                    category,

                'stem':
                    stem,

                'image':
                    image_path,

                'mask':
                    mask_path,
            })

        samples_by_category[
            category
        ] = samples


    print()
    print('[1] USABLE SAMPLES')

    total_usable = 0

    for category in CATEGORIES:

        n = len(
            samples_by_category[
                category
            ]
        )

        total_usable += n

        print(
            f'{category:20s}: '
            f'{n}'
        )

    print(
        f'Total usable: '
        f'{total_usable}'
    )

    print()
    print(
        f'Excluded: '
        f'{len(excluded)}'
    )

    for x in excluded:

        print(
            f'  {x["category"]} / '
            f'{x["stem"]} / '
            f'{x["reason"]}'
        )


    split_data = {
        'train': [],
        'val': [],
        'test': [],
    }

    print()
    print('[2] STRATIFIED SPLIT')

    for idx, category in enumerate(
        CATEGORIES
    ):

        samples = (
            samples_by_category[
                category
            ]
        )

        train, val, test = (
            split_items(
                samples,
                seed=(
                    SPLIT_SEED
                    +
                    idx * 1000
                )
            )
        )

        split_data[
            'train'
        ].extend(
            train
        )

        split_data[
            'val'
        ].extend(
            val
        )

        split_data[
            'test'
        ].extend(
            test
        )

        print(
            f'{category:20s} '
            f'train={len(train):4d} '
            f'val={len(val):4d} '
            f'test={len(test):4d}'
        )


    for idx, split in enumerate([
        'train',
        'val',
        'test',
    ]):

        rng = random.Random(
            SPLIT_SEED
            +
            10000
            +
            idx
        )

        rng.shuffle(
            split_data[
                split
            ]
        )


    manifest_rows = []

    print()
    print('[3] CONVERTING')

    for split in [
        'train',
        'val',
        'test',
    ]:

        image_out_dir = (
            OUTPUT_ROOT
            / 'images'
            / split
        )

        mask_out_dir = (
            OUTPUT_ROOT
            / 'annotations'
            / split
        )

        for idx, item in enumerate(
            split_data[split],
            start=1
        ):

            category = (
                item[
                    'category'
                ]
            )

            stem = (
                item[
                    'stem'
                ]
            )

            name = safe_name(
                category,
                stem
            )

            image_out = (
                image_out_dir
                / f'{name}.jpg'
            )

            mask_out = (
                mask_out_dir
                / f'{name}.png'
            )


            shutil.copy2(
                item['image'],
                image_out
            )


            raw_mask = load_mask(
                item['mask']
            )


            binary_mask = (
                raw_mask
                >
                MASK_THRESHOLD
            ).astype(
                np.uint8
            )

            ok = cv2.imwrite(
                str(mask_out),
                binary_mask
            )

            if not ok:
                raise RuntimeError(
                    f'Failed to save: '
                    f'{mask_out}'
                )

            fg_pixels = int(
                binary_mask.sum()
            )

            total_pixels = int(
                binary_mask.size
            )

            fg_ratio = (
                fg_pixels
                /
                total_pixels
            )

            manifest_rows.append({
                'split':
                    split,

                'category':
                    category,

                'original_stem':
                    stem,

                'filename':
                    name,

                'fg_pixels':
                    fg_pixels,

                'total_pixels':
                    total_pixels,

                'fg_ratio':
                    fg_ratio,
            })

            if (
                idx == 1
                or
                idx % 200 == 0
                or
                idx == len(
                    split_data[
                        split
                    ]
                )
            ):

                print(
                    f'\r{split:5s}: '
                    f'{idx}/'
                    f'{len(split_data[split])}',
                    end='',
                    flush=True
                )

        print()


    manifest_path = (
        OUTPUT_ROOT
        / 'split_manifest.csv'
    )

    with manifest_path.open(
        'w',
        newline='',
        encoding='utf-8-sig'
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                'split',
                'category',
                'original_stem',
                'filename',
                'fg_pixels',
                'total_pixels',
                'fg_ratio',
            ]
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )


    excluded_path = (
        OUTPUT_ROOT
        / 'excluded_samples.csv'
    )

    with excluded_path.open(
        'w',
        newline='',
        encoding='utf-8-sig'
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                'category',
                'stem',
                'reason',
            ]
        )

        writer.writeheader()

        writer.writerows(
            excluded
        )


    print()
    print('[4] FINAL VALIDATION')

    for split in [
        'train',
        'val',
        'test',
    ]:

        image_files = list(
            (
                OUTPUT_ROOT
                / 'images'
                / split
            ).glob(
                '*.jpg'
            )
        )

        mask_files = list(
            (
                OUTPUT_ROOT
                / 'annotations'
                / split
            ).glob(
                '*.png'
            )
        )

        positive = 0
        negative = 0

        all_values = set()

        ratios = []

        for p in mask_files:

            mask = cv2.imread(
                str(p),
                cv2.IMREAD_GRAYSCALE
            )

            values = np.unique(
                mask
            )

            all_values.update(
                values.tolist()
            )

            fg = int(
                mask.sum()
            )

            if fg > 0:
                positive += 1
            else:
                negative += 1

            ratios.append(
                fg
                /
                mask.size
            )

        print()

        print(
            f'{split.upper()}'
        )

        print(
            f'  images       = '
            f'{len(image_files)}'
        )

        print(
            f'  masks        = '
            f'{len(mask_files)}'
        )

        print(
            f'  positive     = '
            f'{positive}'
        )

        print(
            f'  negative     = '
            f'{negative}'
        )

        print(
            f'  mask values  = '
            f'{sorted(all_values)}'
        )

        print(
            f'  mean FG      = '
            f'{np.mean(ratios) * 100:.6f}%'
        )

        print(
            f'  median FG    = '
            f'{np.median(ratios) * 100:.6f}%'
        )

    print()
    print('=' * 90)

    print(
        f'Dataset prepared at:\n'
        f'{OUTPUT_ROOT}'
    )

    print()

    print(
        f'Manifest:\n'
        f'{manifest_path}'
    )

    print()

    print(
        f'Excluded samples:\n'
        f'{excluded_path}'
    )

    print('=' * 90)


if __name__ == '__main__':
    main()
