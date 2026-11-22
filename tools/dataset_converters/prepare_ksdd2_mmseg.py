import argparse
import csv
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

MASK_KEYWORDS = (
    'label',
    'labels',
    'mask',
    'masks',
    'annotation',
    'annotations',
    'groundtruth',
    'ground_truth',
    'gt',
    'seg',
    'segmentation',
)

MASK_SUFFIXES = (
    '_label',
    '_labels',
    '_mask',
    '_masks',
    '_annotation',
    '_annotations',
    '_groundtruth',
    '_ground_truth',
    '_gt',
    '_seg',
    '_segmentation',
)


def parse_args():
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description='将官方 KSDD2 整理为 MMSegmentation 二分类目录结构。'
    )

    parser.add_argument(
        '--raw-root',
        type=Path,
        default=project_root / 'KSDD2',
        help='官方解压后的 KSDD2 根目录。'
    )

    parser.add_argument(
        '--output-root',
        type=Path,
        default=project_root / 'data' / 'KSDD2',
        help='整理后的 MMSegmentation 数据目录。'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.10,
        help='从官方训练集划给验证集的比例，默认 0.10。'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=2026,
        help='划分随机种子。'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='若输出目录已存在，允许覆盖其中同名文件。'
    )

    return parser.parse_args()


def is_mask_name(path: Path) -> bool:
    stem = path.stem.lower()
    parts = [part.lower() for part in path.parts]

    if any(keyword in stem for keyword in MASK_KEYWORDS):
        return True

    return any(
        part in {'label', 'labels', 'mask', 'masks',
                 'annotation', 'annotations',
                 'ground_truth', 'groundtruth', 'gt'}
        for part in parts
    )


def find_images(split_dir: Path):
    image_paths = []

    for path in sorted(split_dir.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if is_mask_name(path):
            continue
        image_paths.append(path)

    return image_paths


def find_mask_for_image(image_path: Path, split_dir: Path):
    """
    支持 KSDD2 常见命名：
    image.jpg       -> image_label.bmp / image_mask.png / image_gt.png
    image.jpg       -> labels/image.png
    image.jpg       -> masks/image.png
    """

    stem = image_path.stem
    suffixes = list(IMAGE_SUFFIXES)

    candidates = []


    for mask_suffix in MASK_SUFFIXES:
        for extension in suffixes:
            candidates.append(
                image_path.with_name(f'{stem}{mask_suffix}{extension}')
            )


    for extension in suffixes:
        candidates.append(image_path.with_suffix(extension))


    label_dir_names = (
        'label',
        'labels',
        'mask',
        'masks',
        'annotation',
        'annotations',
        'ground_truth',
        'groundtruth',
        'gt',
    )

    relative_parent = image_path.parent.relative_to(split_dir)

    for dirname in label_dir_names:
        label_dir = split_dir / dirname / relative_parent

        for extension in suffixes:
            candidates.append(label_dir / f'{stem}{extension}')

        for mask_suffix in MASK_SUFFIXES:
            for extension in suffixes:
                candidates.append(
                    label_dir / f'{stem}{mask_suffix}{extension}'
                )

    checked = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate in checked:
            continue

        checked.add(candidate)

        if candidate.exists() and candidate.is_file():
            if candidate.resolve() != image_path.resolve():
                return candidate

    return None


def read_binary_mask(mask_path: Path, expected_height: int, expected_width: int):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise RuntimeError(f'无法读取标签文件：{mask_path}')

    if mask.ndim == 3:
        binary_mask = np.any(mask > 0, axis=2).astype(np.uint8)
    else:
        binary_mask = (mask > 0).astype(np.uint8)

    if binary_mask.shape != (expected_height, expected_width):
        raise RuntimeError(
            f'图像与标签尺寸不一致：\n'
            f'标签：{mask_path}\n'
            f'标签尺寸：{binary_mask.shape[::-1]}\n'
            f'图像尺寸：{(expected_width, expected_height)}'
        )

    return binary_mask


def link_or_copy(source: Path, destination: Path, overwrite: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f'目标文件已存在：{destination}\n'
                '如确认需要覆盖，请加入 --overwrite。'
            )
        destination.unlink()

    try:
        os.link(source, destination)
        return 'hardlink'
    except OSError:
        shutil.copy2(source, destination)
        return 'copy'


def output_filename(image_path: Path, split_dir: Path):
    """
    防止未来原始数据出现重名：
    train/a/001.jpg -> a__001.jpg
    当前 KSDD2 为平铺目录时，文件名保持不变。
    """
    relative_path = image_path.relative_to(split_dir)
    parts = list(relative_path.parts)
    parts[-1] = Path(parts[-1]).name
    return '__'.join(parts)


def collect_samples(raw_split_dir: Path):
    if not raw_split_dir.exists():
        raise FileNotFoundError(f'找不到目录：{raw_split_dir}')

    samples = []

    for image_path in find_images(raw_split_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f'无法读取图像：{image_path}')

        height, width = image.shape[:2]
        mask_path = find_mask_for_image(image_path, raw_split_dir)

        if mask_path is None:
            mask = np.zeros((height, width), dtype=np.uint8)
            is_positive = False
            mask_source = ''
        else:
            mask = read_binary_mask(
                mask_path=mask_path,
                expected_height=height,
                expected_width=width
            )
            is_positive = bool(mask.any())
            mask_source = str(mask_path)

        samples.append(
            dict(
                image_path=image_path,
                image_name=output_filename(image_path, raw_split_dir),
                mask=mask,
                is_positive=is_positive,
                width=width,
                height=height,
                mask_source=mask_source,
            )
        )

    return samples


def stratified_train_val_split(samples, val_ratio, seed):
    if not 0 < val_ratio < 1:
        raise ValueError('--val-ratio 必须在 0 和 1 之间。')

    positives = [sample for sample in samples if sample['is_positive']]
    negatives = [sample for sample in samples if not sample['is_positive']]

    random_generator = random.Random(seed)
    random_generator.shuffle(positives)
    random_generator.shuffle(negatives)

    positive_val_count = max(1, math.ceil(len(positives) * val_ratio))
    negative_val_count = max(1, math.ceil(len(negatives) * val_ratio))

    val_samples = (
        positives[:positive_val_count]
        + negatives[:negative_val_count]
    )

    train_samples = (
        positives[positive_val_count:]
        + negatives[negative_val_count:]
    )

    random_generator.shuffle(train_samples)
    random_generator.shuffle(val_samples)

    return train_samples, val_samples


def write_split(samples, split_name, output_root, overwrite):
    image_dir = output_root / 'images' / split_name
    annotation_dir = output_root / 'annotations' / split_name

    split_entries = []
    metadata_rows = []

    hardlink_count = 0
    copy_count = 0
    positive_count = 0

    for sample in samples:
        image_name = sample['image_name']
        image_destination = image_dir / image_name
        annotation_destination = (
            annotation_dir / f'{Path(image_name).stem}.png'
        )

        operation = link_or_copy(
            source=sample['image_path'],
            destination=image_destination,
            overwrite=overwrite
        )

        if operation == 'hardlink':
            hardlink_count += 1
        else:
            copy_count += 1

        annotation_destination.parent.mkdir(parents=True, exist_ok=True)

        if annotation_destination.exists() and not overwrite:
            raise FileExistsError(
                f'目标标签已存在：{annotation_destination}\n'
                '如确认需要覆盖，请加入 --overwrite。'
            )

        success = cv2.imwrite(
            str(annotation_destination),
            sample['mask'].astype(np.uint8)
        )

        if not success:
            raise RuntimeError(f'无法保存标签：{annotation_destination}')

        if sample['is_positive']:
            positive_count += 1

        split_entries.append(Path(image_name).stem)

        metadata_rows.append(
            dict(
                split=split_name,
                filename=image_name,
                is_positive=int(sample['is_positive']),
                width=sample['width'],
                height=sample['height'],
                source_image=str(sample['image_path']),
                source_mask=sample['mask_source'],
            )
        )

    return (
        split_entries,
        metadata_rows,
        hardlink_count,
        copy_count,
        positive_count
    )


def main():
    args = parse_args()

    raw_train_dir = args.raw_root / 'train'
    raw_test_dir = args.raw_root / 'test'

    print('=' * 70)
    print('KSDD2 -> MMSegmentation 数据整理')
    print('=' * 70)
    print(f'原始数据目录：{args.raw_root}')
    print(f'输出数据目录：{args.output_root}')
    print(f'验证集比例：{args.val_ratio}')
    print(f'随机种子：{args.seed}')

    print('\n正在读取官方 train 目录……')
    official_train_samples = collect_samples(raw_train_dir)

    print('正在读取官方 test 目录……')
    official_test_samples = collect_samples(raw_test_dir)

    train_positive_count = sum(
        sample['is_positive'] for sample in official_train_samples
    )
    test_positive_count = sum(
        sample['is_positive'] for sample in official_test_samples
    )

    print('\n===== 原始数据统计 =====')
    print(
        f'官方 train：{len(official_train_samples)} 张，'
        f'缺陷图 {train_positive_count} 张，'
        f'正常图 {len(official_train_samples) - train_positive_count} 张'
    )
    print(
        f'官方 test ：{len(official_test_samples)} 张，'
        f'缺陷图 {test_positive_count} 张，'
        f'正常图 {len(official_test_samples) - test_positive_count} 张'
    )


    expected_train_positive = 246
    expected_test_positive = 110

    if (
        train_positive_count != expected_train_positive
        or test_positive_count != expected_test_positive
    ):
        raise RuntimeError(
            '\n检测到的缺陷图数量与 KSDD2 官方全监督划分不一致，'
            '为避免把缺陷图错误当成正常图，脚本已停止。\n'
            f'期望：train={expected_train_positive}，'
            f'test={expected_test_positive}\n'
            f'实际：train={train_positive_count}，'
            f'test={test_positive_count}\n\n'
            '请先执行下面命令，把输出发给我：\n'
            'Get-ChildItem KSDD2\\train -Recurse -File | '
            'Select-Object -First 30 FullName\n'
            'Get-ChildItem KSDD2\\test -Recurse -File | '
            'Select-Object -First 30 FullName'
        )

    train_samples, val_samples = stratified_train_val_split(
        samples=official_train_samples,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    print('\n===== 新划分统计 =====')
    print(
        f'train：{len(train_samples)} 张，'
        f'缺陷图 {sum(x["is_positive"] for x in train_samples)} 张'
    )
    print(
        f'val  ：{len(val_samples)} 张，'
        f'缺陷图 {sum(x["is_positive"] for x in val_samples)} 张'
    )
    print(
        f'test ：{len(official_test_samples)} 张，'
        f'缺陷图 {sum(x["is_positive"] for x in official_test_samples)} 张'
    )

    all_metadata = []
    split_contents = {}
    total_hardlinks = 0
    total_copies = 0

    for split_name, split_samples in (
        ('train', train_samples),
        ('val', val_samples),
        ('test', official_test_samples),
    ):
        print(f'\n正在生成 {split_name}……')

        (
            split_entries,
            metadata_rows,
            hardlink_count,
            copy_count,
            positive_count,
        ) = write_split(
            samples=split_samples,
            split_name=split_name,
            output_root=args.output_root,
            overwrite=args.overwrite
        )

        split_contents[split_name] = split_entries
        all_metadata.extend(metadata_rows)
        total_hardlinks += hardlink_count
        total_copies += copy_count

        print(
            f'{split_name} 完成：{len(split_entries)} 张，'
            f'缺陷图 {positive_count} 张'
        )

    splits_dir = args.output_root / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, entries in split_contents.items():
        split_path = splits_dir / f'{split_name}.txt'

        if split_path.exists() and not args.overwrite:
            raise FileExistsError(
                f'划分文件已存在：{split_path}\n'
                '如确认需要覆盖，请加入 --overwrite。'
            )

        split_path.write_text(
            '\n'.join(entries) + '\n',
            encoding='utf-8'
        )

    metadata_path = args.output_root / 'metadata.csv'

    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(
            f'元数据文件已存在：{metadata_path}\n'
            '如确认需要覆盖，请加入 --overwrite。'
        )

    with metadata_path.open('w', newline='', encoding='utf-8-sig') as file:
        fieldnames = [
            'split',
            'filename',
            'is_positive',
            'width',
            'height',
            'source_image',
            'source_mask',
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metadata)

    print('\n' + '=' * 70)
    print('整理完成。')
    print(f'硬链接图像：{total_hardlinks} 张')
    print(f'普通复制图像：{total_copies} 张')
    print(f'输出目录：{args.output_root}')
    print('=' * 70)


if __name__ == '__main__':
    main()
