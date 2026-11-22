import argparse
from collections import defaultdict
from pathlib import Path
import random

import cv2
import numpy as np


DATA_ROOT = Path('data/NEU_Seg')

TRAIN_IMAGE_DIR = DATA_ROOT / "images" / "training"
TRAIN_MASK_DIR = DATA_ROOT / "annotations" / "training"

TEST_IMAGE_DIR = DATA_ROOT / "images" / "test"
TEST_MASK_DIR = DATA_ROOT / "annotations" / "test"

SPLIT_DIR = DATA_ROOT / "splits"

VAL_RATIO = 0.2
RANDOM_SEED = 42
VALID_LABELS = {0, 1, 2, 3}


def check_image_mask_pairs(image_dir, mask_dir):
    """检查.jpg图像与.png掩码是否一一对应。"""

    image_paths = sorted(image_dir.glob("*.jpg"))
    mask_paths = sorted(mask_dir.glob("*.png"))

    image_stems = {path.stem for path in image_paths}
    mask_stems = {path.stem for path in mask_paths}

    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)

    if missing_masks:
        raise RuntimeError(
            f"有{len(missing_masks)}张图像缺少标签，例如：{missing_masks[:5]}"
        )

    if missing_images:
        raise RuntimeError(
            f"有{len(missing_images)}个标签缺少图像，例如：{missing_images[:5]}"
        )

    if len(image_paths) == 0:
        raise RuntimeError(f"没有在该目录找到.jpg图像：{image_dir}")

    return image_paths


def get_label_combination(mask_path):
    """获取一张标签图中出现的像素类别组合。"""

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise RuntimeError(f"无法读取标签：{mask_path}")

    if mask.ndim != 2:
        raise RuntimeError(
            f"标签不是单通道图像：{mask_path}，尺寸为{mask.shape}"
        )

    labels = {int(value) for value in np.unique(mask)}

    invalid_labels = labels - VALID_LABELS
    if invalid_labels:
        raise RuntimeError(
            f"标签{mask_path.name}出现非法像素值：{sorted(invalid_labels)}"
        )

    return tuple(sorted(labels))


def write_split_file(file_path, stems):
    """将文件名写入划分文件，不包含扩展名。"""

    with file_path.open("w", encoding="utf-8") as file:
        for stem in sorted(stems):
            file.write(stem + "\n")


def create_train_val_split():
    train_image_paths = check_image_mask_pairs(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR
    )

    groups = defaultdict(list)

    print("正在读取训练标签并统计类别组合……")

    for index, image_path in enumerate(train_image_paths, start=1):
        mask_path = TRAIN_MASK_DIR / f"{image_path.stem}.png"
        label_combination = get_label_combination(mask_path)
        groups[label_combination].append(image_path.stem)

        if index % 500 == 0:
            print(f"已检查：{index}/{len(train_image_paths)}")

    total_number = len(train_image_paths)
    target_val_number = round(total_number * VAL_RATIO)


    val_numbers = {}
    decimal_parts = []

    for label_combination in sorted(groups):
        group_number = len(groups[label_combination])
        exact_val_number = group_number * VAL_RATIO
        base_val_number = int(exact_val_number)

        val_numbers[label_combination] = base_val_number

        decimal_parts.append(
            (
                exact_val_number - base_val_number,
                label_combination
            )
        )


    remaining_number = (
        target_val_number - sum(val_numbers.values())
    )

    decimal_parts.sort(
        key=lambda item: (-item[0], item[1])
    )

    for _, label_combination in decimal_parts:
        if remaining_number <= 0:
            break

        if val_numbers[label_combination] < len(groups[label_combination]):
            val_numbers[label_combination] += 1
            remaining_number -= 1

    random_generator = random.Random(RANDOM_SEED)

    train_stems = []
    val_stems = []

    print("\n各类别组合划分结果：")

    for label_combination in sorted(groups):
        group_stems = sorted(groups[label_combination])
        random_generator.shuffle(group_stems)

        val_number = val_numbers[label_combination]

        current_val_stems = group_stems[:val_number]
        current_train_stems = group_stems[val_number:]

        val_stems.extend(current_val_stems)
        train_stems.extend(current_train_stems)

        print(
            f"  {label_combination}: "
            f"总数={len(group_stems)}, "
            f"训练={len(current_train_stems)}, "
            f"验证={len(current_val_stems)}"
        )

    return train_stems, val_stems


def main():
    global DATA_ROOT
    global TRAIN_IMAGE_DIR, TRAIN_MASK_DIR
    global TEST_IMAGE_DIR, TEST_MASK_DIR, SPLIT_DIR
    global VAL_RATIO, RANDOM_SEED

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--val-ratio', type=float, default=VAL_RATIO)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    VAL_RATIO = args.val_ratio
    RANDOM_SEED = args.seed
    TRAIN_IMAGE_DIR = DATA_ROOT / 'images' / 'training'
    TRAIN_MASK_DIR = DATA_ROOT / 'annotations' / 'training'
    TEST_IMAGE_DIR = DATA_ROOT / 'images' / 'test'
    TEST_MASK_DIR = DATA_ROOT / 'annotations' / 'test'
    SPLIT_DIR = DATA_ROOT / 'splits'

    train_stems, val_stems = create_train_val_split()

    test_image_paths = check_image_mask_pairs(
        TEST_IMAGE_DIR,
        TEST_MASK_DIR
    )
    test_stems = [path.stem for path in test_image_paths]

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    write_split_file(SPLIT_DIR / "train.txt", train_stems)
    write_split_file(SPLIT_DIR / "val.txt", val_stems)
    write_split_file(SPLIT_DIR / "test.txt", test_stems)

    print("\n===== 划分完成 =====")
    print(f"训练集数量：{len(train_stems)}")
    print(f"验证集数量：{len(val_stems)}")
    print(f"测试集数量：{len(test_stems)}")
    print(f"随机种子：{RANDOM_SEED}")
    print(f"划分文件保存位置：{SPLIT_DIR}")

    assert len(train_stems) == 2904
    assert len(val_stems) == 726
    assert len(test_stems) == 840
    assert set(train_stems).isdisjoint(set(val_stems))

    print("数据数量和训练/验证互斥检查通过。")


if __name__ == "__main__":
    main()
