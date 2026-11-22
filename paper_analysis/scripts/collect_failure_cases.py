from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]

QUAL_ROOT = (
    ROOT
    / 'paper_analysis'
    / 'qualitative'
)

OUT_ROOT = (
    ROOT
    / 'paper_analysis'
    / 'failure_cases'
)


KNOWN_CASES = {
    'ksdd2': [
        (
            '20231',
            'known_regression_or_hard_case'
        ),
        (
            '20259',
            'known_regression_or_hard_case'
        ),
        (
            '20546',
            'known_regression_or_hard_case'
        ),
        (
            '20159',
            'known_regression_or_hard_case'
        ),
    ],

    'mtd': [
        (
            'MT_Uneven__exp6_num_352479',
            'structurally_difficult_case'
        ),
    ],

    'neu': [
        (
            '000834',
            'inclusion_near_complete_failure'
        ),
        (
            '000744',
            'scratches_complete_failure'
        ),
        (
            '000723',
            'patches_hard_case'
        ),
        (
            '000126',
            'ours_regression_case'
        ),
    ],
}


IMAGE_SUFFIXES = {
    '.png',
    '.jpg',
    '.jpeg',
    '.bmp',
    '.tif',
    '.tiff',
}


def is_allowed_image(path: Path):
    if not path.is_file():
        return False

    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False


    if 'pred_masks' in {
        p.lower()
        for p in path.parts
    }:
        return False

    return True


def path_priority(path: Path):
    """
    Prefer ready-made candidate panels over raw masks/images.
    """
    text = str(path).lower()

    if 'candidates' in text:
        level = 0
    elif 'selected' in text:
        level = 1
    elif 'figures' in text:
        level = 2
    else:
        level = 3

    return (
        level,
        len(str(path))
    )


def find_case_visual(
    dataset_root: Path,
    sample_id: str,
) -> Optional[Path]:
    matches = []

    if not dataset_root.exists():
        return None

    for path in dataset_root.rglob('*'):
        if not is_allowed_image(path):
            continue

        if (
            sample_id.lower()
            in path.name.lower()
        ):
            matches.append(path)

    if not matches:
        return None

    matches.sort(
        key=path_priority
    )

    return matches[0]


def collect_candidate_images(
    dataset_root: Path,
    max_count: int = 50,
):
    """
    Collect candidate/selected visual panels for a broad contact sheet.
    This is useful for manually checking additional failure cases,
    especially MTD.
    """
    images = []

    if not dataset_root.exists():
        return images

    for path in dataset_root.rglob('*'):
        if not is_allowed_image(path):
            continue

        text = str(path).lower()

        if (
            'candidates' in text
            or 'selected' in text
        ):
            images.append(path)


    unique = []

    seen = set()

    for path in images:
        resolved = str(
            path.resolve()
        )

        if resolved in seen:
            continue

        seen.add(resolved)
        unique.append(path)

    unique.sort(
        key=lambda p: str(p)
    )

    return unique[:max_count]


def resize_to_width(
    image: Image.Image,
    target_width: int,
):
    if image.width <= target_width:
        return image

    scale = (
        target_width
        / image.width
    )

    target_height = int(
        image.height
        * scale
    )

    return image.resize(
        (
            target_width,
            target_height
        )
    )


def add_caption(
    image: Image.Image,
    caption: str,
):
    caption_height = 46

    canvas = Image.new(
        'RGB',
        (
            image.width,
            image.height
            + caption_height
        ),
        'white'
    )

    canvas.paste(
        image,
        (
            0,
            caption_height
        )
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (
            8,
            12
        ),
        caption,
        fill='black'
    )

    return canvas


def make_vertical_contact_sheet(
    items,
    output_path: Path,
    target_width: int = 1400,
):
    panels = []

    for caption, path in items:
        try:
            image = (
                Image.open(path)
                .convert('RGB')
            )
        except Exception as exc:
            print(
                f'[WARN] Cannot open '
                f'{path}: {exc}'
            )
            continue

        image = resize_to_width(
            image,
            target_width
        )

        image = add_caption(
            image,
            caption
        )

        panels.append(
            image
        )

    if not panels:
        return

    width = max(
        p.width
        for p in panels
    )

    height = sum(
        p.height
        for p in panels
    )

    sheet = Image.new(
        'RGB',
        (
            width,
            height
        ),
        'white'
    )

    y = 0

    for panel in panels:
        sheet.paste(
            panel,
            (
                0,
                y
            )
        )

        y += panel.height

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sheet.save(
        output_path,
        quality=95
    )


def main():
    all_manifest_rows = []

    for dataset_name in (
        'ksdd2',
        'mtd',
        'neu',
    ):
        print(
            '\n'
            + '=' * 60
        )
        print(
            f'{dataset_name.upper()} FAILURE COLLECTION'
        )
        print(
            '=' * 60
        )

        source_root = (
            QUAL_ROOT
            / dataset_name
        )

        output_root = (
            OUT_ROOT
            / dataset_name
        )

        selected_dir = (
            output_root
            / 'selected'
        )

        figure_dir = (
            output_root
            / 'figures'
        )

        selected_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        figure_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        manifest_rows = []
        known_sheet_items = []

        for (
            sample_id,
            reason
        ) in KNOWN_CASES[
            dataset_name
        ]:
            source = find_case_visual(
                source_root,
                sample_id
            )

            if source is None:
                print(
                    f'[NOT FOUND] '
                    f'{sample_id}'
                )

                row = {
                    'dataset':
                        dataset_name,

                    'sample_id':
                        sample_id,

                    'reason':
                        reason,

                    'source_path':
                        '',

                    'copied_path':
                        '',

                    'status':
                        'not_found',
                }

                manifest_rows.append(
                    row
                )

                all_manifest_rows.append(
                    row
                )

                continue

            target_name = (
                f'{sample_id}'
                f'__{source.name}'
            )

            target = (
                selected_dir
                / target_name
            )

            shutil.copy2(
                source,
                target
            )

            print(
                f'[OK] {sample_id}'
            )
            print(
                f'  from: {source}'
            )
            print(
                f'  to:   {target}'
            )

            row = {
                'dataset':
                    dataset_name,

                'sample_id':
                    sample_id,

                'reason':
                    reason,

                'source_path':
                    str(source),

                'copied_path':
                    str(target),

                'status':
                    'found',
            }

            manifest_rows.append(
                row
            )

            all_manifest_rows.append(
                row
            )

            known_sheet_items.append(
                (
                    (
                        f'{sample_id} | '
                        f'{reason}'
                    ),
                    target
                )
            )


        make_vertical_contact_sheet(
            known_sheet_items,
            figure_dir
            / 'known_failure_contact_sheet.jpg'
        )


        broad_candidates = (
            collect_candidate_images(
                source_root,
                max_count=50
            )
        )

        broad_sheet_items = [
            (
                path.name,
                path
            )
            for path
            in broad_candidates
        ]

        make_vertical_contact_sheet(
            broad_sheet_items,
            figure_dir
            / 'all_candidate_contact_sheet.jpg'
        )


        manifest_path = (
            output_root
            / 'failure_manifest.csv'
        )

        with manifest_path.open(
            'w',
            encoding='utf-8-sig',
            newline=''
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    'dataset',
                    'sample_id',
                    'reason',
                    'source_path',
                    'copied_path',
                    'status',
                ]
            )

            writer.writeheader()
            writer.writerows(
                manifest_rows
            )

        print(
            f'Saved manifest: '
            f'{manifest_path}'
        )


    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    global_manifest = (
        OUT_ROOT
        / 'failure_manifest_all.csv'
    )

    with global_manifest.open(
        'w',
        encoding='utf-8-sig',
        newline=''
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'dataset',
                'sample_id',
                'reason',
                'source_path',
                'copied_path',
                'status',
            ]
        )

        writer.writeheader()
        writer.writerows(
            all_manifest_rows
        )

    print(
        '\n===== Finished ====='
    )
    print(
        f'Global manifest: '
        f'{global_manifest}'
    )


if __name__ == '__main__':
    main()
