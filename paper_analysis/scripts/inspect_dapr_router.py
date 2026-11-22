from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmseg.apis import init_model
from mmseg.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CONFIG = (
    ROOT
    / 'projects/industrial_defect_seg/configs/ksdd2/'
      'segformer_mit-b0_dapr_frfn_ksdd2_20k.py'
)

CHECKPOINT = (
    ROOT
    / 'work_dirs/ablation/ksdd2/'
      'segformer_b0_dapr_frfn_ksdd2_20k/'
      'best_mIoU_iter_18000.pth'
)

OUT_DIR = (
    ROOT
    / 'paper_analysis/mechanism/dapr/ksdd2'
)


def main():

    register_all_modules(
        init_default_scope=True
    )

    if not CONFIG.is_file():
        raise FileNotFoundError(
            f'Config not found:\n{CONFIG}'
        )

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f'Checkpoint not found:\n{CHECKPOINT}'
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cfg = Config.fromfile(
        str(CONFIG)
    )

    custom_imports = cfg.get(
        'custom_imports',
        None
    )

    if custom_imports:
        import_modules_from_strings(
            **custom_imports
        )

    device = (
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print(
        f'Using device: {device}'
    )

    model = init_model(
        cfg,
        str(CHECKPOINT),
        device=device
    )

    model.eval()

    head = model.decode_head

    print('\n' + '=' * 80)
    print('DECODE HEAD CLASS')
    print('=' * 80)

    print(
        type(head)
    )

    print(
        f'Module: '
        f'{type(head).__module__}'
    )

    print(
        f'Class: '
        f'{type(head).__name__}'
    )


    tree_lines = []

    print('\n' + '=' * 80)
    print('NAMED MODULES')
    print('=' * 80)

    for name, module in (
        head.named_modules()
    ):

        if name == '':
            display_name = '<decode_head>'
        else:
            display_name = name

        line = (
            f'{display_name:<70} '
            f'{type(module).__name__}'
        )

        tree_lines.append(
            line
        )

        print(
            line
        )

    (
        OUT_DIR
        / 'decode_head_module_tree.txt'
    ).write_text(
        '\n'.join(tree_lines)
        + '\n',
        encoding='utf-8'
    )


    state_lines = []

    print('\n' + '=' * 80)
    print('DECODE HEAD STATE DICT')
    print('=' * 80)

    for name, tensor in (
        head.state_dict().items()
    ):
        line = (
            f'{name:<80} '
            f'{tuple(tensor.shape)}'
        )

        state_lines.append(
            line
        )

        print(
            line
        )

    (
        OUT_DIR
        / 'decode_head_state_dict.txt'
    ).write_text(
        '\n'.join(state_lines)
        + '\n',
        encoding='utf-8'
    )


    print('\n' + '=' * 80)
    print('DECODE HEAD SOURCE')
    print('=' * 80)

    try:
        source = inspect.getsource(
            type(head)
        )

        print(
            source
        )

        (
            OUT_DIR
            / 'decode_head_source.py'
        ).write_text(
            source,
            encoding='utf-8'
        )

    except Exception as exc:
        print(
            f'Cannot inspect decode-head source: '
            f'{exc}'
        )


    keywords = (
        'dapr',
        'route',
        'router',
        'routing',
        'stat',
        'mean',
        'std',
        'weight',
        'gate',
        'partial',
        'frfn',
    )

    filtered = []

    print('\n' + '=' * 80)
    print('DAPR / ROUTING RELATED MODULES')
    print('=' * 80)

    for name, module in (
        head.named_modules()
    ):

        lower = name.lower()

        if any(
            key in lower
            for key in keywords
        ):
            line = (
                f'{name:<70} '
                f'{type(module).__name__}'
            )

            filtered.append(
                line
            )

            print(
                line
            )

    if not filtered:
        print(
            '[No names matched by keyword.]'
        )

    (
        OUT_DIR
        / 'dapr_related_modules.txt'
    ).write_text(
        '\n'.join(filtered)
        + '\n',
        encoding='utf-8'
    )

    print('\n' + '=' * 80)
    print('SAVED')
    print('=' * 80)

    print(
        OUT_DIR
        / 'decode_head_module_tree.txt'
    )

    print(
        OUT_DIR
        / 'decode_head_state_dict.txt'
    )

    print(
        OUT_DIR
        / 'decode_head_source.py'
    )

    print(
        OUT_DIR
        / 'dapr_related_modules.txt'
    )


if __name__ == '__main__':
    main()
