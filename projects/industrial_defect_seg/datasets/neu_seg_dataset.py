from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class NEUSegDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'patches', 'inclusion', 'scratches'),
        palette=[
            [0, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [255, 0, 0],
        ],
    )

    def __init__(
        self,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        **kwargs,
    ):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )
