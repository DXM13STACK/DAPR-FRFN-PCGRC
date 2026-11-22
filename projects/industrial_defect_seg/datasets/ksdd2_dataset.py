from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class KSDD2Dataset(BaseSegDataset):

    METAINFO = dict(

        classes=(
            'background',
            'defect'
        ),

        palette=[
            [0, 0, 0],
            [255, 255, 255]
        ]
    )

    def __init__(self, **kwargs):

        super().__init__(

            img_suffix='.png',

            seg_map_suffix='.png',

            reduce_zero_label=False,

            **kwargs
        )
