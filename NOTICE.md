# Notices and attribution

This repository contains project-specific extensions built for MMSegmentation 1.2.2. The copied MMSegmentation entry points and base configurations are distributed under the Apache License 2.0; see `LICENSE`.

The FRFN implementation was adapted and modified for industrial surface defect segmentation from the feature-refinement design and the Apache-2.0-licensed AST implementation associated with:

> Shihao Zhou, Duosheng Chen, Jinshan Pan, Jinglei Shi, and Jufeng Yang, “Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration,” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2952–2963, 2024. DOI: [10.1109/CVPR52733.2024.00285](https://doi.org/10.1109/CVPR52733.2024.00285).

- Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.html
- Upstream implementation: https://github.com/joshyZhou/AST
- Upstream license: Apache License 2.0

The adapted code has been modified for the architecture and experiments reported in the accompanying manuscript. Additional third-party model implementations are not redistributed; only experiment configurations referring to official MMSegmentation components are included.
