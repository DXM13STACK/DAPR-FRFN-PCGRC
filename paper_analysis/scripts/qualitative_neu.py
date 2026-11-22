import argparse
import csv
import gc
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmseg.apis import init_model, inference_model
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[2]

OUT_ROOT = ROOT / "paper_analysis/qualitative/neu"
PRED_ROOT = OUT_ROOT / "pred_masks"
CANDIDATE_DIR = OUT_ROOT / "candidates"

CLASS_NAMES = [
    "background",
    "patches",
    "inclusion",
    "scratches",
]

CLASS_IDS = {
    "background": 0,
    "patches": 1,
    "inclusion": 2,
    "scratches": 3,
}

MODELS = {
    "b0": {
        "display": "SegFormer-B0",
        "config": ROOT / (
            "work_dirs/segformer_b0_neu_seg_20k/"
            "segformer_mit-b0_neu_seg_20k.py"
        ),
        "checkpoint": ROOT / (
            "work_dirs/segformer_b0_neu_seg_20k/"
            "best_mIoU_iter_20000.pth"
        ),
    },
    "b1": {
        "display": "SegFormer-B1",
        "config": ROOT / (
            "projects/industrial_defect_seg/configs/neu/"
            "segformer_mit-b1_neu_seg_20k.py"
        ),
        "checkpoint": ROOT / (
            "work_dirs/segformer_mit-b1_neu_seg_20k/"
            "best_mIoU_iter_20000.pth"
        ),
    },
    "ours": {
        "display": "Ours",
        "config": ROOT / (
            "projects/industrial_defect_seg/configs/neu/"
            "segformer_mit-b0_dapr_frfn_pcgrc_g001_neu_seg_20k.py"
        ),
        "checkpoint": ROOT / (
            "work_dirs/"
            "segformer_b0_dapr_frfn_pcgrc_g001_neu_seg_20k/"
            "best_mIoU_iter_20000.pth"
        ),
    },
}


def ensure_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    for key in MODELS:
        (PRED_ROOT / key).mkdir(
            parents=True,
            exist_ok=True
        )


def build_test_dataset():
    cfg = Config.fromfile(
        str(MODELS["b0"]["config"])
    )

    if cfg.get("custom_imports", None):
        import_modules_from_strings(
            **cfg.custom_imports
        )

    register_all_modules(
        init_default_scope=True
    )

    dataset = DATASETS.build(
        cfg.test_dataloader.dataset
    )
    dataset.full_init()

    return dataset


def read_gt(path):
    arr = np.asarray(Image.open(path))

    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr.astype(np.uint8)


def read_rgb(path):
    return np.asarray(
        Image.open(path).convert("RGB")
    )


def extract_pred(result):
    pred = result.pred_sem_seg.data

    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()

    pred = np.squeeze(pred)

    return pred.astype(np.uint8)


def resize_multiclass(mask, target_hw):
    h, w = target_hw

    if mask.shape == (h, w):
        return mask.astype(np.uint8)

    im = Image.fromarray(
        mask.astype(np.uint8)
    )

    im = im.resize(
        (w, h),
        Image.Resampling.NEAREST
    )

    return np.asarray(im).astype(np.uint8)


def save_mask(mask, path):
    Image.fromarray(
        mask.astype(np.uint8)
    ).save(path)


def load_or_predict(
    model,
    img_path,
    pred_path,
    target_hw,
):
    if pred_path.exists():
        pred = np.asarray(
            Image.open(pred_path)
        )
        return resize_multiclass(
            pred,
            target_hw
        )

    result = inference_model(
        model,
        str(img_path)
    )

    pred = extract_pred(result)
    pred = resize_multiclass(
        pred,
        target_hw
    )

    save_mask(
        pred,
        pred_path
    )

    return pred


def class_iou(gt, pred, class_id):
    g = (gt == class_id)
    p = (pred == class_id)

    inter = np.logical_and(g, p).sum()
    union = np.logical_or(g, p).sum()


    if union == 0:
        return np.nan

    return float(inter / union)


def evaluate_sample(gt, pred):
    ious = {}

    for cid, cname in enumerate(CLASS_NAMES):
        ious[cname] = class_iou(
            gt,
            pred,
            cid
        )

    present_fg = []

    for cid in [1, 2, 3]:
        if np.any(gt == cid):
            v = ious[CLASS_NAMES[cid]]

            if not np.isnan(v):
                present_fg.append(v)

    if present_fg:
        fg_miou = float(
            np.mean(present_fg)
        )
    else:
        fg_miou = np.nan

    present_all = []

    for cid in range(4):
        if np.any(gt == cid):
            v = ious[CLASS_NAMES[cid]]

            if not np.isnan(v):
                present_all.append(v)

    sample_miou = (
        float(np.mean(present_all))
        if present_all
        else np.nan
    )

    return ious, fg_miou, sample_miou


def dominant_fg_class(gt):
    counts = {}

    for cid in [1, 2, 3]:
        counts[cid] = int(
            np.sum(gt == cid)
        )

    best_id = max(
        counts,
        key=counts.get
    )

    if counts[best_id] == 0:
        return "background"

    return CLASS_NAMES[best_id]


def foreground_pixels(gt):
    return int(
        np.sum(gt > 0)
    )


def add_reason(reason_map, sid, reason):
    reason_map[sid].add(reason)


def valid_delta(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan

    return float(a - b)


def choose_candidates(records):
    valid = [
        r for r in records
        if r["gt_fg_pixels"] > 0
        and not np.isnan(r["ours_fg_miou"])
        and not np.isnan(r["b0_fg_miou"])
        and not np.isnan(r["b1_fg_miou"])
    ]

    reasons = defaultdict(set)


    for r in sorted(
        valid,
        key=lambda x: x["delta_ours_b0"],
        reverse=True
    )[:6]:
        add_reason(
            reasons,
            r["sample_id"],
            "top_gain_vs_b0"
        )


    for r in sorted(
        valid,
        key=lambda x: x["delta_ours_b1"],
        reverse=True
    )[:4]:
        add_reason(
            reasons,
            r["sample_id"],
            "top_gain_vs_b1"
        )


    for r in sorted(
        valid,
        key=lambda x: x["delta_ours_b1"]
    )[:3]:
        add_reason(
            reasons,
            r["sample_id"],
            "b1_better_case"
        )


    for r in sorted(
        valid,
        key=lambda x: x["ours_fg_miou"]
    )[:3]:
        add_reason(
            reasons,
            r["sample_id"],
            "hard_case"
        )


    for cname in [
        "patches",
        "inclusion",
        "scratches",
    ]:
        subset = [
            r for r in valid
            if r["dominant_class"] == cname
        ]

        if not subset:
            continue


        best_gain = max(
            subset,
            key=lambda x: x["delta_ours_b0"]
        )

        add_reason(
            reasons,
            best_gain["sample_id"],
            f"{cname}_gain"
        )


        med = np.median(
            [x["ours_fg_miou"] for x in subset]
        )

        representative = min(
            subset,
            key=lambda x:
            abs(x["ours_fg_miou"] - med)
        )

        add_reason(
            reasons,
            representative["sample_id"],
            f"{cname}_representative"
        )

    selected = [
        r for r in valid
        if r["sample_id"] in reasons
    ]

    for r in selected:
        r["candidate_reasons"] = ";".join(
            sorted(
                reasons[r["sample_id"]]
            )
        )

    selected = sorted(
        selected,
        key=lambda x: (
            x["dominant_class"],
            -x["delta_ours_b0"]
        )
    )

    return selected


PALETTE = np.array([
    [0, 0, 0],
    [220, 60, 60],
    [60, 180, 75],
    [65, 105, 225],
], dtype=np.uint8)


def colorize(mask):
    mask = np.clip(
        mask,
        0,
        len(PALETTE) - 1
    )

    return PALETTE[mask]


def draw_candidate(record):
    img = read_rgb(
        record["img_path"]
    )

    gt = read_gt(
        record["seg_map_path"]
    )

    h, w = gt.shape

    if img.shape[:2] != (h, w):
        img = np.asarray(
            Image.fromarray(img).resize(
                (w, h),
                Image.Resampling.BILINEAR
            )
        )

    masks = {}

    for key in MODELS:
        mask = np.asarray(
            Image.open(
                record[f"{key}_pred_path"]
            )
        )

        masks[key] = resize_multiclass(
            mask,
            (h, w)
        )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(16, 3.8)
    )

    axes[0].imshow(img)
    axes[0].set_title("Input")

    axes[1].imshow(colorize(gt))
    axes[1].set_title(
        f"GT\n"
        f"{record['dominant_class']} | "
        f"fg={record['gt_fg_pixels']}"
    )

    axes[2].imshow(
        colorize(masks["b0"])
    )
    axes[2].set_title(
        "SegFormer-B0\n"
        f"fg-mIoU="
        f"{record['b0_fg_miou']:.3f}"
    )

    axes[3].imshow(
        colorize(masks["b1"])
    )
    axes[3].set_title(
        "SegFormer-B1\n"
        f"fg-mIoU="
        f"{record['b1_fg_miou']:.3f}"
    )

    axes[4].imshow(
        colorize(masks["ours"])
    )
    axes[4].set_title(
        "Ours\n"
        f"fg-mIoU="
        f"{record['ours_fg_miou']:.3f}"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{record['sample_id']} | "
        f"class={record['dominant_class']} | "
        f"ΔOurs-B0="
        f"{record['delta_ours_b0']:+.3f} | "
        f"ΔOurs-B1="
        f"{record['delta_ours_b1']:+.3f}\n"
        f"{record['candidate_reasons']}",
        fontsize=10
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.87]
    )

    out = (
        CANDIDATE_DIR /
        f"{record['sample_id']}.png"
    )

    fig.savefig(
        out,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


def write_metrics_csv(records, path):
    fields = [
        "sample_id",
        "img_path",
        "seg_map_path",
        "dominant_class",
        "gt_fg_pixels",
        "gt_patches_pixels",
        "gt_inclusion_pixels",
        "gt_scratches_pixels",

        "b0_fg_miou",
        "b1_fg_miou",
        "ours_fg_miou",

        "b0_sample_miou",
        "b1_sample_miou",
        "ours_sample_miou",

        "b0_patches_iou",
        "b0_inclusion_iou",
        "b0_scratches_iou",

        "b1_patches_iou",
        "b1_inclusion_iou",
        "b1_scratches_iou",

        "ours_patches_iou",
        "ours_inclusion_iou",
        "ours_scratches_iou",

        "delta_ours_b0",
        "delta_ours_b1",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )
        writer.writeheader()

        for r in records:
            writer.writerow({
                k: r.get(k, "")
                for k in fields
            })


def write_candidate_csv(records, path):
    fields = [
        "sample_id",
        "dominant_class",
        "gt_fg_pixels",
        "b0_fg_miou",
        "b1_fg_miou",
        "ours_fg_miou",
        "delta_ours_b0",
        "delta_ours_b1",
        "candidate_reasons",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )
        writer.writeheader()

        for r in records:
            writer.writerow({
                k: r.get(k, "")
                for k in fields
            })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = parser.parse_args()

    ensure_dirs()

    print(
        "===== BUILD NEU TEST DATASET ====="
    )

    dataset = build_test_dataset()


    all_data_list = [
        dataset.get_data_info(i)
        for i in range(len(dataset))
    ]

    all_data_list = sorted(
        all_data_list,
        key=lambda x:
        Path(x["img_path"]).name
    )

    print(
        "Full test images:",
        len(all_data_list)
    )

    if args.limit is not None:
        data_list = all_data_list[
            :args.limit
        ]
    else:
        data_list = all_data_list

    print(
        "Selected images:",
        len(data_list)
    )

    print()
    print("===== LOAD MODELS =====")

    models = {}

    for key, info in MODELS.items():
        print(
            f"Loading "
            f"{info['display']} ..."
        )

        if not info["config"].is_file():
            raise FileNotFoundError(
                info["config"]
            )

        if not info["checkpoint"].is_file():
            raise FileNotFoundError(
                info["checkpoint"]
            )

        models[key] = init_model(
            str(info["config"]),
            str(info["checkpoint"]),
            device="cuda:0"
        )

        models[key].eval()

    records = []

    print()
    print(
        "===== RUN NEU TEST INFERENCE ====="
    )

    for idx, info in enumerate(
        data_list,
        start=1
    ):
        img_path = Path(
            info["img_path"]
        )

        seg_path = Path(
            info["seg_map_path"]
        )

        sample_id = img_path.stem

        gt = read_gt(seg_path)
        h, w = gt.shape

        record = {
            "sample_id": sample_id,
            "img_path": str(img_path),
            "seg_map_path": str(seg_path),
            "dominant_class":
                dominant_fg_class(gt),
            "gt_fg_pixels":
                foreground_pixels(gt),
            "gt_patches_pixels":
                int(np.sum(gt == 1)),
            "gt_inclusion_pixels":
                int(np.sum(gt == 2)),
            "gt_scratches_pixels":
                int(np.sum(gt == 3)),
        }

        for key, model in models.items():
            pred_path = (
                PRED_ROOT /
                key /
                f"{sample_id}.png"
            )

            pred = load_or_predict(
                model,
                img_path,
                pred_path,
                (h, w)
            )

            ious, fg_miou, sample_miou = (
                evaluate_sample(
                    gt,
                    pred
                )
            )

            record[f"{key}_fg_miou"] = (
                fg_miou
            )

            record[
                f"{key}_sample_miou"
            ] = sample_miou

            for cname in [
                "patches",
                "inclusion",
                "scratches",
            ]:
                record[
                    f"{key}_{cname}_iou"
                ] = ious[cname]

            record[
                f"{key}_pred_path"
            ] = str(pred_path)

        record["delta_ours_b0"] = (
            valid_delta(
                record["ours_fg_miou"],
                record["b0_fg_miou"]
            )
        )

        record["delta_ours_b1"] = (
            valid_delta(
                record["ours_fg_miou"],
                record["b1_fg_miou"]
            )
        )

        records.append(record)

        if (
            idx == 1
            or idx % 50 == 0
            or idx == len(data_list)
        ):
            print(
                f"[{idx:4d}/"
                f"{len(data_list)}] "
                f"{sample_id}"
            )

    print()
    print(
        "===== SELECT CANDIDATES ====="
    )

    candidates = choose_candidates(
        records
    )

    metrics_csv = (
        OUT_ROOT /
        "qualitative_metrics.csv"
    )

    candidate_csv = (
        OUT_ROOT /
        "candidate_selection.csv"
    )

    write_metrics_csv(
        records,
        metrics_csv
    )

    write_candidate_csv(
        candidates,
        candidate_csv
    )

    for p in CANDIDATE_DIR.glob(
        "*.png"
    ):
        p.unlink()

    for idx, r in enumerate(
        candidates,
        start=1
    ):
        print(
            f"Rendering candidate "
            f"{idx}/"
            f"{len(candidates)}: "
            f"{r['sample_id']}"
        )

        draw_candidate(r)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        "Total test images :",
        len(records)
    )

    for cname in [
        "patches",
        "inclusion",
        "scratches",
    ]:
        n = sum(
            r["dominant_class"] == cname
            for r in records
        )

        print(
            f"Dominant {cname:<10}:",
            n
        )

    print(
        "Candidate count   :",
        len(candidates)
    )

    print(
        "Metrics CSV       :",
        metrics_csv
    )

    print(
        "Candidate CSV     :",
        candidate_csv
    )

    print(
        "Candidate figures :",
        CANDIDATE_DIR
    )

    del models
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
