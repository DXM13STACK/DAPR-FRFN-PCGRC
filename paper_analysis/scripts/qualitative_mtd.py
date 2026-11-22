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

OUT_ROOT = ROOT / "paper_analysis/qualitative/mtd"
PRED_ROOT = OUT_ROOT / "pred_masks"
CANDIDATE_DIR = OUT_ROOT / "candidates"

MODELS = {
    "b0": {
        "display": "SegFormer-B0",
        "config": ROOT / (
            "projects/industrial_defect_seg/configs/mtd/"
            "segformer_mit-b0_mtd_binary_20k.py"
        ),
        "checkpoint": ROOT / (
            "work_dirs/segformer_b0_mtd_binary_20k_bs4/"
            "best_mIoU_iter_3000.pth"
        ),
    },
    "segnext": {
        "display": "SegNeXt-T",
        "config": ROOT / (
            "projects/industrial_defect_seg/configs/mtd/"
            "segnext_t_mtd_20k.py"
        ),
        "checkpoint": ROOT / (
            "work_dirs/segnext_t_mtd_20k/"
            "best_mIoU_iter_9500.pth"
        ),
    },
    "ours": {
        "display": "Ours",
        "config": ROOT / (
            "projects/industrial_defect_seg/configs/mtd/"
            "segformer_mit-b0_dapr_frfn_pcgrc_g001_mtd_binary_20k_bs4.py"
        ),
        "checkpoint": ROOT / (
            "work_dirs/segformer_b0_dapr_frfn_pcgrc_g001_mtd_binary_20k_bs4/"
            "best_mIoU_iter_9500.pth"
        ),
    },
}


def ensure_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    for key in MODELS:
        (PRED_ROOT / key).mkdir(parents=True, exist_ok=True)


def build_test_dataset():
    cfg_path = MODELS["b0"]["config"]
    cfg = Config.fromfile(str(cfg_path))

    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg.custom_imports)

    register_all_modules(init_default_scope=True)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    dataset.full_init()

    return dataset


def infer_defect_label(data_list):
    """Infer the raw positive label from the GT masks."""
    observed = set()

    for info in data_list:
        seg_path = Path(info["seg_map_path"])
        arr = np.asarray(Image.open(seg_path))
        observed.update(np.unique(arr).tolist())

    print("Observed raw GT values:", sorted(observed))


    if 1 in observed:
        return 1

    if 255 in observed and observed.issubset({0, 255}):
        return 255

    raise RuntimeError(
        f"Cannot infer defect label from GT values: {sorted(observed)}"
    )


def read_gt(path, defect_label):
    arr = np.asarray(Image.open(path))

    if arr.ndim == 3:
        arr = arr[..., 0]

    return (arr == defect_label).astype(np.uint8)


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def resize_binary(mask, size_hw):
    h, w = size_hw

    if mask.shape == (h, w):
        return mask.astype(np.uint8)

    pil = Image.fromarray((mask.astype(np.uint8) * 255))
    pil = pil.resize((w, h), Image.Resampling.NEAREST)

    return (np.asarray(pil) > 0).astype(np.uint8)


def extract_pred(result):
    pred = result.pred_sem_seg.data

    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()

    pred = np.squeeze(pred)

    return (pred > 0).astype(np.uint8)


def binary_metrics(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    tp = inter
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    if union == 0:
        iou = 1.0
    else:
        iou = inter / union

    denom = 2 * tp + fp + fn

    if denom == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / denom

    return {
        "iou": float(iou),
        "dice": float(dice),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def save_mask(mask, path):
    Image.fromarray(
        (mask.astype(np.uint8) * 255)
    ).save(path)


def load_or_predict(model, img_path, out_path, target_hw):
    if out_path.exists():
        cached = np.asarray(Image.open(out_path))
        cached = (cached > 0).astype(np.uint8)
        return resize_binary(cached, target_hw)

    result = inference_model(model, str(img_path))
    pred = extract_pred(result)
    pred = resize_binary(pred, target_hw)

    save_mask(pred, out_path)

    return pred


def add_reason(reason_map, sample_id, reason):
    reason_map[sample_id].add(reason)


def choose_candidates(records):
    positive = [r for r in records if r["gt_pixels"] > 0]

    if not positive:
        print(
            "No positive MTD samples in the current subset; "
            "candidate selection is skipped."
        )
        return [], 0.0, 0.0

    gt_sizes = np.array(
        [r["gt_pixels"] for r in positive],
        dtype=np.float64,
    )

    q1, q2 = np.quantile(gt_sizes, [1 / 3, 2 / 3])

    for r in positive:
        if r["gt_pixels"] <= q1:
            r["scale_group"] = "SMALL"
        elif r["gt_pixels"] <= q2:
            r["scale_group"] = "MEDIUM"
        else:
            r["scale_group"] = "LARGE"

    print()
    print("Positive GT samples:", len(positive))
    print(f"Scale thresholds: Q1={q1:.2f}, Q2={q2:.2f}")

    reasons = defaultdict(set)


    for r in sorted(
        positive,
        key=lambda x: x["delta_ours_b0"],
        reverse=True,
    )[:6]:
        add_reason(
            reasons,
            r["sample_id"],
            "top_gain_vs_b0",
        )


    for r in sorted(
        positive,
        key=lambda x: x["delta_ours_segnext"],
        reverse=True,
    )[:6]:
        add_reason(
            reasons,
            r["sample_id"],
            "top_gain_vs_segnext",
        )


    for r in sorted(
        positive,
        key=lambda x: x["delta_ours_b0"],
    )[:4]:
        add_reason(
            reasons,
            r["sample_id"],
            "regression_vs_b0",
        )


    for r in sorted(
        positive,
        key=lambda x: x["ours_iou"],
    )[:4]:
        add_reason(
            reasons,
            r["sample_id"],
            "hard_case",
        )


    for group in ["SMALL", "MEDIUM", "LARGE"]:
        subset = [
            r for r in positive
            if r["scale_group"] == group
        ]

        if not subset:
            continue

        target = np.median(
            [r["gt_pixels"] for r in subset]
        )

        representative = sorted(
            subset,
            key=lambda x: abs(x["gt_pixels"] - target),
        )[:2]

        for r in representative:
            add_reason(
                reasons,
                r["sample_id"],
                f"{group.lower()}_representative",
            )


        best_gain = max(
            subset,
            key=lambda x: x["delta_ours_b0"],
        )
        add_reason(
            reasons,
            best_gain["sample_id"],
            f"{group.lower()}_gain",
        )

    selected = [
        r for r in positive
        if r["sample_id"] in reasons
    ]

    for r in selected:
        r["candidate_reasons"] = ";".join(
            sorted(reasons[r["sample_id"]])
        )

    selected = sorted(
        selected,
        key=lambda x: (
            x["scale_group"],
            -x["delta_ours_b0"],
        ),
    )

    return selected, q1, q2


def draw_candidate(record):
    img = read_rgb(record["img_path"])
    gt = read_gt(
        record["seg_map_path"],
        record["defect_label"],
    )

    h, w = gt.shape
    if img.shape[:2] != (h, w):
        img = np.asarray(
            Image.fromarray(img).resize(
                (w, h),
                Image.Resampling.BILINEAR,
            )
        )

    masks = {
        key: (
            np.asarray(
                Image.open(record[f"{key}_pred_path"])
            ) > 0
        ).astype(np.uint8)
        for key in MODELS
    }

    for key in masks:
        masks[key] = resize_binary(
            masks[key],
            (h, w),
        )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(18, 4.2),
    )

    axes[0].imshow(img)
    axes[0].set_title("Input")

    axes[1].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(
        f"GT\npixels={record['gt_pixels']}"
    )

    axes[2].imshow(
        masks["b0"],
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        f"SegFormer-B0\n"
        f"IoU={record['b0_iou']:.3f}"
    )

    axes[3].imshow(
        masks["segnext"],
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[3].set_title(
        f"SegNeXt-T\n"
        f"IoU={record['segnext_iou']:.3f}"
    )

    axes[4].imshow(
        masks["ours"],
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[4].set_title(
        f"Ours\n"
        f"IoU={record['ours_iou']:.3f}"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{record['sample_id']} | "
        f"{record['scale_group']} | "
        f"ΔOurs-B0={record['delta_ours_b0']:+.3f} | "
        f"ΔOurs-SegNeXt={record['delta_ours_segnext']:+.3f}\n"
        f"{record['candidate_reasons']}",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = (
        CANDIDATE_DIR /
        f"{record['sample_id']}.png"
    )

    fig.savefig(
        out_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def write_metrics_csv(records, path):
    fields = [
        "sample_id",
        "img_path",
        "seg_map_path",
        "gt_pixels",
        "gt_ratio",
        "positive_gt",
        "scale_group",
        "b0_iou",
        "b0_dice",
        "b0_fp",
        "b0_fn",
        "segnext_iou",
        "segnext_dice",
        "segnext_fp",
        "segnext_fn",
        "ours_iou",
        "ours_dice",
        "ours_fp",
        "ours_fn",
        "delta_ours_b0",
        "delta_ours_segnext",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
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
        "scale_group",
        "gt_pixels",
        "gt_ratio",
        "b0_iou",
        "segnext_iou",
        "ours_iou",
        "delta_ours_b0",
        "delta_ours_segnext",
        "candidate_reasons",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
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
        default=None,
        help=(
            "Optional debug limit. "
            "Do not use for the formal full run."
        ),
    )

    args = parser.parse_args()

    ensure_dirs()

    print("===== BUILD MTD TEST DATASET =====")
    dataset = build_test_dataset()


    all_data_list = [
        dataset.get_data_info(i)
        for i in range(len(dataset))
    ]

    all_data_list = sorted(
        all_data_list,
        key=lambda x: Path(x["img_path"]).name,
    )

    print("Full test images:", len(all_data_list))


    defect_label = infer_defect_label(all_data_list)
    print("Using defect label:", defect_label)

    if args.limit is not None:
        data_list = all_data_list[:args.limit]
    else:
        data_list = all_data_list

    print("Selected images:", len(data_list))

    print()
    print("===== LOAD MODELS =====")

    models = {}

    for key, info in MODELS.items():
        print(
            f"Loading {info['display']} ..."
        )

        models[key] = init_model(
            str(info["config"]),
            str(info["checkpoint"]),
            device="cuda:0",
        )
        models[key].eval()

    records = []

    print()
    print("===== RUN FULL TEST INFERENCE =====")

    for idx, info in enumerate(data_list, start=1):
        img_path = Path(info["img_path"])
        seg_path = Path(info["seg_map_path"])

        sample_id = img_path.stem

        gt = read_gt(
            seg_path,
            defect_label,
        )

        h, w = gt.shape

        gt_pixels = int(gt.sum())
        gt_ratio = float(
            gt_pixels / gt.size
        )

        record = {
            "sample_id": sample_id,
            "img_path": str(img_path),
            "seg_map_path": str(seg_path),
            "defect_label": defect_label,
            "gt_pixels": gt_pixels,
            "gt_ratio": gt_ratio,
            "positive_gt": int(gt_pixels > 0),
            "scale_group": (
                "NEGATIVE"
                if gt_pixels == 0
                else ""
            ),
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
                (h, w),
            )

            m = binary_metrics(gt, pred)

            record[f"{key}_iou"] = m["iou"]
            record[f"{key}_dice"] = m["dice"]
            record[f"{key}_fp"] = m["fp"]
            record[f"{key}_fn"] = m["fn"]
            record[f"{key}_pred_path"] = str(
                pred_path
            )

        record["delta_ours_b0"] = (
            record["ours_iou"]
            - record["b0_iou"]
        )

        record["delta_ours_segnext"] = (
            record["ours_iou"]
            - record["segnext_iou"]
        )

        records.append(record)

        if (
            idx == 1
            or idx % 50 == 0
            or idx == len(data_list)
        ):
            print(
                f"[{idx:4d}/{len(data_list)}] "
                f"{sample_id}"
            )

    print()
    print("===== SELECT CANDIDATES =====")

    candidates, q1, q2 = choose_candidates(
        records
    )


    for r in records:
        if r["gt_pixels"] == 0:
            r["scale_group"] = "NEGATIVE"
        elif r["gt_pixels"] <= q1:
            r["scale_group"] = "SMALL"
        elif r["gt_pixels"] <= q2:
            r["scale_group"] = "MEDIUM"
        else:
            r["scale_group"] = "LARGE"

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
        metrics_csv,
    )

    write_candidate_csv(
        candidates,
        candidate_csv,
    )


    for p in CANDIDATE_DIR.glob("*.png"):
        p.unlink()

    for idx, r in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"Rendering candidate "
            f"{idx}/{len(candidates)}: "
            f"{r['sample_id']}"
        )
        draw_candidate(r)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print("Total test images :", len(records))
    print(
        "Positive samples  :",
        sum(r["positive_gt"] for r in records),
    )
    print(
        "Negative samples  :",
        sum(
            1 - r["positive_gt"]
            for r in records
        ),
    )
    print("Candidate count   :", len(candidates))
    print(
        f"Scale thresholds  : "
        f"Q1={q1:.2f}, Q2={q2:.2f}"
    )
    print("Metrics CSV       :", metrics_csv)
    print("Candidate CSV     :", candidate_csv)
    print("Candidate figures :", CANDIDATE_DIR)


    del models
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
