import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import torch
from mmseg.apis import init_model


ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "DeepLabV3+-R50": {
        "config": ROOT / "projects/industrial_defect_seg/configs/ksdd2/deeplabv3plus_r50_ksdd2_20k.py",
        "checkpoint": ROOT / "work_dirs/deeplabv3plus_r50_ksdd2_20k/best_mIoU_iter_13000.pth",
    },
    "SegNeXt-T": {
        "config": ROOT / "projects/industrial_defect_seg/configs/ksdd2/segnext_t_ksdd2_20k.py",
        "checkpoint": ROOT / "work_dirs/segnext_t_ksdd2_fp32_20k/best_mIoU_iter_6500.pth",
    },
    "SegFormer-B0": {
        "config": ROOT / "projects/industrial_defect_seg/configs/ksdd2/segformer_mit-b0_ksdd2_20k.py",
        "checkpoint": ROOT / "work_dirs/segformer_b0_ksdd2_20k/best_mIoU_iter_14000.pth",
    },
    "SegFormer-B1": {
        "config": ROOT / "work_dirs/segformer_mit-b1_ksdd2_20k/segformer_mit-b1_ksdd2_20k.py",
        "checkpoint": ROOT / "work_dirs/segformer_mit-b1_ksdd2_20k/best_mIoU_iter_13000.pth",
    },
    "OCRNet-HRNet-W18": {
        "config": ROOT / "projects/industrial_defect_seg/configs/ksdd2/ocrnet_hrnet_w18_ksdd2_20k.py",
        "checkpoint": ROOT / "work_dirs/ocrnet_hrnet_w18_ksdd2_20k/best_mIoU_iter_11000.pth",
    },
    "BiSeNetV2": {
        "config": ROOT / "projects/industrial_defect_seg/configs/ksdd2/bisenetv2_ksdd2_20k.py",
        "checkpoint": ROOT / "work_dirs/bisenetv2_ksdd2_20k/best_mIoU_iter_18500.pth",
    },
    "UPerNet-Swin-T": {
        "config": ROOT / "projects/industrial_defect_seg/configs/ksdd2/upernet_swin_t_ksdd2_20k.py",
        "checkpoint": ROOT / "work_dirs/upernet_swin_t_ksdd2_20k/best_mIoU_iter_10000.pth",
    },
    "Ours": {
        "config": ROOT / "projects/industrial_defect_seg/configs/ksdd2/segformer_mit-b0_dapr_frfn_pcgrc_g001_ksdd2_20k.py",
        "checkpoint": ROOT / "work_dirs/segformer_b0_dapr_frfn_pcgrc_g001_ksdd2_20k/best_mIoU_iter_11000.pth",
    },
}


def benchmark(name, config, checkpoint, warmup, repeat):
    print()
    print("=" * 72)
    print(f"MODEL: {name}")
    print(f"CONFIG: {config}")
    print(f"CKPT:   {checkpoint}")
    print("=" * 72)

    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    model = init_model(
        str(config),
        str(checkpoint),
        device="cuda:0",
    )
    model.eval()


    if hasattr(model, "auxiliary_head"):
        model.auxiliary_head = None

    gc.collect()
    torch.cuda.empty_cache()

    x = torch.randn(
        1, 3, 256, 640,
        device="cuda:0",
        dtype=torch.float32,
    )


    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.forward(x, mode="tensor")

    torch.cuda.synchronize()


    torch.cuda.reset_peak_memory_stats()

    starts = []
    ends = []

    with torch.inference_mode():
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model.forward(x, mode="tensor")
            end.record()

            starts.append(start)
            ends.append(end)

    torch.cuda.synchronize()

    times_ms = np.array(
        [s.elapsed_time(e) for s, e in zip(starts, ends)],
        dtype=np.float64,
    )

    mean_ms = float(times_ms.mean())
    median_ms = float(np.median(times_ms))
    std_ms = float(times_ms.std(ddof=1))
    p95_ms = float(np.percentile(times_ms, 95))
    fps = 1000.0 / mean_ms

    peak_alloc_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
    )
    peak_reserved_mb = (
        torch.cuda.max_memory_reserved() / 1024**2
    )

    print(f"Mean latency   : {mean_ms:.4f} ms")
    print(f"Median latency : {median_ms:.4f} ms")
    print(f"Std latency    : {std_ms:.4f} ms")
    print(f"P95 latency    : {p95_ms:.4f} ms")
    print(f"FPS            : {fps:.2f}")
    print(f"Peak allocated : {peak_alloc_mb:.2f} MB")
    print(f"Peak reserved  : {peak_reserved_mb:.2f} MB")

    del x
    del model
    del starts
    del ends

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        "Method": name,
        "MeanLatency_ms": mean_ms,
        "MedianLatency_ms": median_ms,
        "StdLatency_ms": std_ms,
        "P95Latency_ms": p95_ms,
        "FPS": fps,
        "PeakAllocated_MB": peak_alloc_mb,
        "PeakReserved_MB": peak_reserved_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=300)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only selected model names.",
    )
    parser.add_argument(
        "--output",
        default="efficiency_runtime_8models.csv",
    )
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True


    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Input: 1x3x256x640")
    print("Precision: FP32")
    print("TF32: disabled")
    print("Warmup:", args.warmup)
    print("Repeat:", args.repeat)

    selected = (
        list(MODELS.keys())
        if not args.only
        else args.only
    )

    unknown = [x for x in selected if x not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}")

    results = []

    for name in selected:
        item = MODELS[name]
        try:
            result = benchmark(
                name,
                item["config"],
                item["checkpoint"],
                args.warmup,
                args.repeat,
            )
            results.append(result)
        except Exception as e:
            print(f"[FAILED] {name}: {type(e).__name__}: {e}")
            results.append({
                "Method": name,
                "MeanLatency_ms": "",
                "MedianLatency_ms": "",
                "StdLatency_ms": "",
                "P95Latency_ms": "",
                "FPS": "",
                "PeakAllocated_MB": "",
                "PeakReserved_MB": "",
                "Error": f"{type(e).__name__}: {e}",
            })

    fieldnames = [
        "Method",
        "MeanLatency_ms",
        "MedianLatency_ms",
        "StdLatency_ms",
        "P95Latency_ms",
        "FPS",
        "PeakAllocated_MB",
        "PeakReserved_MB",
        "Error",
    ]

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for r in results:
            if "Error" not in r:
                r["Error"] = ""
            writer.writerow(r)

    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    for r in results:
        if r.get("FPS", "") == "":
            print(f"{r['Method']:<28} FAILED")
        else:
            print(
                f"{r['Method']:<28} "
                f"{r['FPS']:>8.2f} FPS | "
                f"{r['MeanLatency_ms']:>8.4f} ms | "
                f"{r['PeakAllocated_MB']:>9.2f} MB"
            )

    print()
    print("CSV:", args.output)


if __name__ == "__main__":
    main()
