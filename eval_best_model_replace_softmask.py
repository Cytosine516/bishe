import os
import gc
import json
import math
import random
import argparse
import importlib.util
import sys
from collections import defaultdict

# 保证优先从脚本所在目录 / 当前工作目录导入项目模块，避免找不到 network.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()
for _p in [
    SCRIPT_DIR,
    CWD,
    os.path.join(SCRIPT_DIR, 'models'),
    os.path.join(CWD, 'models'),
    os.path.join(SCRIPT_DIR, 'utils'),
    os.path.join(CWD, 'utils'),
]:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def _load_module_from_candidates(module_name, filenames):
    candidates = []
    for name in filenames:
        candidates.extend([
            os.path.join(SCRIPT_DIR, name),
            os.path.join(CWD, name),
        ])
    tried = []
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module, path
        tried.append(path)
    raise FileNotFoundError(
        f"未找到 {filenames}。已尝试路径: {tried}"
    )


_config_module, _config_path = _load_module_from_candidates('project_config', ['config.py'])
_network_module, _network_path = _load_module_from_candidates('project_network', ['network.py', os.path.join('models', 'network.py')])
_metrics_module, _metrics_path = _load_module_from_candidates('project_util_metrics', ['util_metrics.py', os.path.join('utils', 'util_metrics.py')])

Config = _config_module.Config
MS2TAN = _network_module.MS2TAN
masked_psnr_cal = _metrics_module.masked_psnr_cal
masked_sam_cal = _metrics_module.masked_sam_cal
masked_ssim_cal = _metrics_module.masked_ssim_cal
masked_rmse_cal = _metrics_module.masked_rmse_cal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_softmask_dataset_class():
    """优先加载同目录下的 dataset_softmask.py。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "dataset_softmask.py"),
        os.path.join(os.getcwd(), "dataset_softmask.py"),
    ]

    dataset_path = None
    for p in candidates:
        if os.path.isfile(p):
            dataset_path = p
            break

    if dataset_path is None:
        raise FileNotFoundError(
            "未找到 dataset_softmask.py。请把本脚本和 dataset_softmask.py 放在同一目录，"
            "或者把 dataset_softmask.py 放到当前工作目录。"
        )

    spec = importlib.util.spec_from_file_location("dataset_softmask", dataset_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TimeSeriesDataset, dataset_path


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        state_dict = checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected


def calc_mean_face(X, mask):
    """与 network.py 保持一致。X:[B,T,C,H,W], mask:[B,T,1,H,W]"""
    eps = 1e-6
    mask_exp = mask.expand_as(X)
    sum_x = torch.sum(X * mask_exp, dim=1, keepdim=True)
    sum_m = torch.sum(mask_exp, dim=1, keepdim=True)
    mean_face = sum_x / (sum_m + eps)
    valid_pixels = (sum_m > 0).float()
    mean_face = mean_face * valid_pixels
    return mean_face.expand_as(X)


def make_binary_ring(mask_4d: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """mask_4d: [N,1,H,W] in {0,1}"""
    if radius <= 0:
        return torch.zeros_like(mask_4d)
    k = 2 * radius + 1
    dilated = F.max_pool2d(mask_4d, kernel_size=k, stride=1, padding=radius)
    ring = (dilated - mask_4d).clamp(0, 1)
    return ring


def metric_or_nan(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    """返回单样本指标；mask 可为软权重。pred/gt:[1,C,H,W], mask:[1,1,H,W]"""
    mask_sum = float(mask.sum().item())
    if mask_sum <= 1e-6:
        return {
            "psnr": np.nan,
            "ssim": np.nan,
            "sam": np.nan,
            "rmse": np.nan,
            "weight_sum": 0.0,
        }

    return {
        "psnr": float(masked_psnr_cal(pred, gt, mask).item()),
        "ssim": float(masked_ssim_cal(pred, gt, mask).item()),
        "sam": float(masked_sam_cal(pred, gt, mask).item()),
        "rmse": float(masked_rmse_cal(pred, gt, mask).item()),
        "weight_sum": mask_sum,
    }


def summarize_metric_dicts(records, prefix: str):
    out = {}
    for key in ["psnr", "ssim", "sam", "rmse"]:
        vals = [r[f"{prefix}_{key}"] for r in records if not pd.isna(r[f"{prefix}_{key}"])]
        out[f"{prefix}_{key}"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
    return out


def composite_index(hole_psnr, hole_ssim, hole_sam, seam_psnr, seam_ssim, seam_sam):
    """
    给排序/挑模型用的相对指数，不替代原始指标。
    主要看洞区，其次看边界。
    """
    def safe(v, default=0.0):
        return default if pd.isna(v) else float(v)

    hole_score = safe(hole_psnr) + 20.0 * safe(hole_ssim) - 2.0 * safe(hole_sam)
    seam_score = safe(seam_psnr) + 20.0 * safe(seam_ssim) - 2.0 * safe(seam_sam)
    return 0.7 * hole_score + 0.3 * seam_score


def to_rgb_img(tensor_5d, b_idx, t_idx):
    return tensor_5d[b_idx, t_idx, [3, 2, 1]].detach().cpu().numpy().transpose(1, 2, 0)


def to_detection_vis(scl_mask, custom_cloud, custom_shadow, b_idx, t_idx):
    scl_np = scl_mask[b_idx, t_idx, 0].detach().cpu().numpy()
    cloud_np = custom_cloud[b_idx, t_idx, 0].detach().cpu().numpy()
    shadow_np = custom_shadow[b_idx, t_idx, 0].detach().cpu().numpy()

    vis = np.ones((scl_np.shape[0], scl_np.shape[1], 3), dtype=np.float32)
    vis[shadow_np > 0.5] = [1.0, 1.0, 0.0]
    vis[cloud_np > 0.5] = [0.0, 1.0, 0.0]
    vis[scl_np > 0.5] = [1.0, 0.0, 0.0]
    return vis


def to_art_vis(art_mask, b_idx, t_idx):
    art_np = art_mask[b_idx, t_idx, 0].detach().cpu().numpy()
    vis = np.ones((art_np.shape[0], art_np.shape[1], 3), dtype=np.float32)
    vis[art_np > 0.5] = [0.0, 0.0, 0.0]
    return vis


def add_target_border(ax, is_target: bool):
    if not is_target:
        return
    for spine in ax.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(3.0)
        spine.set_visible(True)


def save_preview(
    X,
    raw_out,
    replace_out,
    gt,
    obs_mask,
    art_mask,
    scl_mask,
    custom_cloud,
    custom_shadow,
    mean_face,
    target_idx,
    tag,
    path,
    preview_note=None,
):
    os.makedirs(path, exist_ok=True)

    valid_indices = torch.nonzero(art_mask.sum(dim=(2, 3, 4)) > 0)
    b_idx = valid_indices[0][0].item() if len(valid_indices) > 0 else 0
    T = X.shape[1]
    scale = 1.0
    target_t = int(target_idx[b_idx].item())

    obs_mask_np = obs_mask[b_idx, :, 0].detach().cpu().numpy()
    opt_X = torch.where(obs_mask.expand_as(X) == 0, mean_face, X)

    fig, axes = plt.subplots(7, T, figsize=(4 * T, 28))
    if T == 1:
        axes = np.expand_dims(axes, axis=1)

    row_titles = [
        "Detections (Red=SCL, Grn=Cloud, Ylw=Shadow)",
        "Artificial Holes (Black)",
        "Input (True Masked)",
        "Input (Mean Filled)",
        "Ground Truth",
        "Raw Reconstruction",
        "Final Replaced Output",
    ]
    row_labels = ["Detect", "Holes", "Input", "Filled", "GT", "Raw", "Final"]

    for t in range(T):
        is_target = (t == target_t)
        img_det = to_detection_vis(scl_mask, custom_cloud, custom_shadow, b_idx, t)
        img_art = to_art_vis(art_mask, b_idx, t)
        img_in_raw = to_rgb_img(X, b_idx, t)
        img_in_masked = img_in_raw * obs_mask_np[t][..., None]
        img_opt = to_rgb_img(opt_X, b_idx, t)
        img_gt = to_rgb_img(gt, b_idx, t)
        img_raw = to_rgb_img(raw_out, b_idx, t)
        img_final = to_rgb_img(replace_out, b_idx, t)

        imgs = [img_det, img_art, img_in_masked, img_opt, img_gt, img_raw, img_final]
        for r, img in enumerate(imgs):
            ax = axes[r, t]
            if r in [0, 1]:
                ax.imshow(img)
            else:
                ax.imshow(np.clip(img * scale, 0, 1))
            ax.axis("off")
            if t == T // 2:
                ax.set_title(row_titles[r], fontsize=20, pad=12)
            if t == 0:
                ax.text(-0.10, 0.5, row_labels[r], va='center', ha='right', rotation=90,
                        transform=ax.transAxes, fontsize=16, fontweight='bold')
            if r == 0:
                col_title = f"t={t}"
                if is_target:
                    col_title += " ★target"
                ax.text(0.5, 1.08, col_title, transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=12, color=('red' if is_target else 'black'))
            add_target_border(ax, is_target)

    suptitle = f"{tag}"
    if preview_note:
        suptitle += f"\n{preview_note}"
    fig.suptitle(suptitle, fontsize=18, y=0.995)

    plt.subplots_adjust(wspace=0.05, hspace=0.10, top=0.94)
    out_path = os.path.join(path, f"{tag}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    plt.close('all')
    gc.collect()
    return out_path


@torch.no_grad()
def evaluate(args):
    cfg = Config()
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers
    if args.uncertain_valid_weight is not None:
        cfg.UNCERTAIN_VALID_WEIGHT = float(args.uncertain_valid_weight)

    use_cuda = torch.cuda.is_available() and (args.device is None or args.device.startswith("cuda"))
    if args.device is not None and args.device.lower() == "cpu":
        use_cuda = False
    device = torch.device(args.device if args.device is not None else ("cuda" if use_cuda else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)
    preview_dir = os.path.join(args.output_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)

    TimeSeriesDataset, dataset_path = load_softmask_dataset_class()

    # 固定随机性，尽量对齐训练验证口径
    eval_seed = args.eval_seed
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)
    random.seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)

    val_dataset = TimeSeriesDataset(cfg, mode="val")
    val_dataset.set_epoch(100)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    model = MS2TAN(**cfg.MODEL_CONFIG)
    missing, unexpected = load_checkpoint_weights(model, args.checkpoint)
    if len(missing) > 0:
        print(f"[Warn] Missing keys: {missing}")
    if len(unexpected) > 0:
        print(f"[Warn] Unexpected keys: {unexpected}")

    gpu_ids = []
    if use_cuda:
        if args.gpu_ids is not None and str(args.gpu_ids).strip() != "":
            gpu_ids = [int(x) for x in str(args.gpu_ids).split(",") if str(x).strip() != ""]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))

    multi_gpu = use_cuda and len(gpu_ids) > 1
    if multi_gpu:
        primary_device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(primary_device)
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        model = model.to(primary_device)
        device = primary_device
    else:
        model = model.to(device)
    model.eval()

    print(f"[Info] Device           : {device}")
    print(f"[Info] GPU ids          : {gpu_ids if use_cuda else 'CPU only'}")
    print(f"[Info] Multi-GPU        : {multi_gpu}")
    print(f"[Info] Checkpoint       : {args.checkpoint}")
    print(f"[Info] Dataset impl     : {dataset_path}")
    print(f"[Info] Val samples      : {len(val_dataset)}")
    print(f"[Info] Val locations    : {len(getattr(val_dataset, 'target_locations', []))}")
    print(f"[Info] Preview interval : {args.preview_interval}")
    print(f"[Info] Output dir       : {args.output_dir}")

    all_records = []
    num_batches = len(val_loader) if args.max_steps is None else min(len(val_loader), args.max_steps)

    # Location_ID -> Region
    loc_to_region = {}
    if hasattr(val_dataset, "grouped_records"):
        for loc_id, recs in val_dataset.grouped_records.items():
            if len(recs) > 0:
                loc_to_region[loc_id] = str(recs[0].get("Region", "Unknown"))

    for step, batch in enumerate(tqdm(val_loader, total=num_batches, desc="[Val Replace Eval]")):
        if args.max_steps is not None and step >= args.max_steps:
            break

        X = batch["X"].to(device)
        y = batch["y"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        art_mask = batch["art_mask"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        scl_mask = batch["scl_mask"].to(device)
        custom_cloud_mask = batch["custom_cloud_mask"].to(device)
        custom_shadow_mask = batch["custom_shadow_mask"].to(device)
        target_idx = batch["target_idx"].to(device)
        loc_ids = batch["loc_id"]

        out = model(X=X, extend_layers=(obs_mask, art_mask), y=y, mode="val")
        raw_out = out["raw_out"].clamp(0, 1)
        replace_out = out["replace_out"].clamp(0, 1)
        mean_face = out.get("mean_face", calc_mean_face(X, obs_mask))

        if args.preview_interval > 0 and ((step + 1) % args.preview_interval == 0):
            preview_b = 0
            t0 = int(target_idx[preview_b].item())
            pred0 = replace_out[preview_b, t0].unsqueeze(0)
            gt0 = y[preview_b, t0].unsqueeze(0)
            hole_mask0 = (art_mask[preview_b, t0, 0:1].unsqueeze(0) * valid_mask[preview_b, t0, 0:1].unsqueeze(0))
            hole_metrics0 = metric_or_nan(pred0, gt0, hole_mask0)
            note = (
                f"sample_loc={loc_ids[preview_b]} | target_t={t0} | "
                f"Hole PSNR={hole_metrics0['psnr']:.3f} | Hole SSIM={hole_metrics0['ssim']:.4f} | "
                f"Hole SAM={hole_metrics0['sam']:.3f}"
            )
            save_preview(
                X=X,
                raw_out=raw_out,
                replace_out=replace_out,
                gt=y,
                obs_mask=obs_mask,
                art_mask=art_mask,
                scl_mask=scl_mask,
                custom_cloud=custom_cloud_mask,
                custom_shadow=custom_shadow_mask,
                mean_face=mean_face,
                target_idx=target_idx,
                tag=f"val_step_{step + 1:04d}",
                path=preview_dir,
                preview_note=note,
            )

        B = X.shape[0]
        for b in range(B):
            t = int(target_idx[b].item())
            loc_id = str(loc_ids[b])
            region = loc_to_region.get(loc_id, "Unknown")

            pred = replace_out[b, t].unsqueeze(0)
            gt = y[b, t].unsqueeze(0)
            soft_valid = valid_mask[b, t, 0:1].unsqueeze(0)
            art = art_mask[b, t, 0:1].unsqueeze(0)

            # 边界环带：人工洞向外膨胀 radius，再扣掉洞本体；最后乘 soft_valid 作为权重。
            art_binary = (art > 0.5).float()
            seam_binary = make_binary_ring(art_binary, radius=args.seam_radius)

            hole_mask = art * soft_valid
            seam_mask = seam_binary * soft_valid
            final_mask = soft_valid

            hole_metrics = metric_or_nan(pred, gt, hole_mask)
            seam_metrics = metric_or_nan(pred, gt, seam_mask)
            final_metrics = metric_or_nan(pred, gt, final_mask)

            record = {
                "sample_index": len(all_records),
                "batch_step": step + 1,
                "loc_id": loc_id,
                "region": region,
                "target_idx": t,
                "hole_weight_sum": hole_metrics["weight_sum"],
                "seam_weight_sum": seam_metrics["weight_sum"],
                "final_weight_sum": final_metrics["weight_sum"],
                "hole_psnr": hole_metrics["psnr"],
                "hole_ssim": hole_metrics["ssim"],
                "hole_sam": hole_metrics["sam"],
                "hole_rmse": hole_metrics["rmse"],
                "seam_psnr": seam_metrics["psnr"],
                "seam_ssim": seam_metrics["ssim"],
                "seam_sam": seam_metrics["sam"],
                "seam_rmse": seam_metrics["rmse"],
                "final_psnr": final_metrics["psnr"],
                "final_ssim": final_metrics["ssim"],
                "final_sam": final_metrics["sam"],
                "final_rmse": final_metrics["rmse"],
            }
            record["quality_index"] = composite_index(
                record["hole_psnr"], record["hole_ssim"], record["hole_sam"],
                record["seam_psnr"], record["seam_ssim"], record["seam_sam"],
            )
            all_records.append(record)

        del X, y, obs_mask, art_mask, valid_mask, scl_mask, custom_cloud_mask, custom_shadow_mask, target_idx
        del out, raw_out, replace_out, mean_face
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_records) == 0:
        raise RuntimeError("没有生成任何验证记录，请检查验证集是否为空。")

    df = pd.DataFrame(all_records)
    per_sample_csv = os.path.join(args.output_dir, "per_sample_metrics.csv")
    df.to_csv(per_sample_csv, index=False, encoding="utf-8-sig")

    # 1) 全部样本混合（sample-level macro）
    overall_sample = {
        "num_samples": int(len(df)),
        "num_locations": int(df["loc_id"].nunique()),
        "num_regions": int(df["region"].nunique()),
    }
    overall_sample.update(summarize_metric_dicts(all_records, "hole"))
    overall_sample.update(summarize_metric_dicts(all_records, "seam"))
    overall_sample.update(summarize_metric_dicts(all_records, "final"))
    overall_sample["quality_index"] = float(np.nanmean(df["quality_index"].values))

    # 2) 每个小地点先平均，再做 location-macro（更公平，避免高频地点主导）
    metric_cols = [
        "hole_psnr", "hole_ssim", "hole_sam", "hole_rmse",
        "seam_psnr", "seam_ssim", "seam_sam", "seam_rmse",
        "final_psnr", "final_ssim", "final_sam", "final_rmse",
        "quality_index",
        "hole_weight_sum", "seam_weight_sum", "final_weight_sum",
    ]
    per_location = df.groupby(["region", "loc_id"], dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    per_location_csv = os.path.join(args.output_dir, "per_location_metrics.csv")
    per_location.to_csv(per_location_csv, index=False, encoding="utf-8-sig")

    location_macro = {"num_locations": int(len(per_location))}
    for col in [
        "hole_psnr", "hole_ssim", "hole_sam", "hole_rmse",
        "seam_psnr", "seam_ssim", "seam_sam", "seam_rmse",
        "final_psnr", "final_ssim", "final_sam", "final_rmse",
        "quality_index",
    ]:
        location_macro[col] = float(np.nanmean(per_location[col].values))

    # 3) 按 Region 汇总，便于看 5 个大类
    per_region = per_location.groupby("region", dropna=False)[[
        "hole_psnr", "hole_ssim", "hole_sam", "hole_rmse",
        "seam_psnr", "seam_ssim", "seam_sam", "seam_rmse",
        "final_psnr", "final_ssim", "final_sam", "final_rmse",
        "quality_index",
    ]].mean(numeric_only=True).reset_index()
    per_region["num_locations"] = per_location.groupby("region", dropna=False)["loc_id"].nunique().values
    per_region_csv = os.path.join(args.output_dir, "per_region_metrics.csv")
    per_region.to_csv(per_region_csv, index=False, encoding="utf-8-sig")

    summary = {
        "checkpoint": args.checkpoint,
        "dataset_impl": dataset_path,
        "eval_seed": eval_seed,
        "batch_size": cfg.BATCH_SIZE,
        "num_workers": cfg.NUM_WORKERS,
        "preview_interval": args.preview_interval,
        "seam_radius": args.seam_radius,
        "uncertain_valid_weight": float(getattr(cfg, "UNCERTAIN_VALID_WEIGHT", 0.20)),
        "overall_sample_avg": overall_sample,
        "overall_location_macro": location_macro,
        "files": {
            "per_sample_csv": per_sample_csv,
            "per_location_csv": per_location_csv,
            "per_region_csv": per_region_csv,
            "preview_dir": preview_dir,
        },
    }

    summary_json = os.path.join(args.output_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ Final Replace Evaluation ================")
    print(f"Samples           : {overall_sample['num_samples']}")
    print(f"Locations         : {overall_sample['num_locations']}")
    print(f"Regions           : {overall_sample['num_regions']}")
    print("----------------------------------------------------------")
    print(
        "[Overall Sample Avg] "
        f"Hole PSNR={overall_sample['hole_psnr']:.4f} | Hole SSIM={overall_sample['hole_ssim']:.4f} | "
        f"Hole SAM={overall_sample['hole_sam']:.4f} | Hole RMSE={overall_sample['hole_rmse']:.4f}"
    )
    print(
        "[Overall Sample Avg] "
        f"Seam PSNR={overall_sample['seam_psnr']:.4f} | Seam SSIM={overall_sample['seam_ssim']:.4f} | "
        f"Seam SAM={overall_sample['seam_sam']:.4f} | Seam RMSE={overall_sample['seam_rmse']:.4f}"
    )
    print(
        "[Overall Sample Avg] "
        f"Final PSNR={overall_sample['final_psnr']:.4f} | Final SSIM={overall_sample['final_ssim']:.4f} | "
        f"Final SAM={overall_sample['final_sam']:.4f} | Final RMSE={overall_sample['final_rmse']:.4f}"
    )
    print(f"[Overall Sample Avg] QualityIndex={overall_sample['quality_index']:.4f}")
    print("----------------------------------------------------------")
    print(
        "[Location Macro]   "
        f"Hole PSNR={location_macro['hole_psnr']:.4f} | Hole SSIM={location_macro['hole_ssim']:.4f} | "
        f"Hole SAM={location_macro['hole_sam']:.4f} | Hole RMSE={location_macro['hole_rmse']:.4f}"
    )
    print(
        "[Location Macro]   "
        f"Seam PSNR={location_macro['seam_psnr']:.4f} | Seam SSIM={location_macro['seam_ssim']:.4f} | "
        f"Seam SAM={location_macro['seam_sam']:.4f} | Seam RMSE={location_macro['seam_rmse']:.4f}"
    )
    print(
        "[Location Macro]   "
        f"Final PSNR={location_macro['final_psnr']:.4f} | Final SSIM={location_macro['final_ssim']:.4f} | "
        f"Final SAM={location_macro['final_sam']:.4f} | Final RMSE={location_macro['final_rmse']:.4f}"
    )
    print(f"[Location Macro]   QualityIndex={location_macro['quality_index']:.4f}")
    print("==========================================================")
    print(f"[Saved] {summary_json}")
    print(f"[Saved] {per_sample_csv}")
    print(f"[Saved] {per_location_csv}")
    print(f"[Saved] {per_region_csv}")
    print(f"[Saved] {preview_dir}")

    return summary


def build_default_output_dir(checkpoint_path: str) -> str:
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    return os.path.join(ckpt_dir, "val_replace_softmask_eval")


def main():
    parser = argparse.ArgumentParser(
        description="使用 best_model 在验证集上评估最终替换结果（replace_out），并按 softmask 新逻辑打分。"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/chai/Data/wuyiru/Project_Root/checkpoints/experiment_full_15frames_refined/best_model.pth",
        help="best_model.pth 路径",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="结果输出目录")
    parser.add_argument("--batch-size", type=int, default=6, help="覆盖 Config.BATCH_SIZE")
    parser.add_argument("--num-workers", type=int, default=9, help="覆盖 Config.NUM_WORKERS")
    parser.add_argument("--max-steps", type=int, default=None, help="最多评估多少个验证 batch")
    parser.add_argument("--preview-interval", type=int, default=200, help="每隔多少个验证 step 存一张 preview")
    parser.add_argument("--seam-radius", type=int, default=3, help="边界环带半径，默认 3 像素")
    parser.add_argument("--eval-seed", type=int, default=12345, help="验证随机种子")
    parser.add_argument("--uncertain-valid-weight", type=float, default=None, help="覆盖 softmask 中可疑区域权重，例如 0.2")
    parser.add_argument("--device", type=str, default="cuda", help="例如 cuda / cuda:0 / cpu")
    parser.add_argument("--gpu-ids", type=str, default="1,2,3", help="多卡推理使用的 GPU 编号，如 '0,1,2,3'；默认自动使用全部可见 GPU")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = build_default_output_dir(args.checkpoint)

    evaluate(args)


if __name__ == "__main__":
    main()
