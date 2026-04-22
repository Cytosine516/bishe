import os
import sys
import logging
import argparse
import gc  
import random 
from datetime import datetime

import matplotlib
matplotlib.use('Agg') 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from config import Config
from dataset import TimeSeriesDataset

from losses import (
    PerceptualStyleLoss, 
    MaskedL1Loss, 
    SAMLoss, 
    TVLoss, 
    GradientLoss, 
    FFTLoss
)
from utils.pytorch_ssim import SSIM
from utils.util_metrics import (
    masked_psnr_cal,
    masked_sam_cal,
    masked_rmse_cal,
    masked_ssim_cal  # ✅ 新增：导入带有掩码的 SSIM 计算函数
)

sys.path.append("MS2TAN_Project")
from models.network import MS2TAN, init_weights


def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(
        save_dir,
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================
# Preview Utility
# ============================================================
def save_preview(X, raw, gt, art_mask, valid_mask, scl_mask, custom_cloud, custom_shadow, mean_face, tag, path):
    import gc
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    valid_indices = torch.nonzero(art_mask.sum(dim=(2, 3, 4)) > 0)
    b_idx = valid_indices[0][0].item() if len(valid_indices) > 0 else 0

    T = X.shape[1]
    scale = 1.0 

    def to_img(tensor, t_idx):
        return tensor[b_idx, t_idx, [3, 2, 1]].detach().cpu().numpy().transpose(1, 2, 0)
        
    def to_detection_vis(scl_m, c_cloud, c_shadow, t_idx):
        scl_np = scl_m[b_idx, t_idx, 0].detach().cpu().numpy()
        cloud_np = c_cloud[b_idx, t_idx, 0].detach().cpu().numpy()
        shadow_np = c_shadow[b_idx, t_idx, 0].detach().cpu().numpy()
        
        vis = np.ones((scl_np.shape[0], scl_np.shape[1], 3), dtype=np.float32)
        
        # 顺序叠加：底色白 -> 阴影黄 -> 自定义云绿 -> SCL红(最高优)
        vis[shadow_np == 1] = [1.0, 1.0, 0.0] 
        vis[cloud_np == 1] = [0.0, 1.0, 0.0] 
        vis[scl_np == 1] = [1.0, 0.0, 0.0] 
        return vis

    def to_art_vis(art, t_idx):
        art_np = art[b_idx, t_idx, 0].detach().cpu().numpy()
        vis = np.ones((art_np.shape[0], art_np.shape[1], 3), dtype=np.float32)
        vis[art_np == 1] = [0.0, 0.0, 0.0] # 人工黑洞用黑色
        return vis

    # 1. 计算真正被挖空的 Input (人工洞 + SCL + 程序判定云阴影 全涂黑)
    obs_mask = valid_mask * (1.0 - art_mask)
    obs_mask_np = obs_mask[b_idx, :, 0].detach().cpu().numpy() # [T, H, W]
    
    # 2. 均值填充后的 Input
    obs_mask_exp = obs_mask.expand_as(X)
    opt_X = torch.where(obs_mask_exp == 0, mean_face, X)

    fig, axes = plt.subplots(6, T, figsize=(4 * T, 24)) 
    
    for t in range(T):
        img_det = to_detection_vis(scl_mask, custom_cloud, custom_shadow, t)
        img_art = to_art_vis(art_mask, t)
        
        img_in_raw = to_img(X, t)
        img_in_masked = img_in_raw * obs_mask_np[t][..., None]
        
        img_opt = to_img(opt_X, t)
        img_gt = to_img(gt, t)
        img_raw = to_img(raw, t)
        
        # Row 1: Detection Masks (Red=SCL, Green=Cloud, Yellow=Shadow)
        ax = axes[0, t] if T > 1 else axes[0]
        ax.imshow(img_det)
        ax.axis("off")
        if t == T // 2: ax.set_title("Detections (Red=SCL, Grn=Cloud, Ylw=Shadow)", fontsize=22, pad=15)
        if t == 0: ax.text(-0.1, 0.5, 'Detect', va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=18, fontweight='bold')

        # Row 2: Artificial Holes Only
        ax = axes[1, t] if T > 1 else axes[1]
        ax.imshow(img_art)
        ax.axis("off")
        if t == T // 2: ax.set_title("Artificial Holes (Black)", fontsize=22, pad=15)
        if t == 0: ax.text(-0.1, 0.5, 'Holes', va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=18, fontweight='bold')

        # Row 3: Input with everything masked out
        ax = axes[2, t] if T > 1 else axes[2]
        ax.imshow(np.clip(img_in_masked * scale, 0, 1))
        ax.axis("off")
        if t == T // 2: ax.set_title("Input (True Masked)", fontsize=22, pad=15)
        if t == 0: ax.text(-0.1, 0.5, 'Input', va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=18, fontweight='bold')

        # Row 4: Mean Filled Input
        ax = axes[3, t] if T > 1 else axes[3]
        ax.imshow(np.clip(img_opt * scale, 0, 1))
        ax.axis("off")
        if t == T // 2: ax.set_title("Input (Mean Filled)", fontsize=22, pad=15)
        if t == 0: ax.text(-0.1, 0.5, 'Filled', va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=18, fontweight='bold')

        # Row 5: Ground Truth
        ax = axes[4, t] if T > 1 else axes[4]
        ax.imshow(np.clip(img_gt * scale, 0, 1))
        ax.axis("off")
        if t == T // 2: ax.set_title("Ground Truth", fontsize=22, pad=15)
        if t == 0: ax.text(-0.1, 0.5, 'GT', va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=18, fontweight='bold')

        # Row 6: Reconstruction
        ax = axes[5, t] if T > 1 else axes[5]
        ax.imshow(np.clip(img_raw * scale, 0, 1))
        ax.axis("off")
        if t == T // 2: ax.set_title("Reconstruction", fontsize=22, pad=15)
        if t == 0: ax.text(-0.1, 0.5, 'Recon', va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=18, fontweight='bold')

    os.makedirs(path, exist_ok=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(os.path.join(path, f"{tag}.png"), bbox_inches='tight', dpi=150)
    
    plt.close(fig)
    plt.close('all')
    del fig, axes
    gc.collect()


# ============================================================
# TrainWrapper
# ============================================================
class TrainWrapper(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        
        T_WEIGHT = 2.0 
        self.l1 = MaskedL1Loss(target_weight=T_WEIGHT)
        self.sam = SAMLoss(target_weight=T_WEIGHT)
        self.ssim = SSIM() 
        self.tv = TVLoss(weight=1.0, target_weight=T_WEIGHT)
        self.grad = GradientLoss(target_weight=T_WEIGHT)
        self.fft = FFTLoss(target_weight=T_WEIGHT)
        
        if config.MODEL_CONFIG.get("enable_percept", False):
            self.percept_style = PerceptualStyleLoss(rgb_indices=[3, 2, 1], target_weight=T_WEIGHT)
        else:
            self.percept_style = None

    def forward(self, X, y, obs_mask, art_mask, valid_mask, is_clean_flag, target_idx, mode="train"):
        out = self.model(
            X=X,
            extend_layers=(obs_mask, art_mask),
            y=y,
            mode=mode,
        )

        if mode == "val":
            return out

        raw = out["raw_out"] 

        loss_l1 = self.l1(raw, y, torch.ones_like(art_mask), torch.ones_like(is_clean_flag), valid_mask, target_idx)
        loss_hole = self.l1(raw, y, art_mask, is_clean_flag, valid_mask, target_idx)
        
        # ✅ 核心优化：修复 SSIM “纯黑图骗局”
        b, t, c, h, w = raw.shape
        clean_mask_5d = is_clean_flag.view(b, t, 1, 1, 1)
        raw_clean = raw * clean_mask_5d
        y_clean = y * clean_mask_5d

        clean_mask_1d = is_clean_flag.view(-1).bool() 
        raw_flat = raw.reshape(-1, c, h, w)
        y_flat = y.reshape(-1, c, h, w)
        
        if clean_mask_1d.any():
            raw_clean_filtered = raw_flat[clean_mask_1d]
            y_clean_filtered = y_flat[clean_mask_1d]
            loss_ssim = 1.0 - self.ssim(raw_clean_filtered, y_clean_filtered)
        else:
            loss_ssim = torch.tensor(0.0, device=raw.device, requires_grad=True)
        
        loss_sam = self.sam(raw, y, valid_mask, target_idx)
        loss_sam_hole = self.sam(raw, y, art_mask, target_idx)
        
        loss_tv = self.tv(raw, target_idx)
        loss_grad = self.grad(raw, y, valid_mask, target_idx)
        loss_fft = self.fft(raw_clean, y_clean, target_idx) 

        loss_percept = torch.tensor(0.0, device=raw.device)
        loss_style = torch.tensor(0.0, device=raw.device)

        if self.percept_style is not None:
            loss_percept, loss_style = self.percept_style(raw_clean, y_clean, with_style=False, target_idx=target_idx)

        w_l1       = 5.0     
        w_hole     = 10.0    
        w_ssim     = 2.0     
        w_sam      = 5.0     
        w_sam_hole = 10.0    
        w_grad     = 8.0    
        w_fft      = 12.0     
        w_percept  = 5.0     
        w_style    = 0.0     
        w_tv       = 0.1     

        total_loss = (
            w_l1 * loss_l1 +
            w_hole * loss_hole +
            w_ssim * loss_ssim +
            w_sam * loss_sam +
            w_sam_hole * loss_sam_hole +
            w_fft * loss_fft +    
            w_percept * loss_percept +
            w_style * loss_style +
            w_tv * loss_tv +
            w_grad * loss_grad
        )

        loss_sam_display = (loss_sam + loss_sam_hole) / 2

        return torch.stack([
            total_loss, 
            loss_l1, loss_hole, loss_ssim, loss_sam_display, loss_fft, 
            loss_percept, loss_style, loss_tv, loss_grad
        ])

# ============================================================
# Train Main Loop
# ============================================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    cfg = Config()
    cfg.VAL_STEPS = 500 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(cfg.SAVE_DIR)

    logger.info(f"🚀 Start Training | GPUs: {torch.cuda.device_count()} | Val Steps: {cfg.VAL_STEPS}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_dataset = TimeSeriesDataset(cfg, mode="train")
    val_dataset = TimeSeriesDataset(cfg, mode="val")
    val_dataset.set_epoch(100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    base_model = MS2TAN(**cfg.MODEL_CONFIG)
    init_weights(base_model, "kaiming")

    try:
        model_wrapper = TrainWrapper(base_model, cfg).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("\n❌ CUDA Out Of Memory Error 发生于模型初始化阶段！")
            logger.error("这通常意味着 GPU 0 当前被其他进程占用，或者你的 VRAM 存在未清理的碎片。")
            if torch.cuda.is_available():
                logger.error(torch.cuda.memory_summary(device=device))
            raise RuntimeError("请在终端运行 `nvidia-smi` 检查是否有僵尸进程正在占用显存。") from e
        else:
            raise e

    if torch.cuda.device_count() > 1:
        model_wrapper = nn.DataParallel(model_wrapper)

    raw_model = model_wrapper.module.model if isinstance(model_wrapper, nn.DataParallel) else model_wrapper.model
    optimizer = optim.Adam(raw_model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_models = [] 
    global_step = 0
    
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  
                logger.info(f"✅ Epoch recovered. Will start from Epoch {start_epoch + 1}.")
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("✅ Optimizer state loaded.")
        else:
            state_dict = checkpoint

        new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('percept_loss.')}
        final_state_dict = {}
        for k, v in new_state_dict.items():
            if k.startswith('module.'):
                final_state_dict[k[7:]] = v
            else:
                final_state_dict[k] = v
        
        missing, unexpected = raw_model.load_state_dict(final_state_dict, strict=False)
        if len(missing) > 0: logger.warning(f"⚠️ Missing keys: {missing}")
        if len(unexpected) > 0: logger.info(f"ℹ️ Unexpected keys: {unexpected}")
            
        logger.info("✅ Model weights loaded. Starting fine-tuning.")
    else:
        logger.info("🆕 Starting from scratch.")

    eps = 1e-6

    for epoch in range(start_epoch, cfg.EPOCHS):
        train_dataset.set_epoch(epoch)
        model_wrapper.train()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]", total=cfg.STEPS_PER_EPOCH)

        for step, batch in enumerate(loop):
            if step >= cfg.STEPS_PER_EPOCH:
                break
            global_step += 1

            X = batch["X"].to(device)
            y = batch["y"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            art_mask = batch["art_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            
            scl_mask = batch["scl_mask"].to(device) 
            custom_cloud_mask = batch["custom_cloud_mask"].to(device)
            custom_shadow_mask = batch["custom_shadow_mask"].to(device)
            
            is_clean_flag = batch["is_clean_flag"].to(device)
            target_idx = batch["target_idx"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                losses_vec = model_wrapper(
                    X, y, obs_mask, art_mask, valid_mask, is_clean_flag, target_idx, mode="train"
                )
                
                if losses_vec.dim() == 1:
                    losses_vec = losses_vec.view(-1, 10)
                
                losses_mean = losses_vec.mean(dim=0)
                
                (total_loss, loss_l1, loss_hole, loss_ssim, loss_sam, 
                 loss_fft, loss_percept, loss_style, loss_tv, loss_grad) = losses_mean

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(
                l1=f"{loss_l1.item():.3f}",
                sam=f"{loss_sam.item():.3f}", 
                fft=f"{loss_fft.item():.3f}",
                total=f"{total_loss.item():.3f}",
            )
            
            if global_step % 100 == 0:
                logger.debug(
                    f"Step {global_step} | Total={total_loss:.4f} | L1={loss_l1:.4f} | "
                    f"Hole={loss_hole:.4f} | SAM(Avg)={loss_sam:.4f} | FFT={loss_fft:.4f} | "
                    f"SSIM={loss_ssim:.4f} | Grad={loss_grad:.4f} | Perc={loss_percept:.4f}"
                )

            if global_step % cfg.PREVIEW_INTERVAL == 0:
                model_wrapper.eval()
                with torch.no_grad():
                    out = model_wrapper(X, y, obs_mask, art_mask, valid_mask, is_clean_flag, target_idx, mode="val")
                    raw_out_preview = out["raw_out"]
                    mean_face = out["mean_face"]
                    
                    save_preview(
                        X, raw_out_preview.clamp(0, 1), y, art_mask, valid_mask, 
                        scl_mask, custom_cloud_mask, custom_shadow_mask, mean_face,
                        f"e{epoch+1}_s{global_step}",
                        os.path.join(cfg.SAVE_DIR, "previews"),
                    )
                    
                    # 预览画完后，手动清除占用的预测输出
                    del out, raw_out_preview, mean_face
                model_wrapper.train()

            # ==========================================
            # ✅ 核心优化 1：清空当前 Step 产生的大量张量
            # ==========================================
            del X, y, obs_mask, art_mask, valid_mask, scl_mask, custom_cloud_mask, custom_shadow_mask, is_clean_flag, target_idx
            del losses_vec, losses_mean, total_loss
            # 如果不显式删除，这些张量会一直留存到下一次循环覆盖，占用宝贵的峰值显存空间。

        scheduler.step()

        # ==========================================
        # ✅ 核心优化 2：Train 结束进入 Val 前的深度清理
        # ==========================================
        gc.collect()
        torch.cuda.empty_cache()

        # ============================
        # Validation
        # ============================
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available(): cuda_rng_state = torch.cuda.get_rng_state()
        np_rng_state = np.random.get_state()
        py_rng_state = random.getstate()

        eval_seed = 12345
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(eval_seed)
        np.random.seed(eval_seed)
        random.seed(eval_seed)

        model_wrapper.eval()
        
        # ✅ 核心优化：新增局部与全局指标容器
        masked_psnrs, masked_sams, masked_ssims, masked_mses = [], [], [], []
        global_psnrs, global_ssims, global_sams = [], [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", total=cfg.VAL_STEPS)):
                if i >= cfg.VAL_STEPS:
                    break

                X = batch["X"].to(device)
                y = batch["y"].to(device)
                obs_mask = batch["obs_mask"].to(device)
                art_mask = batch["art_mask"].to(device)
                valid_mask = batch["valid_mask"].to(device)
                is_clean_flag = batch["is_clean_flag"].to(device)
                target_idx = batch["target_idx"].to(device)

                out = model_wrapper(
                    X, y, obs_mask, art_mask, valid_mask, is_clean_flag, target_idx, mode="val"
                )
                
                B = X.shape[0]
                batch_indices = torch.arange(B, device=device)
                
                rec_target = out["raw_out"][batch_indices, target_idx].unsqueeze(1).clamp(0, 1)
                gt_target = y[batch_indices, target_idx].unsqueeze(1).clamp(0, 1)
                art_mask_target = art_mask[batch_indices, target_idx].unsqueeze(1)
                valid_mask_target = valid_mask[batch_indices, target_idx].unsqueeze(1)

                rec_f = rec_target.reshape(-1, gt_target.shape[2], gt_target.shape[3], gt_target.shape[4])
                gt_f = gt_target.reshape_as(rec_f)
                
                # ✅ 核心掩膜定义：精确区分局部与全局，彻底剔除真实云和阴影
                local_mask_f = (art_mask_target * valid_mask_target).reshape(-1, 1, gt_target.shape[3], gt_target.shape[4])
                global_mask_f = valid_mask_target.reshape(-1, 1, gt_target.shape[3], gt_target.shape[4])

                # 1. 计算 Local (局部) 指标
                if local_mask_f.sum() > 0:
                    masked_psnrs.append(masked_psnr_cal(rec_f, gt_f, local_mask_f).item())
                    masked_sams.append(masked_sam_cal(rec_f, gt_f, local_mask_f).item())
                    masked_ssims.append(masked_ssim_cal(rec_f, gt_f, local_mask_f).item())
                    masked_mses.append(
                        ((rec_f - gt_f) ** 2 * local_mask_f).sum().item()
                        / (local_mask_f.sum().item() + eps)
                    )
                
                # 2. 计算 Global (全局) 指标
                if global_mask_f.sum() > 0:
                    global_ssims.append(masked_ssim_cal(rec_f, gt_f, global_mask_f).item())
                    global_sams.append(masked_sam_cal(rec_f, gt_f, global_mask_f).item())
                    
                    global_mse_val = ((rec_f - gt_f) ** 2 * global_mask_f).sum().item() / (global_mask_f.sum().item() + eps)
                    if global_mse_val > 0:
                        global_psnrs.append(10 * np.log10(1.0 / global_mse_val))
                    else:
                        global_psnrs.append(100.0)

                # ==========================================
                # ✅ 核心优化 3：验证循环内部清理，防止验证集 OOM
                # ==========================================
                del X, y, obs_mask, art_mask, valid_mask, is_clean_flag, target_idx
                del out, rec_target, gt_target, art_mask_target, valid_mask_target, rec_f, gt_f, local_mask_f, global_mask_f

        torch.set_rng_state(rng_state)
        if torch.cuda.is_available(): torch.cuda.set_rng_state(cuda_rng_state)
        np.random.set_state(np_rng_state)
        random.setstate(py_rng_state)

        # 安全防护，防止空列表计算报错
        if len(masked_psnrs) == 0: masked_psnrs = [0]
        if len(masked_ssims) == 0: masked_ssims = [0]
        if len(masked_sams) == 0: masked_sams = [0]
        
        if len(global_psnrs) == 0: global_psnrs = [0]
        if len(global_ssims) == 0: global_ssims = [0]
        if len(global_sams) == 0: global_sams = [0]

        avg_m_psnr = np.mean(masked_psnrs)
        avg_m_ssim = np.mean(masked_ssims)
        avg_m_sam = np.mean(masked_sams)
        
        avg_g_psnr = np.mean(global_psnrs)
        avg_g_ssim = np.mean(global_ssims)
        avg_g_sam = np.mean(global_sams)

        # ✅ 核心优化：科学的双重 Score 融合打分体系
        score_local = (1.0 * avg_m_psnr) + (20.0 * avg_m_ssim) - (100.0 * avg_m_sam)
        score_global = (1.0 * avg_g_psnr) + (20.0 * avg_g_ssim) - (100.0 * avg_g_sam)
        score = (0.7 * score_local) + (0.3 * score_global)

        logger.info(
            f"[Val-DualMetrics] Epoch {epoch+1} | Score {score:.4f}\n"
            f"   ├─ [Local (Holes)]  PSNR: {avg_m_psnr:.2f} | SSIM: {avg_m_ssim:.4f} | SAM: {avg_m_sam:.4f}\n"
            f"   └─ [Global (Clean)] PSNR: {avg_g_psnr:.2f} | SSIM: {avg_g_ssim:.4f} | SAM: {avg_g_sam:.4f}"
        )

        # ==========================================
        # ✅ 核心优化 4：在保存权重前，执行终极清理
        # ==========================================
        gc.collect()
        torch.cuda.empty_cache()

        torch.save(
            {'epoch': epoch, 'state_dict': raw_model.state_dict(), 'optimizer': optimizer.state_dict()},
            os.path.join(cfg.SAVE_DIR, "last_checkpoint.pth")
        )

        # 保存文件名保留 avg_m_psnr 供直观参考
        save_name = os.path.join(
            cfg.SAVE_DIR, 
            f"model_e{epoch+1}_score{score:.2f}_psnr{avg_m_psnr:.2f}.pth"
        )
        
        best_models.append((score, epoch, save_name))
        best_models.sort(key=lambda x: x[0], reverse=True)
        
        if (score, epoch, save_name) in best_models[:3]:
            torch.save(raw_model.state_dict(), save_name)
            logger.info(f"💾 Saved Top-3 Model (Best Score): {save_name}")
            
            if (score, epoch, save_name) == best_models[0]:
                torch.save(
                    raw_model.state_dict(),
                    os.path.join(cfg.SAVE_DIR, "best_model.pth"),
                )
                logger.info("🏆 Best Model Updated (Based on Composite Score)")
            
        if len(best_models) > 3:
            worst_model = best_models.pop() 
            if os.path.exists(worst_model[2]):
                os.remove(worst_model[2]) 
                logger.info(f"🗑️ Removed old model: {worst_model[2]}")
        
        # 确保收尾显存干干净净
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()