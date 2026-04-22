import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from math import exp
from sklearn import metrics

def masked_mae_cal(inputs, target, mask):
    """
    计算平均绝对误差 (MAE)
    修复：分母乘以通道数 inputs.shape[-3]，适配多波段数据
    """
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) * inputs.shape[-3] + 1e-6)

def masked_mse_cal(inputs, target, mask):
    """
    计算均方误差 (MSE)
    修复：分母乘以通道数 inputs.shape[-3]，防止 PSNR 计算出错
    """
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) * inputs.shape[-3] + 1e-6)

def masked_rmse_cal(inputs, target, mask):
    """计算均方根误差 (RMSE)"""
    return torch.sqrt(masked_mse_cal(inputs, target, mask))

def masked_mre_cal(inputs, target, mask):
    """计算平均相对误差 (MRE)"""
    return torch.sum(torch.abs(inputs - target) * mask) / (
        torch.sum(torch.abs(target * mask)) + 1e-6
    )

def masked_psnr_cal(inputs, target, mask, data_range=1.0):
    """
    计算峰值信噪比 (PSNR)
    修正了数值稳定性，确保 data_range 与输入设备一致
    """
    mse = masked_mse_cal(inputs, target, mask)
    # 防止 log(0) 导致负无穷或异常
    if mse < 1e-10:
        return torch.tensor(100.0).to(inputs.device)
    
    dr = torch.tensor(data_range).to(inputs.device)
    return 20 * torch.log10(dr) - 10 * torch.log10(mse)

def masked_sam_cal(inputs, target, mask):
    """
    计算光谱角制图 (SAM)
    单位：角度 (°)
    """
    # 取第一个通道的mask（假设所有通道的mask相同）
    if mask.ndim == 5:  # [B, T, C, H, W]
        mask_spatial = mask[:, :, 0:1, :, :]  # [B, T, 1, H, W]
    elif mask.ndim == 4:  # [B, C, H, W]
        mask_spatial = mask[:, 0:1, :, :]  # [B, 1, H, W]
    else:
        mask_spatial = mask
    
    # 计算内积和模长
    # dim=-3 对应通道维度 C (无论是 4D 还是 5D 张量)
    dot_product = torch.sum(inputs * target, dim=-3, keepdim=True)  
    norm_pred = torch.sqrt(torch.sum(inputs ** 2, dim=-3, keepdim=True) + 1e-6)
    norm_target = torch.sqrt(torch.sum(target ** 2, dim=-3, keepdim=True) + 1e-6)
    
    # 计算余弦值，限制在[-1, 1]范围内
    cos_angle = dot_product / (norm_pred * norm_target + 1e-6)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    # 计算角度（弧度转度）
    angle_rad = torch.acos(cos_angle)
    angle_deg = angle_rad * 180.0 / np.pi
    
    # 只在有效区域计算平均值
    sam = torch.sum(angle_deg * mask_spatial) / (torch.sum(mask_spatial) + 1e-6)
    
    return sam

# ============================================================
# 新增：局部掩膜 SSIM 计算 (Masked SSIM)
# ============================================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def masked_ssim_cal(img1, img2, mask, window_size=11):
    """
    计算带有不规则掩膜的 SSIM。
    img1, img2: [B, C, H, W], 值域为 [0, 1]
    mask: [B, 1, H, W], 1 为有效区域，0 为无效区域
    """
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    # 计算局部均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算局部方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # L=1.0 (因为数据已经归一化到 0~1)
    C1 = 0.0001 # (0.01 * 1.0)^2
    C2 = 0.0009 # (0.03 * 1.0)^2

    # 计算 SSIM 响应图
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 使用 Mask 截取有效区域的 SSIM 进行平均
    mask_sum = mask.sum()
    if mask_sum > 1e-5:
        # 确保通道数对齐后相乘
        return (ssim_map * mask).sum() / (mask_sum * channel)
    else:
        return torch.tensor(1.0, device=img1.device)

def calc_ergas(pred, target, scale_ratio=1):
    """
    计算相对整体维数合成误差 (ERGAS)
    """
    # 兼容 5D 输入 [B, T, C, H, W]，将其展平为 [N, C, H, W] 进行计算
    if pred.dim() == 5:
        B, T, C, H, W = pred.shape
        pred = pred.reshape(-1, C, H, W)
        target = target.reshape(-1, C, H, W)
    
    B, C, H, W = pred.shape
    rmse_sum = 0.0
    for i in range(C):
        diff = pred[:, i, :, :] - target[:, i, :, :]
        rmse = torch.sqrt(torch.mean(diff ** 2))
        mean = torch.mean(target[:, i, :, :])
        rmse_sum += (rmse / (mean + 1e-6)) ** 2
    
    ergas = 100 * scale_ratio * torch.sqrt(rmse_sum / C)
    return ergas

def precision_recall(y_pred, y_test):
    precisions, recalls, thresholds = metrics.precision_recall_curve(
        y_true=y_test, probas_pred=y_pred
    )
    area = metrics.auc(recalls, precisions)
    return area, precisions, recalls, thresholds

def auc_roc(y_pred, y_test):
    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise TypeError("Boolean value expected.")

# 保留旧的 calc_sam 以兼容旧代码，但逻辑其实不如 masked_sam_cal 严谨
def calc_sam(pred, target, mask=None, eps=1e-6):
    """
    计算光谱角制图 (SAM) - 旧版兼容
    """
    dot_product = torch.sum(pred * target, dim=1) 
    pred_norm = torch.norm(pred, dim=1)           
    target_norm = torch.norm(target, dim=1)       
    
    cos_theta = dot_product / (pred_norm * target_norm + eps)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    sam_map = torch.acos(cos_theta) * (180.0 / np.pi) 
    
    if mask is not None:
        mask = mask.squeeze(1) 
        return torch.sum(sam_map * mask) / (torch.sum(mask) + 1e-6)
    else:
        return torch.mean(sam_map)