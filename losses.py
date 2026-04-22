import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ===============================
# 1. L1 Masked Reconstruction Loss
# ===============================
class MaskedL1Loss(nn.Module):
    def __init__(self, eps=1e-4, target_weight=2.0): # 狠狠修改：1e-6 -> 1e-4
        super().__init__()
        self.eps = eps
        self.target_weight = target_weight

    def forward(self, pred, target, art_mask, is_clean_flag, valid_mask=None, target_idx=None):
        pred = pred.float()
        target = target.float()
        
        diff = pred - target
        loss_map = torch.sqrt(diff * diff + self.eps)

        valid_frames = is_clean_flag.view(pred.size(0), pred.size(1), 1, 1, 1).float()
        final_mask = art_mask * valid_frames

        if valid_mask is not None:
            final_mask = final_mask * valid_mask

        B, T = pred.shape[:2]
        temporal_weights = torch.ones((B, T, 1, 1, 1), device=pred.device, dtype=pred.dtype)
        if target_idx is not None:
            temporal_weights[torch.arange(B), target_idx, :, :, :] = self.target_weight
        
        weighted_mask = final_mask * temporal_weights
        masked_loss = torch.where(weighted_mask > 0, loss_map * weighted_mask, torch.zeros_like(loss_map))

        denom = weighted_mask.sum()
        if denom > 1e-5:
            return masked_loss.sum() / denom
        else:
            return (pred * 0.0).sum()


# ===============================
# 2. SAM Loss (Spectral Consistency)
# ===============================
class SAMLoss(nn.Module):
    def __init__(self, target_weight=2.0):
        super().__init__()
        self.target_weight = target_weight

    def forward(self, pred, target, mask=None, target_idx=None):
        pred = pred.float()
        target = target.float()
        
        temporal_weights = None
        
        if pred.dim() == 5:
            B, T = pred.shape[:2]
            temporal_weights = torch.ones((B, T, 1, 1, 1), device=pred.device, dtype=pred.dtype)
            if target_idx is not None:
                temporal_weights[torch.arange(B), target_idx, :, :, :] = self.target_weight
            temporal_weights = temporal_weights.reshape(-1, 1, 1, 1)
            
            pred = pred.reshape(-1, *pred.shape[2:])
            target = target.reshape(-1, *target.shape[2:])
            if mask is not None:
                mask = mask.reshape(-1, *mask.shape[2:])

        if mask is not None:
            mask_bool = mask > 0.5
            pred_clean = torch.where(mask_bool, pred, torch.zeros_like(pred))
            target_clean = torch.where(mask_bool, target, torch.zeros_like(target))
        else:
            pred_clean = pred
            target_clean = target

        dot = (pred_clean * target_clean).sum(dim=1)
        norm_p = torch.norm(pred_clean, dim=1)
        norm_t = torch.norm(target_clean, dim=1)

        denominator = torch.clamp(norm_p * norm_t, min=1e-8)
        # 狠狠修改：clamp 边界提升到 1e-4，避开 acos 的奇点
        cos = torch.clamp(dot / denominator, -1.0 + 1e-4, 1.0 - 1e-4)
        sam = torch.acos(cos)
        
        if temporal_weights is not None:
            sam = sam * temporal_weights.view(-1, 1, 1)  
            if mask is not None:
                mask = mask * temporal_weights

        if mask is not None:
            mask_sum = mask.sum()
            if mask_sum > 1e-5:
                return (sam * mask.squeeze(1)).sum() / mask_sum
            else:
                return (pred * 0.0).sum()
        else:
            return sam.mean()


# ===============================
# 3. Total Variation Loss (去噪)
# ===============================
class TVLoss(nn.Module):
    def __init__(self, weight=1.0, target_weight=2.0):
        super().__init__()
        self.weight = weight
        self.target_weight = target_weight

    def forward(self, x, target_idx=None):
        x = x.float()
        
        temporal_weights = None
        if x.dim() == 5:
            B, T = x.shape[:2]
            temporal_weights = torch.ones((B, T, 1, 1, 1), device=x.device, dtype=x.dtype)
            if target_idx is not None:
                temporal_weights[torch.arange(B), target_idx, :, :, :] = self.target_weight
            temporal_weights = temporal_weights.reshape(-1, 1, 1, 1)
            
            x = x.reshape(-1, *x.shape[2:])

        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)
        
        if temporal_weights is not None:
            h_tv = h_tv * temporal_weights
            w_tv = w_tv * temporal_weights
            
        return self.weight * 2 * (h_tv.sum() / count_h + w_tv.sum() / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size(1) * t.size(2) * t.size(3)


# ===============================
# 4. Perceptual & Style Loss
# ===============================
class PerceptualStyleLoss(nn.Module):
    def __init__(self, rgb_indices=None, target_weight=2.0):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.slice1 = nn.Sequential(*list(vgg.features)[:4]).eval()
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9]).eval()
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16]).eval()
        self.slice4 = nn.Sequential(*list(vgg.features)[16:23]).eval()

        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.rgb_indices = rgb_indices
        self.target_weight = target_weight

    def gram_matrix(self, input):
        a, b, c, d = input.size() 
        features = input.view(a * b, c * d) 
        G = torch.mm(features, features.t()) 
        return G.div(a * b * c * d + 1e-8)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, y, with_style=False, target_idx=None): 
        x = x.float()
        y = y.float()
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)

        temporal_weights = None
        if x.dim() == 5:
            B, T = x.shape[:2]
            temporal_weights = torch.ones((B, T, 1, 1, 1), device=x.device, dtype=x.dtype)
            if target_idx is not None:
                temporal_weights[torch.arange(B), target_idx, :, :, :] = self.target_weight
            temporal_weights = temporal_weights.reshape(-1, 1, 1, 1)

            x = x.reshape(-1, *x.shape[2:])
            y = y.reshape(-1, *y.shape[2:])
        
        if self.rgb_indices is not None:
            x_in = x[:, self.rgb_indices, :, :]
            y_in = y[:, self.rgb_indices, :, :]
        else:
            if x.shape[1] > 3:
                x_in = x[:, :3, :, :]
                y_in = y[:, :3, :, :]
            else:
                x_in = x
                y_in = y

        x_in = (x_in - self.mean) / self.std
        y_in = (y_in - self.mean) / self.std

        x_relu1 = self.slice1(x_in)
        y_relu1 = self.slice1(y_in)
        x_relu2 = self.slice2(x_relu1) 
        y_relu2 = self.slice2(y_relu1)
        x_relu3 = self.slice3(x_relu2) 
        y_relu3 = self.slice3(y_relu2)
        x_relu4 = self.slice4(x_relu3) 
        y_relu4 = self.slice4(y_relu3)

        l_percept2 = F.l1_loss(x_relu2, y_relu2, reduction='none')
        l_percept3 = F.l1_loss(x_relu3, y_relu3, reduction='none')
        l_percept4 = F.l1_loss(x_relu4, y_relu4, reduction='none')
        
        if temporal_weights is not None:
            l_percept2 = l_percept2 * temporal_weights
            l_percept3 = l_percept3 * temporal_weights
            l_percept4 = l_percept4 * temporal_weights
            
        l_percept = (l_percept2.mean() + l_percept3.mean() + l_percept4.mean()) / 3.0
        
        l_style = torch.tensor(0.0, device=x.device)
        if with_style:
            style_diff = F.l1_loss(self.gram_matrix(x_relu3), self.gram_matrix(y_relu3), reduction='none')
            if temporal_weights is not None:
                style_diff = style_diff * temporal_weights.view(-1, 1)
            l_style = style_diff.mean()

        return l_percept, l_style
    
# ===============================
# 5. Gradient / Edge Loss
# ===============================
class GradientLoss(nn.Module):
    def __init__(self, target_weight=2.0):
        super().__init__()

        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = sobel_x.transpose(2, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.target_weight = target_weight

    def forward(self, pred, target, mask=None, target_idx=None):
        pred = pred.float()
        target = target.float()
        
        temporal_weights = None
        if pred.dim() == 5:
            B, T = pred.shape[:2]
            temporal_weights = torch.ones((B, T, 1, 1, 1), device=pred.device, dtype=pred.dtype)
            if target_idx is not None:
                temporal_weights[torch.arange(B), target_idx, :, :, :] = self.target_weight
            temporal_weights = temporal_weights.reshape(-1, 1, 1, 1)

            b, t, c, h, w = pred.shape
            pred = pred.reshape(-1, c, h, w)
            target = target.reshape(-1, c, h, w)
            if mask is not None:
                mask = mask.reshape(-1, 1, h, w)

        sobel_x = self.sobel_x.repeat(pred.shape[1], 1, 1, 1)
        sobel_y = self.sobel_y.repeat(pred.shape[1], 1, 1, 1)

        grad_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
        grad_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])
        # 狠狠修改：1e-6 -> 1e-4
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-4)

        grad_x_gt = F.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
        grad_y_gt = F.conv2d(target, sobel_y, padding=1, groups=target.shape[1])
        # 狠狠修改：1e-6 -> 1e-4
        grad_gt = torch.sqrt(grad_x_gt ** 2 + grad_y_gt ** 2 + 1e-4)

        if mask is not None:
            if temporal_weights is not None:
                mask = mask * temporal_weights
            grad = grad * mask
            grad_gt = grad_gt * mask
            
            denom = mask.sum() * pred.shape[1]
            if denom > 1e-5:
                return torch.abs(grad - grad_gt).sum() / denom
            else:
                return (pred * 0.0).sum()
        else:
            loss_map = torch.abs(grad - grad_gt)
            if temporal_weights is not None:
                loss_map = loss_map * temporal_weights
            return loss_map.mean()


# ===============================
# 6. FFT Loss (频域损失)
# ===============================
class FFTLoss(nn.Module):
    def __init__(self, target_weight=2.0):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none') 
        self.target_weight = target_weight

    def forward(self, pred, target, target_idx=None):
        pred = pred.float()
        target = target.float()
        
        temporal_weights = None
        if pred.dim() == 5:
            B, T = pred.shape[:2]
            temporal_weights = torch.ones((B, T, 1, 1, 1), device=pred.device, dtype=pred.dtype)
            if target_idx is not None:
                temporal_weights[torch.arange(B), target_idx, :, :, :] = self.target_weight

        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # 狠狠修改：1e-8 -> 1e-4 (1e-8的平方根导数大得离谱！)
        pred_mag = torch.sqrt(pred_fft.real**2 + pred_fft.imag**2 + 1e-4)
        target_mag = torch.sqrt(target_fft.real**2 + target_fft.imag**2 + 1e-4)

        loss_map = self.l1(pred_mag, target_mag)
        
        if temporal_weights is not None:
            loss_map = loss_map * temporal_weights
            
        return loss_map.mean()