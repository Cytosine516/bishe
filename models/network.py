import torch
import torch.nn as nn
from models.attention import MFE  

# ============================================================
# Utility: Mean-Face Warm Start (✅ Fixed for FP16 Stability)
# ============================================================
def calc_mean_face(X, mask):
    sum_x = torch.sum(X * mask, dim=1, keepdim=True)
    sum_m = torch.sum(mask, dim=1, keepdim=True)
    
    # 狠狠修改：提升 epsilon 到 1e-4，彻底避开 FP16 的次正规数下溢区
    eps = 1e-4 
    mean_face = sum_x / (sum_m + eps)
    
    # 确保在 15 帧全遮挡区域返回纯 0 而非脏数据
    valid_pixels = (sum_m > 0).to(X.dtype)
    mean_face = mean_face * valid_pixels
    
    return mean_face.expand_as(X)


# ============================================================
# Weight Initialization
# ============================================================
def init_weights(net, init_type="kaiming", gain=0.02):
    from torch.nn import init

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0.2, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    f"Initialization [{init_type}] not implemented"
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm2d") != -1 or classname.find("GroupNorm") != -1:
            # 兼容 GroupNorm 的初始化
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


# ============================================================
# MS2TAN Network
# ============================================================
class MS2TAN(nn.Module):
    def __init__(
        self,
        dim_list=[256, 192, 128], 
        num_frame=15,
        image_size=256,
        patch_list=[16, 32, 8],
        in_chans=14,
        out_chans=14,
        depth_list=[2, 2, 2],
        heads_list=[8, 6, 4],
        dim_head_list=[32, 32, 32],
        attn_dropout=0.0,
        ff_dropout=0.0,
        optim_input=True,
        missing_mask=True,
        enable_model=True,
        enable_conv=True,  
        enable_mse=True,
        enable_struct=True,
        enable_percept=True,
    ):
        super().__init__()

        self.num_block = len(dim_list)
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.optim_input = optim_input
        self.enable_model = enable_model
        self.enable_conv = enable_conv
        
        self.blocks = nn.ModuleList()
        if enable_model:
            for i in range(self.num_block):
                self.blocks.append(
                    MFE(
                        dim=dim_list[i],
                        num_frames=num_frame,
                        image_size=image_size,
                        patch_size=patch_list[i],
                        in_channels=in_chans,
                        out_channels=out_chans,
                        depth=depth_list[i],
                        heads=heads_list[i],
                        dim_head=dim_head_list[i],
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        missing_mask=(i == 0 and missing_mask), 
                        diag_mask=False,
                    )
                )

        self.first_conv = None
        self.after_conv = nn.ModuleList()

        if self.enable_conv:
            cnum = 32
            
            # 狠狠修改：用 GroupNorm(eps=1e-4) 替换所有 BatchNorm2d！
            # num_groups=4 意味着每 8 个通道一组算统计量，完美避开 batch 维度的崩塌
            self.first_conv = nn.Sequential(
                nn.Flatten(0, 1), 
                nn.Conv2d(out_chans, cnum, kernel_size=5, stride=1, padding=2),
                nn.GroupNorm(num_groups=4, num_channels=cnum, eps=1e-4),
                nn.PReLU(cnum),
                nn.Conv2d(cnum, 2 * cnum, kernel_size=3, stride=1, padding=1),
                nn.PReLU(2 * cnum),
                nn.Conv2d(2 * cnum, 2 * cnum, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(num_groups=8, num_channels=2 * cnum, eps=1e-4),
                nn.PReLU(2 * cnum),
                nn.Conv2d(2 * cnum, 2 * cnum, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=2 * cnum, eps=1e-4),
                nn.PReLU(2 * cnum),
                nn.Conv2d(2 * cnum, out_chans, kernel_size=5, stride=1, padding=2),
                nn.PReLU(out_chans),
                nn.Unflatten(0, (-1, num_frame)), 
            )

            for _ in range(self.num_block):
                self.after_conv.append(
                    nn.Sequential(
                        nn.Flatten(0, 1),
                        nn.Conv2d(out_chans, cnum, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=4, num_channels=cnum, eps=1e-4),
                        nn.PReLU(cnum),
                        nn.Conv2d(cnum, cnum, kernel_size=1, stride=1, padding=0),
                        nn.GroupNorm(num_groups=4, num_channels=cnum, eps=1e-4),
                        nn.PReLU(cnum),
                        nn.Conv2d(cnum, out_chans, kernel_size=3, stride=1, padding=1),
                        nn.Unflatten(0, (-1, num_frame)),
                    )
                )

    def forward(self, X, extend_layers, y=None, mode="train", return_attn=False):
        b, t, c, h, w = X.shape
        obs_mask, art_mask = extend_layers

        mean_face = calc_mean_face(X, obs_mask)
        obs_mask_exp = obs_mask.expand(-1, -1, c, -1, -1)
        opt_X = torch.where(obs_mask_exp == 0, mean_face, X)

        out = opt_X if self.optim_input else X

        if self.enable_conv and self.first_conv is not None:
            out = out + self.first_conv(out)

        block_out = []
        block_attn = []

        for idx, block in enumerate(self.blocks):
            if self.enable_model:
                merge = torch.where(obs_mask_exp != 0, opt_X, out)

                if return_attn:
                    res, attn = block(merge, obs_mask, return_attn=True)
                    out = out + res
                    block_attn.append(attn)
                else:
                    out = out + block(merge, obs_mask)

            if self.enable_conv and idx < len(self.after_conv):
                out = out + self.after_conv[idx](out)

            block_out.append(out)

        raw_out = torch.sigmoid(block_out[-1])
        replace_out = torch.where(obs_mask_exp != 0, opt_X, raw_out)

        return {
            "raw_out": raw_out,         
            "replace_out": replace_out, 
            "mean_face": mean_face,     
            "block_attn": block_attn,   
        }