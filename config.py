import os

class Config:
    # ==================================================
    # 路径配置  
    # ==================================================
    DATA_ROOT = r"/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_Patches_FullBand"
    CSV_PATH = os.path.join(DATA_ROOT, "dataset_path_cleaned.csv")
    CENTER_CSV_PATH = os.path.join(DATA_ROOT, "dataset_center_clean5.csv")
    SAVE_DIR = "./checkpoints/experiment_full_15frames_refined"  # 建议改个名字，以免覆盖旧实验

    # ==================================================
    # 数据配置 
    # ==================================================
    IMG_SIZE = 256
    NUM_FRAMES = 15          # ±7 + center
    INPUT_CHANNELS = 14

    # ==================================================
    # 训练配置
    # ==================================================
    BATCH_SIZE = 3
    EPOCHS = 150             
    LR = 2.5e-5
    NUM_WORKERS = 6
    STEPS_PER_EPOCH = 2000
    VAL_STEPS = 200
    
    # ==================================================
    # 日志 / 可视化频率
    # ==================================================
    LOG_INTERVAL = 100        # 改为每 100 step 打印一次，反馈更及时
    PREVIEW_INTERVAL = 200    # 每 500 step 保存一次预览图
    VAL_INTERVAL = 1          # 每 1 个 epoch 验证

    # ==================================================
    # 模型参数 (MS2TAN)
    # ==================================================
    MODEL_CONFIG = {
        # Transformer 结构参数
        "dim_list": [256, 192, 128],
        "num_frame": NUM_FRAMES,
        "image_size": IMG_SIZE,
        "patch_list": [16, 32, 8],
        "in_chans": INPUT_CHANNELS,
        "out_chans": INPUT_CHANNELS,
        "depth_list": [2, 2, 2],
        "heads_list": [8, 6, 4],
        "dim_head_list": [32, 32, 32],

        # 功能开关
        "missing_mask": True,
        "enable_model": True,
        
        # ✅ 核心修改：开启 CNN Refinement 消除块状伪影
        "enable_conv": True,
        
        # Loss 开关 (network.py 实际上不怎么用这些了，主要靠 train.py)
        "enable_mse": True,
        "enable_struct": True,
        "enable_percept": True,
    }

# 自动创建保存目录
os.makedirs(Config.SAVE_DIR, exist_ok=True)