import os
import pandas as pd
import numpy as np
import rasterio
import shutil
from tqdm import tqdm
import concurrent.futures
from collections import defaultdict

# ==========================================================
# 基本配置
# ==========================================================

SRC_ROOT = "/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_Patches_FullBand"

# 定义多个存储根目录，程序会按顺序检查剩余空间
DST_ROOTS = [
    "/media/chai/NewDisk/Xinjiang_Tianshan_NPY",
    "/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_NPY"
]

CSV_PATH = os.path.join(SRC_ROOT, "dataset_path_fixed.csv")
NUM_WORKERS = 16   

# 阈值：如果磁盘剩余空间小于 5GB，则切换到下一个路径
MIN_FREE_SPACE_GB = 5 

# ==========================================================
# 工具函数
# ==========================================================

def get_free_space_gb(path):
    """获取路径所在磁盘的剩余空间（GB）"""
    # 如果路径不存在，检查其父目录
    check_path = path
    while not os.path.exists(check_path):
        check_path = os.path.dirname(check_path)
    total, used, free = shutil.disk_usage(check_path)
    return free / (1024**3)

def get_valid_save_path(region, scene, patch_id):
    """
    根据磁盘空间返回可用的保存路径。
    优先检查是否已经存在于任何一个 DST_ROOT 中（断点续传）。
    """
    filename = f"{patch_id}.npy"
    relative_path = os.path.join(region, scene, filename)

    # 1. 检查是否已经存在（断点续传逻辑）
    for root in DST_ROOTS:
        full_path = os.path.join(root, relative_path)
        if os.path.exists(full_path):
            return full_path, "exists"

    # 2. 寻找有足够空间的磁盘
    for root in DST_ROOTS:
        if get_free_space_gb(root) > MIN_FREE_SPACE_GB:
            full_path = os.path.join(root, relative_path)
            return full_path, "new"
            
    return None, "full"

# ==========================================================
# 波段定义
# ==========================================================

INT_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
FLOAT_BANDS = ["AOT", "WVP"]
LABEL_BANDS = ["SCL"]

# ==========================================================
# 单样本处理函数
# ==========================================================

def process_one_sample(args):
    region, scene, patch_id = args

    # 获取当前可用的存储路径
    save_path, status = get_valid_save_path(region, scene, patch_id)
    
    if status == "exists":
        return "skipped"
    if status == "full":
        return "disk_full"

    base_path = os.path.join(SRC_ROOT, region, scene)
    channels = []

    try:
        # 1️⃣ 连续光谱波段
        for band in INT_BANDS:
            tif = os.path.join(base_path, band, f"{patch_id}.tif")
            if not os.path.exists(tif): return "missing"
            with rasterio.open(tif) as src:
                channels.append(src.read(1).astype(np.uint16))

        # 2️⃣ 大气变量
        for band in FLOAT_BANDS:
            tif = os.path.join(base_path, band, f"{patch_id}.tif")
            if not os.path.exists(tif): return "missing"
            with rasterio.open(tif) as src:
                channels.append(src.read(1).astype(np.float32))

        # 3️⃣ SCL
        scl_tif = os.path.join(base_path, "SCL", f"{patch_id}.tif")
        if not os.path.exists(scl_tif): return "missing"
        with rasterio.open(scl_tif) as src:
            channels.append(src.read(1).astype(np.uint8))

        data = np.stack(channels, axis=0)  # (15, 256, 256)

        # 确保目录存在并保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, data)

        return "ok"

    except Exception as e:
        return f"error: {e}"

# ==========================================================
# 主程序
# ==========================================================

def main():
    print("🚀 Sentinel-2 TIF → NPY (多盘自动切换模式)")
    print(f"📂 来源: {SRC_ROOT}")
    print(f"💾 目标列表: {DST_ROOTS}")
    print(f"⚠️  临界空间: {MIN_FREE_SPACE_GB} GB")
    print("-" * 60)

    df = pd.read_csv(CSV_PATH)
    tasks = [(row["Region"], row["Scene"], row["Patch_ID"]) for _, row in df.iterrows()]
    stats = defaultdict(int)

    # 使用 ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = executor.map(process_one_sample, tasks)

        for result in tqdm(futures, total=len(tasks), unit="patch", desc="🛰️ Processing"):
            stats[result] += 1
            # 如果所有磁盘都满了，可以考虑提前终止
            if result == "disk_full" and stats["disk_full"] > 10: 
                print("\n❌ 所有指定的存储空间已耗尽！程序退出。")
                break

    print("\n✅ 转换完成统计：")
    for k, v in stats.items():
        print(f"  {k:>12s}: {v}")

if __name__ == "__main__":
    main()