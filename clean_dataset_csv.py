import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# ==============================================================
# 配置区域
# ==============================================================

# 1. 原始 CSV 路径
INPUT_CSV_PATH = r"/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_Patches_FullBand/dataset_path_cleaned.csv"

# 2. 清洗后保存的 CSV 路径 (建议用这个新名字)
OUTPUT_CSV_PATH = r"/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_Patches_FullBand/dataset_path_deep_cleaned.csv"

# 3. 你的 NPY 根目录列表
NPY_ROOTS = [
    "/media/chai/NewDisk/Xinjiang_Tianshan_NPY",
    "/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_NPY",
]

# ==============================================================
# 主逻辑
# ==============================================================

def get_npy_path(row, roots):
    """查找文件路径"""
    region = str(row["Region"])
    scene = str(row["Scene"])
    patch_id = str(row["Patch_ID"])
    rel_path = os.path.join(region, scene, f"{patch_id}.npy")

    for root in roots:
        full_path = os.path.join(root, rel_path)
        if os.path.exists(full_path):
            return full_path
    return None

def verify_file(path):
    """
    尝试加载文件，如果损坏则返回 False
    """
    try:
        # allow_pickle=True 以防万一，mmap_mode='r' 避免完全读入内存加快速度
        # 但为了检测 EOF 错误，有时必须尝试读取数据
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 尝试加载
            data = np.load(path, mmap_mode=None) 
            # 简单的完整性检查：检查是否能读取形状
            _ = data.shape 
        return True
    except (EOFError, ValueError, OSError, Exception) as e:
        print(f"\n[损坏] Found corrupted file: {path} | Error: {e}")
        return False

def deep_clean():
    print(f"🚀 开始深度清洗 (验证文件完整性)...")
    print(f"📂 读取 CSV: {INPUT_CSV_PATH}")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print("❌ 找不到输入 CSV 文件")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    total = len(df)
    valid_indices = []
    bad_count = 0
    missing_count = 0

    print("🔍 正在扫描所有 .npy 文件 (这可能需要几分钟)...")

    for idx, row in tqdm(df.iterrows(), total=total, unit="img"):
        path = get_npy_path(row, NPY_ROOTS)
        
        if path is None:
            missing_count += 1
        else:
            # 找到文件了，现在检查是否损坏
            if verify_file(path):
                valid_indices.append(idx)
            else:
                bad_count += 1

    # 生成新表格
    df_clean = df.loc[valid_indices].reset_index(drop=True)

    print("-" * 50)
    print(f"📊 扫描完成")
    print(f"🔴 丢失文件 (Missing): {missing_count}")
    print(f"💀 损坏文件 (Corrupted): {bad_count}")
    print(f"🟢 有效文件 (Valid): {len(df_clean)} / {total}")
    print("-" * 50)

    if len(df_clean) > 0:
        df_clean.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"💾 已保存清洗后的 CSV 到: {OUTPUT_CSV_PATH}")
        print(f"⚠️ 请记得修改 config.py 中的 CSV_PATH！")
    else:
        print("❌ 所有文件都无法读取，请检查路径配置！")

if __name__ == "__main__":
    deep_clean()