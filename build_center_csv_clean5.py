import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# 配置
# =========================
DATA_ROOT = "/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_Patches_FullBand"
SRC_CSV = os.path.join(DATA_ROOT, "dataset_path_cleaned.csv")
DST_CSV = os.path.join(DATA_ROOT, "dataset_center_clean5.csv")

NPY_ROOTS = [
    "/media/chai/NewDisk/Xinjiang_Tianshan_NPY",
    "/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_NPY",
]

IMG_SIZE = 256
CLOUD_CLASSES = [3, 8, 9, 10]

CSV_CLOUD_TH = 5      # 原 CSV 粗筛（%）
REAL_CLOUD_TH = 0.05 # 精筛（比例）


def find_npy(region, scene, patch_id):
    rel = os.path.join(str(region), str(scene), f"{patch_id}.npy")
    for root in NPY_ROOTS:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return None


def calc_real_cloud_ratio(npy_path):
    data = np.load(npy_path)

    img = data[:14]
    scl = data[14]

    is_cloud = np.isin(scl, CLOUD_CLASSES)
    is_nodata = (scl == 0) | (np.max(img, axis=0) == 0)

    invalid = is_cloud | is_nodata
    return invalid.sum() / (IMG_SIZE * IMG_SIZE)


def main():
    df = pd.read_csv(SRC_CSV)

    # =========================
    # Step 1：CSV 粗筛
    # =========================
    df = df[df["Cloud_Rate"] < CSV_CLOUD_TH].reset_index(drop=True)
    print(f"[INFO] After CSV filter: {len(df)}")

    new_rows = []

    # =========================
    # Step 2：精筛 + 重新算云量
    # =========================
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = find_npy(row["Region"], row["Scene"], row["Patch_ID"])
        if path is None:
            continue

        try:
            real_cloud = calc_real_cloud_ratio(path)
        except Exception:
            continue

        if real_cloud < REAL_CLOUD_TH:
            r = row.to_dict()
            r["Real_Cloud_Rate"] = real_cloud * 100.0
            new_rows.append(r)

    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(DST_CSV, index=False)

    print(f"[DONE] Clean center CSV saved:")
    print(f"       {DST_CSV}")
    print(f"       Total clean centers: {len(df_new)}")


if __name__ == "__main__":
    main()
