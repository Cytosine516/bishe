import os
import csv

# =====================================================
# 路径配置
# =====================================================

# 原始 Patch TIF 根目录（待删除的是这里）
SRC_ROOT = "/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_Patches_FullBand"

# 多个 NPY 存储根目录（按需增减）
DST_ROOTS = [
    "/media/chai/NewDisk/Xinjiang_Tianshan_NPY",
    "/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_NPY"
]

# 输出 CSV 报告
OUT_CSV = "scene_completion_report.csv"

# =====================================================
# 工具函数
# =====================================================

def list_all_scenes(src_root):
    """
    扫描 SRC_ROOT 下所有 (Region, Scene)
    """
    scenes = []
    for region in sorted(os.listdir(src_root)):
        region_path = os.path.join(src_root, region)
        if not os.path.isdir(region_path):
            continue
        for scene in sorted(os.listdir(region_path)):
            scene_path = os.path.join(region_path, scene)
            if os.path.isdir(scene_path):
                scenes.append((region, scene))
    return scenes


def count_tif_patches(scene_path):
    """
    统计一个 Scene 下的 Patch 数
    默认使用 B01 波段（最稳定）
    """
    band_dir = os.path.join(scene_path, "B01")
    if not os.path.exists(band_dir):
        return 0
    return len([
        f for f in os.listdir(band_dir)
        if f.endswith(".tif")
    ])


def count_npy_patches_multi_root(region, scene):
    """
    统计一个 Scene 在所有 DST_ROOTS 中的 NPY 总数
    返回:
      -1 : 所有磁盘都不存在该 Scene
      >=0: NPY 总数
    """
    total = 0
    found_any = False

    for root in DST_ROOTS:
        scene_dir = os.path.join(root, region, scene)
        if os.path.exists(scene_dir):
            found_any = True
            total += len([
                f for f in os.listdir(scene_dir)
                if f.endswith(".npy")
            ])

    if not found_any:
        return -1
    return total


# =====================================================
# 主程序
# =====================================================

def main():
    scenes = list_all_scenes(SRC_ROOT)

    print(f"🔍 共发现 {len(scenes)} 个 Scene，开始完整性校验…")
    print("-" * 80)

    report = []

    for region, scene in scenes:
        scene_path = os.path.join(SRC_ROOT, region, scene)

        tif_count = count_tif_patches(scene_path)
        npy_count = count_npy_patches_multi_root(region, scene)

        if npy_count == -1:
            status = "NPY_MISSING"
        elif tif_count > 0 and npy_count == tif_count:
            status = "READY_TO_DELETE"
        else:
            status = "INCOMPLETE"

        report.append({
            "Region": region,
            "Scene": scene,
            "TIF_Patch_Count": tif_count,
            "NPY_Count": npy_count,
            "Status": status
        })

        print(
            f"{region:<30} | "
            f"{scene:<45} | "
            f"TIF={tif_count:5d} | "
            f"NPY={npy_count:5d} | "
            f"{status}"
        )

    # =================================================
    # 写 CSV 报告
    # =================================================
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Region",
                "Scene",
                "TIF_Patch_Count",
                "NPY_Count",
                "Status"
            ]
        )
        writer.writeheader()
        writer.writerows(report)

    print("-" * 80)
    print(f"📄 Scene 完成度报告已生成: {OUT_CSV}")
    print("状态说明：")
    print("  READY_TO_DELETE : NPY 已完整生成，可安全删除 Patch TIF")
    print("  INCOMPLETE      : 尚未完全生成（继续跑 preprocess）")
    print("  NPY_MISSING     : 所有磁盘均未发现该 Scene 的 NPY")

# =====================================================
# 入口
# =====================================================

if __name__ == "__main__":
    main()
