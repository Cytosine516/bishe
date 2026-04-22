import os
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.features import shapes
import numpy as np
import time
from datetime import datetime
from shapely.geometry import shape
from tqdm import tqdm

# ================= 1. 全局配置区域 (最终修正版) =================

# 1. 服务器/本机原有路径 (保持不变)
SERVER_INPUT_ROOT = r"/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_Dataset"

# 2. 3T 硬盘 (原 E 盘) -> 挂载点 /mnt/data_3t
LOCAL_E_INPUT_ROOT = r"/media/chai/Seagate Backup Plus Drive/Xinjiang_Tianshan_Dataset"

# 3. 输出路径列表
OUTPUT_DIRS = [
    # 原有的服务器输出路径 (保持不变)
    r"/media/chai/Data/wuyiru/Project_Root/data",
    
    r"/media/chai/Seagate Backup Plus Drive/Xinjiang_Tianshan_Cropped_Fixed"
]

# 定义每个 Region 的数据来源列表
# 键名必须与 ROIS 中的键名一致
REGION_SOURCE_MAP = {
    "Region_01_Kashgar_Mixed":   [os.path.join(SERVER_INPUT_ROOT, "Region_01_Kashgar_Mixed")],
    
    "Region_02_Aksu_Cotton":     [os.path.join(LOCAL_E_INPUT_ROOT, "Region_02_Aksu_Cotton")],
    
    "Region_03_Korla_Ecotone":   [os.path.join(LOCAL_E_INPUT_ROOT, "Region_03_Korla_Ecotone")],
    
    "Region_04_Alar_Transition": [os.path.join(LOCAL_E_INPUT_ROOT, "Region_04_Alar_Transition")],
    
    # Region 05 同时从两个地方读取，程序会自动合并文件列表
    "Region_05_Wensu_Foothills": [
        os.path.join(SERVER_INPUT_ROOT, "Region_05_Wensu_Foothills"), # 2023-2024 (从原路径读取)
        os.path.join(LOCAL_E_INPUT_ROOT, "Region_05_Wensu_Foothills") # 2020-2022 (从3T盘读取)
    ]
}

# ROI 坐标定义 (保持不变)
ROIS = {
    #"Region_01_Kashgar_Mixed":    [75.79, 39.36, 76.26, 39.63],
    #"Region_02_Aksu_Cotton":      [80.03, 41.04, 80.51, 41.31],
    #"Region_03_Korla_Ecotone":    [85.98, 41.59, 86.46, 41.86],
    #"Region_04_Alar_Transition":  [81.09, 40.79, 81.56, 41.06],
    "Region_05_Wensu_Foothills":  [80.08, 41.74, 80.56, 42.01]
}

# 目标波段
TARGET_BANDS = [
    "B01", "B02", "B03", "B04",      
    "B05", "B06", "B07",             
    "B08", "B8A",                    
    "B09", "B11", "B12",                    
    "SCL", "AOT", "WVP"              
]

# 报告文件将保存到第一个输出目录
REPORT_FILE_PATH = os.path.join(OUTPUT_DIRS[0], "dataset_process_report.txt")

# =================================================

class CloudStats:
    def __init__(self):
        self.records = []

    def add_record(self, region, scene_id, cloud_pct, category, note=""):
        self.records.append({
            "region": region,
            "scene": scene_id,
            "cloud_pct": cloud_pct,
            "category": category,
            "note": note
        })

    def generate_report(self, save_path):
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"处理报告 (多源合并版) - {datetime.now()}\n")
                f.write("=" * 110 + "\n")
                f.write(f"{'Region':<30} | {'Scene':<40} | {'Cloud%':>8} | {'Status'}\n")
                f.write("-" * 110 + "\n")
                for r in sorted(self.records, key=lambda x: (x['region'], x['scene'])):
                    pct = f"{r['cloud_pct']:.2f}%" if r['cloud_pct'] >= 0 else "N/A"
                    f.write(
                        f"{r['region']:<30} | {r['scene']:<40} | {pct:>8} | "
                        f"{r['category']} {r['note']}\n"
                    )
            print(f"\n[报告] 已保存至: {save_path}")
        except Exception as e:
            print(f"[警告] 无法写入报告: {e}")

def parse_scene_info(scene_name):
    try:
        parts = scene_name.split("_")
        date_str = parts[0]
        if "S2A" in scene_name:
            satellite = "S2A"
        elif "S2B" in scene_name:
            satellite = "S2B"
        else:
            satellite = "UNKNOWN"
        return date_str, satellite
    except:
        return "UNKNOWN", "UNKNOWN"

def check_bbox_intersection(bbox1, bbox2):
    l1, b1, r1, t1 = bbox1
    l2, b2, r2, t2 = bbox2
    return not (l1 > r2 or l2 > r1 or b1 > t2 or b2 > t1)

def calculate_cloud_percentage(scl_data):
    valid_pixels = np.sum(scl_data != 0)
    if valid_pixels == 0: return 100.0
    # SCL: 3=Cloud Shadows, 8=Cloud Medium, 9=Cloud High, 10=Cirrus
    cloud_count = np.sum(np.isin(scl_data, [3, 8, 9, 10]))
    return (cloud_count / valid_pixels) * 100.0

def get_valid_data_geometry(scl_data, transform):
    mask = (scl_data != 0).astype(np.uint8)
    if not np.any(mask): return None
    geoms = []
    # 提取有效区域形状
    for geom, val in shapes(mask, mask=mask, transform=transform):
        if val == 1: geoms.append(shape(geom))
    if not geoms: return None
    from shapely.ops import unary_union
    return unary_union(geoms).buffer(0)

def crop_scene(
    scene_path,
    output_roots, 
    roi_wgs84,
    bands,
    stats,
    region_key,
    coverage_cache
):
    scene_name = os.path.basename(scene_path)

    # 1. 安全读取文件列表并检查完整性
    try:
        all_files = [f for f in os.listdir(scene_path) if f.lower().endswith(".tif")]
        if not all_files:
            stats.add_record(region_key, scene_name, -1, "Skipped", "(空文件夹)")
            return
            
        # === 🛡️ 严格完整性检查 (缺一不可) ===
        missing_bands = []
        for target_band in bands:
            # 只要文件名包含波段名(如 'B02')就算存在
            if not any(target_band in f for f in all_files):
                missing_bands.append(target_band)
        
        if missing_bands:
            # 只要有缺失，直接跳过整个场景
            print(f"  [跳过] 场景不完整 {scene_name[:15]}... 缺失: {missing_bands}")
            stats.add_record(region_key, scene_name, -1, "Skipped", f"(缺失波段: {len(missing_bands)}个)")
            return
        # ========================================

    except Exception as e:
        print(f"  [警告] 读取目录出错 {scene_path}: {e}")
        return

    # 2. 寻找 SCL
    scl_name = next((f for f in all_files if "SCL" in f), None)
    if not scl_name: return # 理论上上面检查过，双重保险

    scl_path = os.path.join(scene_path, scl_name)

    # === 🛡️ SCL 处理保护 ===
    try:
        with rasterio.open(scl_path) as src:
            # 3. 计算 ROI 投影
            min_lon, min_lat, max_lon, max_lat = roi_wgs84
            warped_bounds = transform_bounds(
                "EPSG:4326",
                src.crs,
                min_lon, min_lat, max_lon, max_lat,
                densify_pts=21
            )

            if not check_bbox_intersection(src.bounds, warped_bounds):
                stats.add_record(region_key, scene_name, -1, "Skipped", "(无重叠)")
                return

            # 4. 读取 SCL 数据
            scl_window = from_bounds(*warped_bounds, transform=src.transform)
            scl_window = scl_window.round_offsets().round_lengths()
            scl_data = src.read(1, window=scl_window, boundless=True, fill_value=0)
            window_transform = src.window_transform(scl_window)
            
            # 5. 有效性检查 (ROI内是否有数据)
            current_valid_geom = get_valid_data_geometry(scl_data, window_transform)
            if current_valid_geom is None or current_valid_geom.is_empty:
                 stats.add_record(region_key, scene_name, -1, "Skipped", "(ROI内无有效数据)")
                 return

            # 6. 同日去重逻辑
            date_str, satellite = parse_scene_info(scene_name)
            cache_key = (date_str, satellite)

            if satellite != "UNKNOWN":
                if cache_key not in coverage_cache:
                    coverage_cache[cache_key] = []

                is_covered = False
                for _, prev_geom in coverage_cache[cache_key]:
                    # 如果被之前的同日影像包含，则跳过
                    if prev_geom.contains(current_valid_geom):
                        stats.add_record(region_key, scene_name, -1, "Skipped", f"(被同日影像覆盖)")
                        is_covered = True
                        break
                
                if is_covered: return
                coverage_cache[cache_key].append((scene_name, current_valid_geom))

            # 7. 计算云量
            cloud_pct = calculate_cloud_percentage(scl_data)

    except Exception as e:
        print(f"  [错误] SCL 读取失败: {scene_name} - {e}")
        stats.add_record(region_key, scene_name, -1, "Error", "(SCL损坏)")
        return

    report_tag = "Processed"
    stats.add_record(region_key, scene_name, cloud_pct, report_tag)
    print(f"  [处理中] {scene_name[:18]}... 云量: {cloud_pct:.2f}%")

    # 8. 循环输出到所有目标路径
    for out_root in output_roots:
        target_dir = os.path.join(out_root, region_key, scene_name)
        dir_created = False 

        for b_name in bands:
            f_name = next((f for f in all_files if b_name in f), None)
            
            src_file = os.path.join(scene_path, f_name)
            dst_file = os.path.join(target_dir, f"Crop_{f_name}")

            # 延迟创建文件夹：只有确实能读取到文件时才创建，避免生成空文件夹
            if not dir_created:
                 pass

            # === 🛡️ 波段文件读取保护 ===
            try:
                # 检查是否为 0 字节文件
                if os.path.getsize(src_file) == 0:
                    print(f"    [致命] 发现0字节文件，跳过整个场景: {f_name}")
                    return # 只要有一个坏的，整个场景都不要了，保证原子性

                with rasterio.open(src_file) as src:
                    window = from_bounds(*warped_bounds, transform=src.transform)
                    window = window.round_offsets().round_lengths()
                    
                    # 确定可以读写后，创建输出目录
                    if not dir_created:
                        os.makedirs(target_dir, exist_ok=True)
                        dir_created = True
                        
                    # 断点续传：文件已存在则跳过
                    if os.path.exists(dst_file): continue

                    data = src.read(1, window=window, boundless=True, fill_value=0)

                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": data.shape[0],
                        "width": data.shape[1],
                        "transform": src.window_transform(window),
                        "compress": "lzw" # 压缩节省空间
                    })

                    with rasterio.open(dst_file, "w", **out_meta) as dst:
                        dst.write(data, 1)
                        
            except Exception as e:
                print(f"    [错误] 波段写入失败 {f_name}: {e}")
                return # 写入失败，停止处理该场景

def main():
    print("=== 开始处理 (混合路径修正版) ===")
    for idx, path in enumerate(OUTPUT_DIRS):
        print(f"输出目标 {idx+1}: {path}")

    stats = CloudStats()

    for region_key, roi_bounds in ROIS.items():
        print(f"\n{'-' * 50}")
        print(f"🚀 正在扫描区域: {region_key}")

        # 获取该区域的所有源路径
        source_paths = REGION_SOURCE_MAP.get(region_key, [])
        if not source_paths:
            print("  [警告] 该区域未配置源路径，跳过。")
            continue

        # 收集所有场景 (Scene)，并按场景名去重
        all_scenes = {} 
        
        for sp in source_paths:
            # 检查路径是否存在
            if not os.path.exists(sp):
                print(f"  [提示] 路径不存在 (跳过): {sp}")
                continue
            
            print(f"  -> 扫描路径: {sp}")
            try:
                scenes = os.listdir(sp)
                for sc in scenes:
                    full_path = os.path.join(sp, sc)
                    if os.path.isdir(full_path):
                        if sc not in all_scenes:
                            all_scenes[sc] = full_path
            except Exception as e:
                print(f"  [错误] 扫描出错: {e}")

        sorted_scenes = sorted(all_scenes.keys())
        print(f"📂 共合并发现 {len(sorted_scenes)} 个独立场景")

        scene_coverage_cache = {}

        # 开始处理
        for scene_name in tqdm(sorted_scenes, desc="处理进度"):
            scene_path = all_scenes[scene_name]
            crop_scene(
                scene_path,
                OUTPUT_DIRS, 
                roi_bounds,
                TARGET_BANDS,
                stats,
                region_key,
                scene_coverage_cache
            )

    print("\n=== 全部完成 ===")
    stats.generate_report(REPORT_FILE_PATH)

if __name__ == "__main__":
    main()