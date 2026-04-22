import os
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.features import shapes
import numpy as np
import time
from datetime import datetime
from shapely.geometry import shape, box, Mapping

# ================= 1. 全局配置区域 =================

INPUT_BASE_DIR = r"F:\\Xinjiang_Tianshan_Dataset"
OUTPUT_BASE_DIR = r"F:\\Xinjiang_Tianshan_Cropped_Fixed_V2"  # 建议换个输出目录避免混淆

REPORT_FILE_PATH = os.path.join(OUTPUT_BASE_DIR, "dataset_process_report.txt")

ROIS = {
    "Region_01_Kashgar_Mixed":    [75.79, 39.36, 76.26, 39.63],
    "Region_02_Aksu_Cotton":      [80.03, 41.04, 80.51, 41.31],
    "Region_03_Korla_Ecotone":    [85.98, 41.59, 86.46, 41.86],
    "Region_04_Alar_Transition":  [81.09, 40.79, 81.56, 41.06],
    "Region_05_Wensu_Foothills":  [80.08, 41.74, 80.56, 42.01]
}

FOLDER_MAPPING = {
    k: k for k in ROIS.keys()
}

# === Sentinel-2 L2A 所有可用波段 ===
TARGET_BANDS = [
    "B01", "B02", "B03", "B04",      # 60m / 10m
    "B05", "B06", "B07",             # 20m (Red Edge)
    "B08", "B8A",                    # 10m / 20m (NIR)
    "B09",                           # 60m (Water Vapor)
    "B11", "B12",                    # 20m (SWIR)
    "SCL", "AOT", "WVP"              # Masks & Maps
]

# 虽然不再分文件夹，但保留阈值用于报告中的参考标签
CLEAR_THRESHOLD = 5.0
CLOUDY_THRESHOLD = 30.0

# =================================================


class CloudStats:
    def __init__(self):
        self.records = []
        self.start_time = time.time()

    def add_record(self, region, scene_id, cloud_pct, category, note=""):
        self.records.append({
            "region": region,
            "scene": scene_id,
            "cloud_pct": cloud_pct,
            "category": category,
            "note": note
        })

    def generate_report(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"处理报告 (精确有效区域去重) - {datetime.now()}\n")
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


def parse_scene_info(scene_name):
    """解析场景名称获取日期和卫星编号"""
    parts = scene_name.split("_")
    date_str = parts[0]  # Assuming YYYYMMDD is first
    
    # 简单的卫星判断逻辑，可根据实际文件名调整
    if "S2A" in scene_name:
        satellite = "S2A"
    elif "S2B" in scene_name:
        satellite = "S2B"
    else:
        satellite = "UNKNOWN"
    return date_str, satellite


def check_bbox_intersection(bbox1, bbox2):
    """粗略检查两个矩形是否相交 (l, b, r, t)"""
    l1, b1, r1, t1 = bbox1
    l2, b2, r2, t2 = bbox2
    return not (l1 > r2 or l2 > r1 or b1 > t2 or b2 > t1)


def calculate_cloud_percentage(scl_data):
    """基于 SCL 计算云量"""
    valid_pixels = np.sum(scl_data != 0)
    if valid_pixels == 0:
        return 100.0
    # SCL: 3=Cloud Shadows, 8=Cloud Medium, 9=Cloud High, 10=Cirrus
    cloud_count = np.sum(np.isin(scl_data, [3, 8, 9, 10]))
    return (cloud_count / valid_pixels) * 100.0


def get_valid_data_geometry(scl_data, transform):
    """
    将 SCL 数据中非 0 (有效数据) 的部分提取为 shapely Geometry。
    如果有多个不连续区域，会返回 MultiPolygon。
    """
    # 创建掩膜：1 代表有效数据，0 代表无数据
    mask = (scl_data != 0).astype(np.uint8)
    
    # 如果全是 0，返回空
    if not np.any(mask):
        return None

    # 提取形状 (features)
    # shapes 返回的是 (geojson_geometry, value) 的生成器
    # 我们只关心 value=1 的区域
    geoms = []
    for geom, val in shapes(mask, mask=mask, transform=transform):
        if val == 1:
            geoms.append(shape(geom))
    
    if not geoms:
        return None
    
    # 将所有有效碎片合并为一个几何体
    from shapely.ops import unary_union
    union_geom = unary_union(geoms)
    
    # 为了容错，稍微做一点 buffer(0) 修复可能的拓扑错误
    return union_geom.buffer(0)


def crop_scene(
    scene_path,
    output_root,
    roi_wgs84,
    bands,
    stats,
    region_key,
    coverage_cache
):
    scene_name = os.path.basename(scene_path)

    try:
        all_files = [f for f in os.listdir(scene_path) if f.lower().endswith(".tif")]
    except FileNotFoundError:
        return

    # 1. 寻找 SCL
    scl_name = next((f for f in all_files if "SCL" in f), None)
    if not scl_name:
        stats.add_record(region_key, scene_name, -1, "Skipped", "(No SCL)")
        return

    scl_path = os.path.join(scene_path, scl_name)

    with rasterio.open(scl_path) as src:
        src_crs = src.crs
        src_bounds = src.bounds

        # 2. 计算 ROI 在当前投影下的边界
        min_lon, min_lat, max_lon, max_lat = roi_wgs84
        warped_bounds = transform_bounds(
            "EPSG:4326",
            src_crs,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            densify_pts=21
        )

        # 粗筛：如果 ROI 框和影像框完全不挨着，直接跳过
        if not check_bbox_intersection(src_bounds, warped_bounds):
            stats.add_record(region_key, scene_name, -1, "Skipped", "(无重叠)")
            return

        # 3. 读取 ROI 区域内的 SCL 数据
        scl_window = from_bounds(*warped_bounds, transform=src.transform)
        scl_window = scl_window.round_offsets().round_lengths()
        
        # 读取数据 (保留空间参考以便转换坐标)
        scl_data = src.read(1, window=scl_window, boundless=True, fill_value=0)
        window_transform = src.window_transform(scl_window)
        
        # 4. === 核心修改：计算实际数据的几何形状 ===
        current_valid_geom = get_valid_data_geometry(scl_data, window_transform)
        
        if current_valid_geom is None or current_valid_geom.is_empty:
             stats.add_record(region_key, scene_name, -1, "Skipped", "(ROI内无有效数据)")
             return

        # 5. ========= 基于实际几何的去重逻辑 =========
        date_str, satellite = parse_scene_info(scene_name)
        cache_key = (date_str, satellite)

        if satellite != "UNKNOWN":
            if cache_key not in coverage_cache:
                coverage_cache[cache_key] = []

            # 检查当前有效区域是否被之前的某个场景完全覆盖
            is_covered = False
            for prev_scene, prev_geom in coverage_cache[cache_key]:
                # 容差处理：如果 prev_geom 包含了 current 的 99.9% 也可以认为覆盖
                # 这里使用 strict contains: prev_geom.contains(current_valid_geom)
                if prev_geom.contains(current_valid_geom):
                    stats.add_record(
                        region_key, scene_name, -1,
                        "Skipped", f"(被 {prev_scene} 覆盖)"
                    )
                    is_covered = True
                    break
            
            if is_covered:
                return

            # 如果当前场景没有被覆盖，检查它是否覆盖了之前的场景（如果是，标记之前的为移除）
            # 注意：实际文件已经生成了可能很难物理删除，这里主要是逻辑上的“更优选择”
            # 或者您可以选择在此处不做反向剔除，只做增量添加。
            # 为了简化，这里只做“当前是否被已有覆盖”，并将当前加入缓存。
            
            # 优化缓存：如果当前场景很大，或许能覆盖未来的场景
            coverage_cache[cache_key].append((scene_name, current_valid_geom))
        # ============================================

        # 6. 计算云量
        cloud_pct = calculate_cloud_percentage(scl_data)

    # 报告标签
    if cloud_pct <= CLEAR_THRESHOLD:
        report_tag = "Clear(Ref)"
    elif cloud_pct >= CLOUDY_THRESHOLD:
        report_tag = "Cloudy(Ref)"
    else:
        report_tag = "Normal(Ref)"

    stats.add_record(region_key, scene_name, cloud_pct, report_tag)
    print(f"  [Processing] {scene_name[:20]}... 云量: {cloud_pct:.2f}%")

    # 7. 输出文件
    target_dir = os.path.join(output_root, scene_name)
    os.makedirs(target_dir, exist_ok=True)

    for b_name in bands:
        f_name = next((f for f in all_files if b_name in f), None)
        if not f_name:
            continue

        src_file = os.path.join(scene_path, f_name)
        dst_file = os.path.join(target_dir, f"Crop_{f_name}")

        if os.path.exists(dst_file):
            continue

        with rasterio.open(src_file) as src:
            # 重新计算 window (虽然上面算过，但为了安全重新用 warped_bounds 生成)
            window = from_bounds(*warped_bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
            data = src.read(1, window=window, boundless=True, fill_value=0)

            out_meta = src.meta.copy()
            out_meta.update({
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": src.window_transform(window),
                "compress": "lzw" # 加上压缩以节省空间
            })

            with rasterio.open(dst_file, "w", **out_meta) as dst:
                dst.write(data, 1)


def main():
    print("=== 开始处理 (基于实际数据几何去重) ===")
    print(f"输入: {INPUT_BASE_DIR}")
    print(f"输出: {OUTPUT_BASE_DIR}")

    stats = CloudStats()

    for region_key, roi_bounds in ROIS.items():
        print(f"\n{'-' * 40}")
        print(f"🚀 正在处理区域: {region_key}")

        region_path = os.path.join(INPUT_BASE_DIR, FOLDER_MAPPING[region_key])
        if not os.path.exists(region_path):
            print(f"⚠️  路径不存在: {region_path}")
            continue

        scenes = sorted(os.listdir(region_path))
        print(f"📂 发现 {len(scenes)} 个场景")

        region_out_dir = os.path.join(OUTPUT_BASE_DIR, region_key)
        
        # 缓存：Key=(Date, Sat), Value=[(scene_name, valid_geometry), ...]
        scene_coverage_cache = {}

        # 建议：如果是为了保留数据量最大的，可以先按文件大小或某个粗略指标排序
        # 目前按文件名排序通常是按时间或Tile编号
        for scene in scenes:
            scene_full_path = os.path.join(region_path, scene)
            if os.path.isdir(scene_full_path):
                crop_scene(
                    scene_full_path,
                    region_out_dir,
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