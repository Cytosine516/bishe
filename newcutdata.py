import os
import rasterio
import numpy as np
import pandas as pd
from rasterio.windows import Window
from rasterio.enums import Resampling
from tqdm import tqdm
from datetime import datetime
import concurrent.futures
import time
import shutil
import subprocess

# ================= 🔧 核心配置区域 =================

# 输入路径
INPUT_ROOT = r"/media/chai/Seagate Backup Plus Drive/Xinjiang_Tianshan_Cropped_Fixed" 

# 输出路径 (本地硬盘/SSD - 处理速度快)
OUTPUT_ROOT = r"/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_Patches_FullBand"

# 备份路径 (移动硬盘)
BACKUP_ROOT = r"/media/chai/Seagate Backup Plus Drive/Xinjiang_Tianshan_Patches_FullBand"

# ⚡ 并行进程数 (保持 10 个不变)
MAX_WORKERS = 10

PATCH_SIZE = 256
STRIDE = 128 

# 目标波段
TARGET_BANDS = [
    "B01", "B02", "B03", "B04", 
    "B05", "B06", "B07", "B08", "B8A", 
    "B09", "B11", "B12", 
    "AOT", "WVP", 
    "SCL"         
]

# SCL 云像素值
CLOUD_CLASSES = [3, 8, 9, 10]

# =================================================

def rsync_backup(source, destination):
    """
    使用 rsync 将源目录增量同步到目标目录
    """
    if not os.path.exists(source):
        print(f"[备份跳过] 源目录不存在: {source}")
        return

    print(f"\n{'='*20}")
    print(f"🚀 开始备份数据...")
    print(f"📂 源: {source}")
    print(f"💾 目标: {destination}")
    print(f"{'='*20}")

    # 如果目标目录不存在，创建它
    os.makedirs(destination, exist_ok=True)

    # 构建命令
    # -a: 归档模式，保留文件属性
    # -v: 显示详细信息
    # -h: 人类可读的进度
    # --progress: 显示进度条
    # /: 确保源目录末尾有斜杠，表示同步目录内的内容
    cmd = ["rsync", "-avh", "--progress", source + "/", destination]
    
    try:
        # 调用系统命令
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # 实时打印输出，这样你可以看到 rsync 的进度
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ 备份成功完成！数据已同步至: {destination}")
        else:
            print(f"\n[警告] rsync 退出码: {process.returncode}，备份可能未完全成功。")
            
    except Exception as e:
        print(f"[错误] rsync 执行失败: {e}")
        print("[提示] 如果是 Windows 系统，请检查是否安装了 rsync (如通过 Git Bash 或 Cygwin)。Linux/Mac 通常自带。")

def get_scene_date(scene_name):
    try:
        date_str = scene_name.split('_')[0]
        dt = datetime.strptime(date_str, "%Y%m%d")
        return dt.year, dt.month
    except:
        return 0, 0

def calculate_cloud_ratio(scl_data):
    valid_mask = scl_data != 0
    valid_count = np.sum(valid_mask)
    if valid_count == 0:
        return 0.0, True
    cloud_count = np.sum(np.isin(scl_data, CLOUD_CLASSES))
    return (cloud_count / valid_count) * 100.0, False

def process_scene_safe(args):
    """
    多进程包装函数
    """
    scene_path, output_scene_root, scene_name = args
    
    # === 修改这里：用 tqdm.write 代替 print ===
    # 这样进度条就不会被打断了
    tqdm.write(f"[P-{os.getpid()}] 正在处理 -> {scene_name[:30]}...") 
    
    try:
        return process_scene_core(scene_path, output_scene_root, scene_name)
    except Exception as e:
        tqdm.write(f"\n[Error] 处理场景失败 {scene_name}: {e}") # 这里也改成 tqdm.write
        return []

def process_scene_core(scene_path, output_scene_root, scene_name):
    patch_records = []
    
    if not os.path.exists(scene_path):
        return []

    try:
        files = os.listdir(scene_path)
    except FileNotFoundError:
        return []

    # 1. 寻找 10m 参考波段
    ref_file_name = next((f for f in files if "B02" in f and f.endswith('.tif')), None)
    if not ref_file_name:
        return []

    ref_path = os.path.join(scene_path, ref_file_name)
    
    with rasterio.open(ref_path) as src:
        ref_meta = src.meta.copy()
        height, width = src.height, src.width
        ref_transform = src.transform
        
    if width < PATCH_SIZE or height < PATCH_SIZE:
        return []

    grid_coords = []
    for r in range(0, height - PATCH_SIZE + 1, STRIDE):
        for c in range(0, width - PATCH_SIZE + 1, STRIDE):
            grid_coords.append((r, c))

    year, month = get_scene_date(scene_name)
    
    # 提前创建波段文件夹
    scene_band_dirs = {}
    for band in TARGET_BANDS:
        band_dir = os.path.join(output_scene_root, band)
        # 多进程下 makedirs 偶尔会竞争，加 try 保护
        try:
            os.makedirs(band_dir, exist_ok=True)
        except:
            pass
        scene_band_dirs[band] = band_dir

    for r, c in grid_coords:
        patch_id = f"Patch_R{r}_C{c}"
        
        # --- SCL 处理 ---
        scl_file = next((f for f in files if "SCL" in f), None)
        if not scl_file: continue
        
        min_x, max_y = ref_transform * (c, r)
        max_x, min_y = ref_transform * (c + PATCH_SIZE, r + PATCH_SIZE)
        patch_bounds = (min_x, min_y, max_x, max_y)
        
        with rasterio.open(os.path.join(scene_path, scl_file)) as src_scl:
            scl_window = rasterio.windows.from_bounds(*patch_bounds, transform=src_scl.transform)
            scl_data = src_scl.read(1, window=scl_window, out_shape=(PATCH_SIZE, PATCH_SIZE), resampling=Resampling.nearest)
            
            cloud_pct, is_empty = calculate_cloud_ratio(scl_data)
            if is_empty: continue
        
        patch_records.append({
            "Scene": scene_name,
            "Patch_ID": patch_id,
            "Year": year,
            "Month": month,
            "Cloud_Rate": round(cloud_pct, 2)
        })

        # --- 保存 SCL ---
        scl_out_meta = ref_meta.copy()
        scl_out_meta.update({
            "height": PATCH_SIZE, "width": PATCH_SIZE,
            "transform": rasterio.transform.from_bounds(*patch_bounds, PATCH_SIZE, PATCH_SIZE),
            "dtype": rasterio.uint8, "count": 1, "driver": "GTiff", "compress": "lzw"
        })
        
        scl_save_path = os.path.join(scene_band_dirs["SCL"], f"{patch_id}.tif")
        if not os.path.exists(scl_save_path):
            with rasterio.open(scl_save_path, "w", **scl_out_meta) as dst:
                dst.write(scl_data.astype(rasterio.uint8), 1)

        # --- 保存其他波段 ---
        for band_key in TARGET_BANDS:
            if band_key == "SCL": continue
            
            fname = next((f for f in files if band_key in f and f.endswith(".tif")), None)
            if not fname: continue
            
            save_path = os.path.join(scene_band_dirs[band_key], f"{patch_id}.tif")
            if os.path.exists(save_path): continue

            with rasterio.open(os.path.join(scene_path, fname)) as src_band:
                window = rasterio.windows.from_bounds(*patch_bounds, transform=src_band.transform)
                data = src_band.read(1, window=window, out_shape=(PATCH_SIZE, PATCH_SIZE), resampling=Resampling.bilinear)
                
                out_meta = scl_out_meta.copy()
                out_meta.update({"dtype": data.dtype})
                
                with rasterio.open(save_path, "w", **out_meta) as dst:
                    dst.write(data, 1)

    return patch_records

def main():
    print(f"=== 开始多进程切片处理 (Workers: {MAX_WORKERS}) - 详细模式 ===")
    
    if not os.path.exists(INPUT_ROOT):
        print(f"[错误] 输入路径不存在: {INPUT_ROOT}")
        return

    regions = sorted([d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))])
    
    all_tasks = []
    print("正在扫描所有区域的任务队列...")
    
    for region in regions:
        region_path = os.path.join(INPUT_ROOT, region)
        try:
            scenes = os.listdir(region_path)
            for scene in scenes:
                scene_full_path = os.path.join(region_path, scene)
                if os.path.isdir(scene_full_path):
                    out_scene_root = os.path.join(OUTPUT_ROOT, region, scene)
                    all_tasks.append((scene_full_path, out_scene_root, scene))
        except:
            pass

    total_tasks = len(all_tasks)
    print(f"✅ 扫描完成！共发现 {total_tasks} 个场景待处理。")
    print(f"🚀 启动 {MAX_WORKERS} 个并行进程...")
    print("-" * 50)

    all_records = []
    
    # 这里使用 chunksize=1 可以让任务分发更平滑
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(process_scene_safe, all_tasks), 
            total=total_tasks, 
            unit="scene",
            desc="总进度"
        ))

    print("\n正在汇总数据...")
    for res in results:
        all_records.extend(res)

    if all_records:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        csv_path = os.path.join(OUTPUT_ROOT, "dataset_report_mp.csv")
        df = pd.DataFrame(all_records)
        # 如果存在旧 csv，可以选择合并覆盖，这里简单处理为覆盖
        df.to_csv(csv_path, index=False)
        print(f"🎉 切片处理完成！共生成 {len(all_records)} 个切片。")
    else:
        print("\n[警告] 未生成任何切片数据。")

    # ================= 📦 自动备份逻辑 =================
    # 无论是否有新切片，都尝试执行一次同步，防止漏掉文件
    rsync_backup(OUTPUT_ROOT, BACKUP_ROOT)
    # =================================================

if __name__ == "__main__":
    main()
