import pystac_client
import planetary_computer
import os
import requests
import time
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TransferSpeedColumn,
    TimeRemainingColumn
)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= 配置区域 =================
BASE_SAVE_DIR = "/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_Dataset"

ROIS = {
    #"Region_01_Kashgar_Mixed":   [75.79, 39.36, 76.26, 39.63],
    #"Region_02_Aksu_Cotton":     [80.03, 41.04, 80.51, 41.31],
    #"Region_03_Korla_Ecotone":   [85.98, 41.59, 86.46, 41.86],
    #"Region_04_Alar_Transition": [81.09, 40.79, 81.56, 41.06],
    "Region_05_Wensu_Foothills": [80.08, 41.74, 80.56, 42.01],
}

DATE_RANGE = "2024-12-01/2024-12-31"
MAX_CLOUD_COVER = 100
MAX_WORKERS = 6
MAX_PENDING_FUTURES = 20     # 硬限制，防止线程池堆死
DOWNLOAD_STALL_TIMEOUT = 120 # 秒：无数据即中断
# ===========================================

STOP_EVENT = threading.Event()

def signal_handler(signum, frame):
    if not STOP_EVENT.is_set():
        print("\n检测到 Ctrl+C，正在安全退出（等待当前任务结束）...")
        STOP_EVENT.set()

signal.signal(signal.SIGINT, signal_handler)

def get_session():
    session = requests.Session()
    retries = Retry(
        total=8,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

def download_single_asset(url, save_path, asset_name, session, progress, global_task_id):
    """
    Worker 线程函数：
    - 接收 progress 和 global_task_id 以更新总速率
    """
    if STOP_EVENT.is_set():
        return ("stopped", asset_name)

    temp_path = save_path + ".tmp"

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return ("skip", asset_name)

    for _ in range(3):
        try:
            with session.get(url, stream=True, timeout=(10, 30)) as r:
                r.raise_for_status()
                last_data_time = time.time()

                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if STOP_EVENT.is_set():
                            raise RuntimeError("Stopped")

                        if chunk:
                            f.write(chunk)
                            # === 新增：实时更新全局下载量，从而计算总速率 ===
                            progress.update(global_task_id, advance=len(chunk))
                            last_data_time = time.time()

                        if time.time() - last_data_time > DOWNLOAD_STALL_TIMEOUT:
                            raise TimeoutError("Download stalled")

            os.replace(temp_path, save_path)
            return ("ok", asset_name)

        except Exception:
            time.sleep(2)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return ("fail", asset_name)

def main():
    print("连接 Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )
    session = get_session()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        # === 新增：添加全局总速率任务（置顶显示） ===
        # total=None 表示无限模式，专注于显示速率
        global_task_id = progress.add_task("[green]🚀 总下载速率", total=None)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            for region_name, bbox in ROIS.items():
                if STOP_EVENT.is_set():
                    break

                progress.print(f"\n=== 区域: {region_name} ===")
                region_dir = os.path.join(BASE_SAVE_DIR, region_name)
                os.makedirs(region_dir, exist_ok=True)

                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=DATE_RANGE,
                    query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
                )

                items = list(search.item_collection())
                progress.print(f"  找到 {len(items)} 景影像")

                futures = []

                for item in items:
                    if STOP_EVENT.is_set():
                        break

                    try:
                        item = planetary_computer.sign(item)
                    except Exception:
                        continue

                    date_str = item.datetime.date()
                    cloud = int(item.properties.get("eo:cloud_cover", 999))
                    scene_dir = os.path.join(
                        region_dir, f"{date_str}_{item.id}_Cloud{cloud}"
                    )
                    os.makedirs(scene_dir, exist_ok=True)

                    # ================= 修改区域开始 =================
                    assets = {
                        # === 原有波段 (已注释) ===
                        "B02": "B02_Blue.tif",
                        "B03": "B03_Green.tif",
                        "B04": "B04_Red.tif",
                        "B08": "B08_NIR.tif",
                        "B11": "B11_SWIR1.tif",
                        "B12": "B12_SWIR2.tif",
                        "SCL": "SCL_Mask.tif",

                        # === 其他波段 (新增下载目标) ===
                        "B01": "B01_Coastal.tif",        # 60m: 海岸/气溶胶
                        "B05": "B05_RedEdge1.tif",       # 20m: 植被红边1
                        "B06": "B06_RedEdge2.tif",       # 20m: 植被红边2
                        "B07": "B07_RedEdge3.tif",       # 20m: 植被红边3
                        "B8A": "B8A_NarrowNIR.tif",      # 20m: 窄波段近红外
                        "B09": "B09_WaterVapor.tif",     # 60m: 水蒸气
                        "AOT": "AOT_Aerosol.tif",        # 气溶胶光学厚度
                        "WVP": "WVP_WaterVaporMap.tif",  # 水蒸气图
                        
                        # 如果还需要真彩色预览图，可取消下方注释:
                        #"visual": "Visual_RGB.tif"
                    }
                    # ================= 修改区域结束 =================

                    for key, fname in assets.items():
                        if key not in item.assets:
                            # 某些早期数据可能缺失部分波段
                            continue

                        save_path = os.path.join(scene_dir, fname)
                        task_name = f"{date_str} {fname}"

                        task_id = progress.add_task(task_name, total=1)
                        
                        # === 修改：传递 progress 和 global_task_id ===
                        future = executor.submit(
                            download_single_asset,
                            item.assets[key].href, # url
                            save_path,             # save_path
                            task_name,             # asset_name
                            session,
                            progress,
                            global_task_id
                        )
                        future.task_id = task_id
                        futures.append(future)

                        # === 硬限制 futures 数量 ===
                        while len(futures) >= MAX_PENDING_FUTURES:
                            done, _ = wait(futures, timeout=1)
                            for f in done:
                                status, _ = f.result()
                                progress.update(f.task_id, advance=1)
                                progress.remove_task(f.task_id)
                                futures.remove(f)

                # 清空本区域剩余任务
                while futures:
                    done, _ = wait(futures, timeout=1)
                    for f in done:
                        status, _ = f.result()
                        progress.update(f.task_id, advance=1)
                        progress.remove_task(f.task_id)
                        futures.remove(f)

    print("\n程序已安全退出")
    if STOP_EVENT.is_set():
        print("下次运行将自动跳过已下载文件")

if __name__ == "__main__":
    main()