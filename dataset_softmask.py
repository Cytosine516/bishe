import os
import random
import torch
import numpy as np
import pandas as pd
from scipy import ndimage  # ✅ 新增：导入 scipy 形态学处理库
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        config,
        mode="train",
        split_ratio=(0.8, 0.1, 0.1),
        seed=42,
    ):
        self.config = config
        self.mode = mode
        self.num_frames = config.NUM_FRAMES
        self.half_window = self.num_frames // 2
        
        self.current_epoch = 0
        self.cloud_col_name = "Cloud_Rate" 

        self.npy_roots = [
            "/media/chai/NewDisk/Xinjiang_Tianshan_NPY",
            "/media/chai/Data/wuyiru/Project_Root/Xinjiang_Tianshan_NPY",
        ]

        self.df_all = pd.read_csv(config.CSV_PATH)
        self.df_center = pd.read_csv(config.CENTER_CSV_PATH)
        
        for df in [self.df_all, self.df_center]:
            if "Region" not in df.columns:
                df["Region"] = "Unknown"
            df["Location_ID"] = (
                df["Region"].astype(str) + "_" + df["Patch_ID"].astype(str)
            )

        all_locations = sorted(self.df_all["Location_ID"].unique())
        rng = random.Random(seed)
        rng.shuffle(all_locations)

        n_total = len(all_locations)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        if mode == "train":
            self.target_locations = all_locations[:n_train]
        elif mode == "val":
            self.target_locations = all_locations[n_train:n_train + n_val]
        else:
            self.target_locations = all_locations[n_train + n_val:]

        self.samples = []
        df_center = self.df_center[
            self.df_center["Location_ID"].isin(self.target_locations)
        ]
        
        for _, row in df_center.iterrows():
            self.samples.append((row["Location_ID"], row["Scene"]))
        
        rng.shuffle(self.samples)
        print(f"[{mode}] Samples (Clean Targets): {len(self.samples)}")

        self.grouped_records = {}
        grouped = self.df_all[self.df_all["Location_ID"].isin(self.target_locations)].groupby("Location_ID")
        
        for loc, group in grouped:
            self.grouped_records[loc] = group.sort_values("Scene").to_dict("records")

        # ✅ 定义形态学操作的结构元素 (十字形，用于后续的 5 像素膨胀)
        self.struct_element = ndimage.generate_binary_structure(2, 1)

        # ✅ 软权重掩膜：
        # - SCL / NoData 这类“硬无效”区域仍然使用 0.0
        # - 自定义亮云 / 阴影这类“可疑但不绝对”的区域不再一刀切置 0，
        #   而是给予较低的监督权重，避免误判雪 / 边缘异常时把正常像素彻底错杀。
        # 这里默认给 0.20；如 config 中定义了 UNCERTAIN_VALID_WEIGHT，则优先使用。
        self.uncertain_valid_weight = float(getattr(config, "UNCERTAIN_VALID_WEIGHT", 0.20))
        self.uncertain_valid_weight = max(0.0, min(1.0, self.uncertain_valid_weight))

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _get_cloud_range(self):
        if self.current_epoch < 3:
            return (5.0, 60.0)
        elif self.current_epoch < 5:
            return (5.0, 80.0)
        else:
            prob = random.random()
            if prob < 0.5:
                return (5.0, 60.0)
            elif prob < 0.8:
                return (60.0, 90.0)
            else:
                return (90.0, 100.0)

    def __len__(self):
        return len(self.samples)

    def find_npy_path(self, region, scene, patch_id):
        rel = os.path.join(str(region), str(scene), f"{patch_id}.npy")
        for root in self.npy_roots:
            p = os.path.join(root, rel)
            if os.path.exists(p):
                return p
        return None

    def random_shift_mask(self, mask):
        h, w = mask.shape
        shift_h = random.randint(-h // 4, h // 4)
        shift_w = random.randint(-w // 4, w // 4)
        shifted = np.roll(mask, (shift_h, shift_w), axis=(0, 1))
        return shifted

    def load_mask_only(self, record):
        path = self.find_npy_path(record["Region"], record["Scene"], record["Patch_ID"])
        if path is None:
            return None
        try:
            data = np.load(path, mmap_mode='r')
            scl = data[14] 
            is_cloud = np.isin(scl, [3, 8, 9, 10]).astype(np.float32)
            # ✅ 借用的掩膜也进行 5 像素膨胀，保证人工制造的洞特征一致
            is_cloud = ndimage.binary_dilation(is_cloud, structure=self.struct_element, iterations=5).astype(np.float32)
            return is_cloud
        except:
            return None

    def __getitem__(self, idx):
        try:
            loc_id, target_scene = self.samples[idx]
            
            records = self.grouped_records[loc_id] 
            T_total = len(records)
            
            base_t = -1
            for i, r in enumerate(records):
                if r["Scene"] == target_scene:
                    base_t = i
                    break
            
            if base_t == -1:
                return self.__getitem__(random.randint(0, len(self.samples) - 1))

            target_pos = random.randint(0, 14)
            start_t = base_t - target_pos
            
            indices = []
            for o in range(15):
                t = max(0, min(T_total - 1, start_t + o))
                indices.append(t)

            frames_np = []
            scl_invalids = []
            
            for t_idx in indices:
                rec = records[t_idx]
                path = self.find_npy_path(rec["Region"], rec["Scene"], rec["Patch_ID"])
                
                if path is None:
                    img = np.zeros((14, self.config.IMG_SIZE, self.config.IMG_SIZE), dtype=np.float32)
                    scl_inv = np.ones((self.config.IMG_SIZE, self.config.IMG_SIZE), dtype=bool)
                else:
                    data = np.load(path)
                    img = data[:14].astype(np.float32)
                    if img.max() > 10: img /= 10000.0
                    img = np.clip(img, 0, 1)

                    scl = data[14]
                    is_cloud_pixel = np.isin(scl, [3, 8, 9, 10])
                    is_nodata = (scl == 0) | (np.max(img, axis=0) == 0)
                    scl_inv = is_cloud_pixel | is_nodata

                frames_np.append(img)
                scl_invalids.append(scl_inv)

            # =========================================================
            # ✅ 新版物理先验：解决沙尘误判 + 增加云阴影过滤
            # =========================================================
            X_stack = np.stack(frames_np) 
            rgb = X_stack[:, [3, 2, 1], :, :] 
            
            brightness = np.mean(rgb, axis=1) 
            median_brightness = np.median(brightness, axis=0) 

            # 计算颜色的“白度/中立度”(Max RGB - Min RGB)
            max_c = np.max(rgb, axis=1)
            min_c = np.min(rgb, axis=1)
            color_diff = max_c - min_c # 极差越小，越接近纯白/灰

            frames = []
            obs_masks = []
            valid_masks = []
            out_scl_masks = []
            out_custom_clouds = []
            out_custom_shadows = []
            
            for i in range(len(indices)):
                img = X_stack[i]
                
                # 1. 薄云检测 (亮度极高，且颜色不能太偏黄/红)
                is_custom_cloud = (brightness[i] > (median_brightness + 0.25)) & (color_diff[i] < 0.15)
                
                # 2. 云阴影检测 (亮度极低)
                is_custom_shadow = (brightness[i] < (median_brightness - 0.15)) & (brightness[i] < 0.15)
                
                # ==========================================
                # ✅ 核心修改：向外扩大 5 个像素
                # ==========================================
                scl_dilated = ndimage.binary_dilation(scl_invalids[i], structure=self.struct_element, iterations=5)
                cloud_dilated = ndimage.binary_dilation(is_custom_cloud, structure=self.struct_element, iterations=5)
                shadow_dilated = ndimage.binary_dilation(is_custom_shadow, structure=self.struct_element, iterations=5)

                # =========================================================
                # ✅ 软权重 valid_mask（最小非侵入式修改）
                # ---------------------------------------------------------
                # 1) 输入给模型的 obs_mask 仍保持“激进二值掩膜”：
                #    只要被 SCL / 自定义亮云 / 阴影命中，就先从输入中挖掉。
                # 2) 参与 loss / metric 的 valid_mask 改为“软权重”：
                #    - SCL / NoData -> 0.0（硬无效）
                #    - 自定义亮云 / 阴影 -> low weight（默认 0.20）
                #    这样可以在尽量防漏云的同时，降低误判雪 / 边缘异常时的训练副作用。
                # =========================================================
                strict_invalid = scl_dilated
                uncertain_invalid = (cloud_dilated | shadow_dilated) & (~strict_invalid)

                # 输入掩膜：保持原先“更激进”的二值策略，防止漏云直接喂给模型。
                hard_invalid = strict_invalid | uncertain_invalid
                hard_valid = (1.0 - hard_invalid.astype(np.float32))

                # loss / metric 掩膜：仅把 SCL / NoData 设为 0，自定义检测区域降权而不彻底删除。
                soft_valid = np.ones_like(hard_valid, dtype=np.float32)
                soft_valid[strict_invalid] = 0.0
                soft_valid[uncertain_invalid] = self.uncertain_valid_weight

                obs_masks.append(torch.from_numpy(hard_valid).unsqueeze(0))
                valid_masks.append(torch.from_numpy(soft_valid).unsqueeze(0))
                
                # 分别保存各种 Mask 用于可视化 (1 为触发，0 为未触发)
                out_scl_masks.append(torch.from_numpy(scl_dilated.astype(np.float32)).unsqueeze(0))
                out_custom_clouds.append(torch.from_numpy(cloud_dilated.astype(np.float32)).unsqueeze(0))
                out_custom_shadows.append(torch.from_numpy(shadow_dilated.astype(np.float32)).unsqueeze(0))
                
                frames.append(torch.from_numpy(img))

            X = torch.stack(frames)
            y = X.clone()

            # =========================================================
            # 2. 生成 Artificial Masks & 时空管遮挡 (保持不变)
            # =========================================================
            art_masks = []

            all_external_candidates_with_dist = []
            for cand_idx, r in enumerate(records):
                if cand_idx not in indices:
                    dist = abs(cand_idx - base_t)
                    all_external_candidates_with_dist.append((dist, cand_idx, r))
            
            all_external_candidates_with_dist.sort(key=lambda x: x[0])
            all_external_candidates = [item[2] for item in all_external_candidates_with_dist]

            used_donors = set()
            tube_active = False
            tube_mask = None
            tube_duration = 0

            for i in range(len(indices)):
                current_cloud_rate = records[indices[i]][self.cloud_col_name]
                is_target_clean = (current_cloud_rate < 5.0) 

                if is_target_clean and (self.mode == "train" or self.mode == "val"):
                    
                    add_mask = False
                    borrowed_mask = None

                    if tube_active and tube_duration > 0:
                        add_mask = True
                        borrowed_mask = tube_mask
                        tube_duration -= 1
                    else:
                        tube_active = False
                        
                        if i == target_pos:
                            min_c, max_c = self._get_cloud_range()
                            add_mask = True
                        else:
                            prob = random.random()
                            if prob < 0.20:
                                add_mask = False
                            elif prob < 0.80:
                                add_mask = True
                                min_c, max_c = 5.0, 60.0
                            elif prob < 0.90:
                                add_mask = True
                                min_c, max_c = 60.0, 90.0
                            else:
                                add_mask = True
                                min_c, max_c = 90.0, 100.0

                        if add_mask:
                            valid_candidates = [
                                r for r in all_external_candidates 
                                if min_c <= r[self.cloud_col_name] <= max_c
                            ]
                            
                            for cand_record in valid_candidates:
                                cand_id = cand_record["Scene"] 
                                if cand_id not in used_donors:
                                    temp_mask = self.load_mask_only(cand_record)
                                    if temp_mask is not None:
                                        borrowed_mask = temp_mask
                                        used_donors.add(cand_id)
                                        borrowed_mask = self.random_shift_mask(borrowed_mask)
                                        break 

                            if borrowed_mask is not None and random.random() < 0.30:
                                tube_active = True
                                tube_mask = borrowed_mask
                                tube_duration = random.randint(1, 3)

                    if borrowed_mask is not None:
                        for c in range(14):
                            X[i, c][borrowed_mask > 0] = 0.0
                        art_masks.append(torch.from_numpy(borrowed_mask).unsqueeze(0))
                    else:
                        art_masks.append(torch.zeros((1, self.config.IMG_SIZE, self.config.IMG_SIZE)))

                else:
                    tube_active = False 
                    art_masks.append(torch.zeros((1, self.config.IMG_SIZE, self.config.IMG_SIZE)))

            obs_mask = torch.stack(obs_masks)
            valid_mask = torch.stack(valid_masks)
            art_mask = torch.stack(art_masks)
            
            # ✅ 输入给模型的观测掩膜依然是二值的，并继续叠加人工洞。
            final_obs = obs_mask * (1.0 - art_mask)

            is_clean_flags = [
                1.0 if records[idx][self.cloud_col_name] < 5.0 else 0.0 
                for idx in indices
            ]

            return {
                "X": X,
                "y": y,
                "obs_mask": final_obs,
                "art_mask": art_mask,
                "valid_mask": valid_mask,
                "scl_mask": torch.stack(out_scl_masks), 
                "custom_cloud_mask": torch.stack(out_custom_clouds),
                "custom_shadow_mask": torch.stack(out_custom_shadows),
                "is_clean_flag": torch.tensor(is_clean_flags, dtype=torch.float32),
                "loc_id": loc_id,
                "target_idx": torch.tensor(target_pos, dtype=torch.long) 
            }

        except Exception as e:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))