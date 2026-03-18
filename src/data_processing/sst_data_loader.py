# src/data_processing/sst_data_loader.py

import os
import numpy as np
import xarray as xr


class SSTDataLoader:
    def __init__(self, folder_path, variable_name="analysed_sst"):
        """
        folder_path: path to regime folder (e.g., data/T1)
        variable_name: SST variable inside NetCDF
        """
        self.folder_path = folder_path
        self.variable_name = variable_name

    def load_all_files(self):
        """
        Load all .nc files from folder and return list of arrays
        """
        sst_data_list = []

        files = sorted([f for f in os.listdir(self.folder_path) if f.endswith(".nc")])

        if len(files) == 0:
            raise ValueError(f"No NetCDF files found in {self.folder_path}")

        for file in files:
            file_path = os.path.join(self.folder_path, file)

            print(f"Loading: {file_path}")

            ds = xr.open_dataset(file_path)

            if self.variable_name not in ds:
                raise ValueError(f"{self.variable_name} not found in {file}")

            sst = ds[self.variable_name].values

            # Remove time dimension if exists
            if len(sst.shape) == 3:
                sst = sst[0]

            sst_data_list.append(sst)

        return sst_data_list

    def preprocess(self, sst_data_list, downsample_factor=2):
        processed_data = []

        for sst in sst_data_list:
            # Clean data
            sst = np.nan_to_num(sst, nan=0.0, posinf=0.0, neginf=0.0)

            # Downsample
            sst = sst[::downsample_factor, ::downsample_factor]

            # Normalize
            sst_min = np.min(sst)
            sst_max = np.max(sst)

            if sst_max - sst_min == 0:
                sst = np.zeros_like(sst)
            else:
                sst = (sst - sst_min) / (sst_max - sst_min)

            # 🔥 EXTRACT FEATURES HERE (IMPORTANT)
            mean = np.mean(sst)
            std = np.std(sst)
            min_val = np.min(sst)
            max_val = np.max(sst)

            features = np.array([mean, std, min_val, max_val], dtype=np.float32)

            processed_data.append(features)

        return processed_data

    def load_and_process(self):
        """
        Full pipeline: load + preprocess
        """
        raw_data = self.load_all_files()
        processed_data = self.preprocess(raw_data)

        return processed_data


# ==========================
# 🔥 TEST ALL REGIMES HERE
# ==========================
if __name__ == "__main__":

    BASE_PATH = r"C:\Users\DELL\PycharmProjects\CAPSTONE\otec-continual-rl\data"

    regimes = {
        "T1": os.path.join(BASE_PATH, "T1"),
        "T2": os.path.join(BASE_PATH, "T2"),
        "T3": os.path.join(BASE_PATH, "T3"),
        "T4": os.path.join(BASE_PATH, "T4"),
    }

    all_data = {}

    for key, path in regimes.items():
        print(f"\n========== Loading {key} ==========")

        loader = SSTDataLoader(folder_path=path)
        data = loader.load_and_process()

        all_data[key] = data

        print(f"{key} -> Loaded {len(data)} samples")
        print(f"{key} -> Sample shape: {data[0].shape}")