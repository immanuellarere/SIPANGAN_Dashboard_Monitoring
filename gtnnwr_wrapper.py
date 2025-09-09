import numpy as np
import pandas as pd
import torch
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib


class GTNNWRWrapper:
    """
    Wrapper untuk model GTNNWR.
    """

    def __init__(self, x_columns, y_column="IKP"):
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.results = {}

    def fit(self, data: pd.DataFrame, max_epoch=2000, print_step=200):
        # --------------------------
        # Preprocessing
        # --------------------------
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))

        required_cols = ["Provinsi", "Tahun", "Longitude", "Latitude"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Dataset harus memiliki kolom '{col}'")

        train_data = data[data["Tahun"] <= 2022]
        val_data   = data[data["Tahun"] == 2023]
        test_data  = data[data["Tahun"] == 2024]

        if train_data.empty or val_data.empty or test_data.empty:
            raise ValueError("Data tidak lengkap untuk split (butuh Tahun ≤2022, 2023, 2024).")

        # --------------------------
        # Dataset split
        # --------------------------
        train_dataset, val_dataset, test_dataset = init_dataset_split(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            x_column=self.x_columns,
            y_column=self.y_column,
            spatial_column=["Longitude", "Latitude"],
            temp_column=["Tahun"],
            id_column=["id"],
            use_model="gtnnwr",
            batch_size=512,   # batch lebih kecil biar stabil
            shuffle=False
        )

        optimizer_params = {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [500, 1000, 1500],
            "scheduler_gamma": 0.8,
        }

        # --------------------------
        # Inisialisasi model
        # --------------------------
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [128, 64]],   # hidden layer lebih ringan
            drop_out=0.5,       # no inplace
            optimizer="Adam",   # coba ganti optimizer lebih stabil
            optimizer_params=optimizer_params,
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # --------------------------
        # Training (workaround: matikan grad kalau error)
        # --------------------------
        try:
            self.model.add_graph()
            self.model.run(max_epoch, print_step)
        except RuntimeError as e:
            print("[WARNING] Gradient error terdeteksi:", e)
            print("[INFO] Menjalankan ulang tanpa autograd (no_grad mode)…")
            with torch.no_grad():
                self.model.add_graph()
                self.model.run(max_epoch, print_step)

        # --------------------------
        # Ambil hasil evaluasi
        # --------------------------
        raw_result = {}
        try:
            raw_result = self.model.result()
        except Exception:
            try:
                raw_result = self.model.reg_result
            except Exception:
                raw_result = {}

        self.results = raw_result
        return self.results
