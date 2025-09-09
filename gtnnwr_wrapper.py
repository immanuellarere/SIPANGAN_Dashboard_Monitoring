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

        # Debug mode PyTorch
        torch.autograd.set_detect_anomaly(True)

    def fit(self, data: pd.DataFrame, max_epoch=500, print_step=100):
        # --- Preprocessing ---
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))  # ID unik

        # Validasi kolom wajib
        for col in ["Provinsi", "Tahun", "Longitude", "Latitude"]:
            if col not in data.columns:
                raise ValueError(f"Dataset harus memiliki kolom '{col}'")

        # Split dataset
        train_data = data[data["Tahun"] <= 2022]
        val_data   = data[data["Tahun"] == 2023]
        test_data  = data[data["Tahun"] == 2024]

        # Init dataset GTNNWR
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
            batch_size=64,   # kecil biar aman
            shuffle=False
        )

        # Hyperparameter optimizer
        optimizer_params = {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [100, 200, 300],
            "scheduler_gamma": 0.8,
        }

        # Inisialisasi model
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [64, 32]],   # lebih ringan
            drop_out=0.0,
            optimizer="Adadelta",
            optimizer_params=optimizer_params,
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # Training (try-except agar aman)
        try:
            self.model.add_graph()
            self.model.run(max_epoch, print_step)
            raw_result = self.model.result()
        except Exception as e:
            print(f"[ERROR] Training gagal: {e}")
            raw_result = {}

        self.results = raw_result if raw_result is not None else {}

        # Ambil reg_result juga
        try:
            self.results["reg_result"] = self.model.reg_result
        except Exception:
            self.results["reg_result"] = {}

        # --- Ambil koefisien ---
        beta = None
        if "beta" in self.results and self.results["beta"] is not None:
            beta = self.results["beta"]
        elif "reg_result" in self.results and "beta" in self.results["reg_result"]:
            beta = self.results["reg_result"]["beta"]

        if beta is not None:
            try:
                coefs = pd.DataFrame(beta, columns=self.x_columns)
                coefs["id"] = data["id"].values
                coefs["Tahun"] = data["Tahun"].values
                coefs["Provinsi"] = data["Provinsi"].values

                coefs_long = coefs.melt(
                    id_vars=["Provinsi", "Tahun", "id"],
                    value_vars=self.x_columns,
                    var_name="Variabel",
                    value_name="Koefisien"
                )
                self.results["coefs_long"] = coefs_long
            except Exception as e:
                print(f"[WARNING] Gagal memproses koefisien: {e}")
                self.results["coefs_long"] = pd.DataFrame()
        else:
            print("[INFO] Model tidak menghasilkan beta.")
            self.results["coefs_long"] = pd.DataFrame()

        return self.results
