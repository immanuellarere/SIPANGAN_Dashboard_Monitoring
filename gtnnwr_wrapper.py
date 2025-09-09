import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error

from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib


class GTNNWRWrapper:
    """
    Wrapper untuk model GTNNWR.
    Catatan:
    - Training tetap pakai autograd (gradien tidak dimatikan).
    - Jika hasil dari library kosong, wrapper hitung evaluasi manual (R², MSE).
    """

    def __init__(self, x_columns, y_column="IKP"):
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.results = {}

    def fit(self, data: pd.DataFrame, max_epoch=2000, print_step=200):
        """
        Latih model GTNNWR.
        """

        # --------------------------
        # Debug PyTorch: aktifkan anomaly detection
        # --------------------------
        torch.autograd.set_detect_anomaly(True)

        # --------------------------
        # Preprocessing
        # --------------------------
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))

        required_cols = ["Provinsi", "Tahun", "Longitude", "Latitude"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Dataset harus memiliki kolom '{col}'")

        # --------------------------
        # Split dataset (train ≤2022, val=2023, test=2024)
        # --------------------------
        train_data = data[data["Tahun"] <= 2022]
        val_data   = data[data["Tahun"] == 2023]
        test_data  = data[data["Tahun"] == 2024]

        if train_data.empty or val_data.empty or test_data.empty:
            raise ValueError("Data tidak lengkap untuk split (butuh Tahun ≤2022, 2023, 2024).")

        # --------------------------
        # Init dataset GTNNWR
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
            batch_size=512,
            shuffle=False
        )

        # --------------------------
        # Hyperparameter optimizer
        # --------------------------
        optimizer_params = {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [500, 1000, 1500],
            "scheduler_gamma": 0.8,
        }

        # --------------------------
        # Inisialisasi model GTNNWR
        # --------------------------
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [128, 64]],   # hidden layers
            drop_out=0.5,       # jangan inplace
            optimizer="Adam",   # lebih stabil daripada Adadelta
            optimizer_params=optimizer_params,
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # --------------------------
        # Training (dengan autograd)
        # --------------------------
        self.model.add_graph()
        self.model.run(max_epoch, print_step)

        # --------------------------
        # Ambil hasil evaluasi
        # --------------------------
        raw_result = {}
        try:
            raw_result = self.model.result()
        except Exception as e:
            print(f"[WARNING] result() gagal: {e}")

        if not raw_result:
            try:
                raw_result = self.model.reg_result
                print("[INFO] Hasil diambil dari reg_result")
            except Exception:
                raw_result = {}

        # --------------------------
        # Evaluasi manual (fallback)
        # --------------------------
        if not raw_result:
            try:
                y_true = test_data[self.y_column].values
                y_pred = self.model.predict(test_dataset)

                r2 = r2_score(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)

                raw_result = {
                    "R2": float(r2),
                    "MSE": float(mse),
                    "N_test": int(len(y_true))
                }
                print("[INFO] Hasil evaluasi dihitung manual (R², MSE)")
            except Exception as e:
                print(f"[ERROR] Evaluasi manual gagal: {e}")
                raw_result = {}

        self.results = raw_result
        return self.results
