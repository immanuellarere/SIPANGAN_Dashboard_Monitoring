import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib
import torch.nn as nn


# --------------------------
# PATCH bug Identity.p
# --------------------------
if not hasattr(nn.Identity, "p"):
    nn.Identity.p = 0.0


class GTNNWRWrapper:
    """
    Wrapper untuk GTNNWR supaya lebih stabil di Streamlit.
    - Semua hasil evaluasi dikonversi jadi Python native (float/int).
    - Prediksi dijamin numpy array 1D.
    - Ada fallback evaluasi manual kalau hasil model kosong.
    """

    def __init__(self, x_columns, y_column="IKP"):
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.results = {}
        self.test_dataset = None

    def fit(self, data: pd.DataFrame, max_epoch=2000, print_step=200):
        # --------------------------
        # Preprocessing
        # --------------------------
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))

        if "Provinsi" in data.columns:
            prov_col = "Provinsi"
        elif "Nama_Provinsi" in data.columns:
            prov_col = "Nama_Provinsi"
        else:
            raise ValueError("Dataset harus punya kolom 'Provinsi' atau 'Nama_Provinsi'")

        required_cols = [prov_col, "Tahun", "Longitude", "Latitude"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Dataset harus punya kolom '{col}'")

        # --------------------------
        # Split train/val/test
        # --------------------------
        train_data = data[data["Tahun"] <= 2022]
        val_data   = data[data["Tahun"] == 2023]
        test_data  = data[data["Tahun"] == 2024]

        if train_data.empty or val_data.empty or test_data.empty:
            raise ValueError("Data tidak lengkap (butuh Tahun ≤2022, 2023, 2024).")

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
        self.test_dataset = test_dataset

        # --------------------------
        # Inisialisasi model GTNNWR
        # --------------------------
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [128, 64]],   # hidden layers
            drop_out=1e-8,      # fix Identity bug
            optimizer="Adam",
            optimizer_params={
                "scheduler": "MultiStepLR",
                "scheduler_milestones": [500, 1000, 1500],
                "scheduler_gamma": 0.8,
            },
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # --------------------------
        # Training
        # --------------------------
        try:
            self.model.add_graph()
            self.model.run(max_epoch, print_step)
        except Exception as e:
            print(f"[ERROR] Training gagal: {e}")
            self.results = {}
            return self.results

        # --------------------------
        # Ambil hasil evaluasi
        # --------------------------
        result = {}
        try:
            result = self.model.result()
        except Exception:
            try:
                result = self.model.reg_result
            except Exception:
                result = {}

        # --------------------------
        # Fallback evaluasi manual
        # --------------------------
        if not result:
            try:
                y_true = test_data[self.y_column].values.flatten()
                y_pred = self.predict()
                if y_pred is not None:
                    r2 = r2_score(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    result = {
                        "R2": float(r2),
                        "MSE": float(mse),
                        "N_test": int(len(y_true))
                    }
            except Exception as e:
                print(f"[ERROR] Evaluasi manual gagal: {e}")

        # --------------------------
        # Konversi hasil ke Python murni
        # --------------------------
        clean_result = {}
        for k, v in result.items():
            if isinstance(v, (np.generic, np.ndarray)):
                clean_result[k] = float(np.mean(v))
            elif isinstance(v, (list, tuple)):
                clean_result[k] = [float(x) for x in v]
            else:
                clean_result[k] = v

        self.results = clean_result
        return self.results

    def predict(self, dataset=None):
        """
        Prediksi menggunakan model yang sudah dilatih.
        Output selalu numpy array 1D.
        """
        if self.model is None:
            raise ValueError("Model belum dilatih. Jalankan .fit() dulu.")

        if dataset is None:
            dataset = self.test_dataset
        if dataset is None:
            raise ValueError("Dataset test tidak tersedia.")

        try:
            y_pred = self.model.predict(dataset)
            return np.array(y_pred).flatten()
        except Exception as e:
            print(f"[ERROR] Prediksi gagal: {e}")
            return None
