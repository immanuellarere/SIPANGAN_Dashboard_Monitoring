import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib
import torch.nn as nn


# --- PATCH bug Identity.p (library issue) ---
if not hasattr(nn.Identity, "p"):
    nn.Identity.p = 0.0


class GTNNWRWrapper:
    def __init__(self, x_columns, y_column="IKP"):
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.results = {}
        self.test_dataset = None

    def fit(self, data: pd.DataFrame, max_epoch=2000, print_step=200):
        """
        Latih model GTNNWR dan simpan hasil evaluasi.
        """
        torch.autograd.set_detect_anomaly(False)

        # --- Preprocess ---
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

        # --- Split ---
        train_data = data[data["Tahun"] <= 2022]
        val_data   = data[data["Tahun"] == 2023]
        test_data  = data[data["Tahun"] == 2024]

        if train_data.empty or val_data.empty or test_data.empty:
            raise ValueError("Data tidak lengkap (butuh Tahun ≤2022, 2023, 2024).")

        # --- Dataset GTNNWR ---
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

        # --- Model ---
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [128, 64]],   # hidden layers
            drop_out=1e-8,      # hindari bug Identity.p
            optimizer="Adam",
            optimizer_params={
                "scheduler": "MultiStepLR",
                "scheduler_milestones": [500, 1000, 1500],
                "scheduler_gamma": 0.8,
            },
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # --- Train ---
        try:
            self.model.add_graph()
            self.model.run(max_epoch, print_step)
        except Exception as e:
            print(f"[ERROR] Training gagal: {e}")
            self.results = {"ERROR": str(e)}
            return self.results

        # --- Ambil hasil dari library ---
        result = {}
        try:
            result = self.model.result()
        except:
            try:
                result = self.model.reg_result
            except:
                result = {}

        # --- Fallback evaluasi manual ---
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
                result = {}

        # --- Konversi semua ke Python murni (bukan numpy) ---
        clean_result = {}
        if result is None:
            result = {}
        for k, v in result.items():
            if isinstance(v, (np.generic,)):
                clean_result[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean_result[k] = float(np.mean(v))
            elif isinstance(v, (list, tuple)):
                clean_result[k] = [float(x) for x in v]
            else:
                clean_result[k] = v

        if not clean_result:
            clean_result = {"INFO": "Evaluasi tidak tersedia"}

        self.results = clean_result
        return self.results

    def predict(self, dataset=None):
        """
        Prediksi data test (default) atau dataset lain.
        """
        if self.model is None:
            raise ValueError("Model belum dilatih. Jalankan .fit() dulu.")

        if dataset is None:
            dataset = self.test_dataset
        if dataset is None:
            raise ValueError("Dataset test tidak tersedia.")

        try:
            y_pred = self.model.predict(dataset)
            return np.array(y_pred).flatten().astype(float)  # pastikan 1D float
        except Exception as e:
            print(f"[ERROR] Prediksi gagal: {e}")
            return None
