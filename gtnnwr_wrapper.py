import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib


# --- PATCH bug Identity di PyTorch lama ---
if not hasattr(nn.Identity, "p"):
    nn.Identity.p = 0.0


class GTNNWRWrapper:
    def __init__(self, x_columns, y_column="IKP"):
        """
        Wrapper GTNNWR: bisa training baru atau load pretrained .pth

        Parameters
        ----------
        x_columns : list
            Nama kolom fitur prediktor
        y_column : str
            Kolom target (default "IKP")
        """
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.results = {}
        self.test_dataset = None
        self.coef_df = None

    # --------------------------
    # Training baru
    # --------------------------
    def fit(self, data: pd.DataFrame, max_epoch=2000, print_step=200):
        torch.autograd.set_detect_anomaly(False)

        # Preprocess
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))

        prov_col = "Provinsi" if "Provinsi" in data.columns else "Nama_Provinsi"
        required_cols = [prov_col, "Tahun", "Longitude", "Latitude"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Dataset harus punya kolom '{col}'")

        # Split
        train_data = data[data["Tahun"] <= 2022]
        val_data   = data[data["Tahun"] == 2023]
        test_data  = data[data["Tahun"] == 2024]

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

        # Model
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [512, 256, 64]],
            drop_out=0.5,
            optimizer="Adam",
            optimizer_params={
                "scheduler": "MultiStepLR",
                "scheduler_milestones": [500, 1000, 1500],
                "scheduler_gamma": 0.8,
            },
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # Train
        self.model.add_graph()
        self.model.run(max_epoch, print_step)

        # Simpan hasil
        self.results = self._evaluate(test_data)
        self._save_coefs(test_data, prov_col)
        return self.results

    # --------------------------
    # Load pretrained model (.pth berisi state_dict)
    # --------------------------
    def load_pretrained(self, data: pd.DataFrame, model_path: str):
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))
        prov_col = "Provinsi" if "Provinsi" in data.columns else "Nama_Provinsi"

        train_data = data[data["Tahun"] <= 2022]
        val_data   = data[data["Tahun"] == 2023]
        test_data  = data[data["Tahun"] == 2024]

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

        # Inisialisasi model kosong
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [512, 256, 64]],
            drop_out=0.5,
            optimizer="Adam",
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # Load state_dict
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Evaluasi manual
        self.results = self._evaluate(test_data)
        self._save_coefs(test_data, prov_col)
        return self.results

    # --------------------------
    # Evaluasi manual
    # --------------------------
    def _evaluate(self, test_data):
        result = {}
        try:
            y_true = test_data[self.y_column].values.flatten()
            y_pred = self.predict()
            if y_pred is not None:
                r2 = r2_score(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                result = {"R2": float(r2), "MSE": float(mse), "N_test": int(len(y_true))}
        except Exception as e:
            print(f"[WARNING] Evaluasi gagal: {e}")
        return result

    # --------------------------
    # Simpan koefisien
    # --------------------------
    def _save_coefs(self, data, prov_col):
        try:
            coef_df = self.model.reg_result()
            if coef_df is not None and isinstance(coef_df, (pd.DataFrame, np.ndarray, list)):
                coef_df = pd.DataFrame(coef_df)
                if prov_col in data.columns:
                    coef_df[prov_col] = data[prov_col].values[:len(coef_df)]
                if "Tahun" in data.columns:
                    coef_df["Tahun"] = data["Tahun"].values[:len(coef_df)]
                self.coef_df = coef_df
        except Exception as e:
            print(f"[WARNING] Tidak bisa ambil koefisien: {e}")
            self.coef_df = None

    # --------------------------
    # API publik
    # --------------------------
    def predict(self, dataset=None):
        if self.model is None:
            raise ValueError("Model belum diload/dilatih.")
        if dataset is None:
            dataset = self.test_dataset
        try:
            y_pred = self.model.predict(dataset)
            return np.array(y_pred).flatten()
        except Exception as e:
            print(f"[ERROR] Prediksi gagal: {e}")
            return None

    def get_results(self):
        return self.results

    def get_coefs(self):
        return self.coef_df
