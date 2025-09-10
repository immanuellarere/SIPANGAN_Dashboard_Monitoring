import numpy as np
import pandas as pd
import torch
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib


class GTNNWRWrapper:
    def __init__(self, x_columns, y_column="IKP"):
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.test_dataset = None
        self.coef_df = None

    # ----------------------
    # Training dari nol
    # ----------------------
    def fit(self, data: pd.DataFrame, max_epoch=2000, print_step=200):
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))

        prov_col = "Provinsi" if "Provinsi" in data.columns else "Nama_Provinsi"
        required_cols = [prov_col, "Tahun", "Longitude", "Latitude"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Dataset harus memiliki kolom '{col}'")

        # Split dataset
        train_data = data[data["Tahun"] <= 2022]
        val_data = data[data["Tahun"] == 2023]
        test_data = data[data["Tahun"] == 2024]

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

        # Build model
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [512, 256, 64]],
            drop_out=0.5,
            optimizer="Adadelta",
            optimizer_params={
                "scheduler": "MultiStepLR",
                "scheduler_milestones": [500, 1000, 1500],
                "scheduler_gamma": 0.8,
            },
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        self.model.add_graph()
        self.model.run(max_epoch, print_step)

        return self.result()

    # ----------------------
    # Save model (state_dict)
    # ----------------------
    def save_model(self, path="./GTNNWR_DSi_model.pth"):
        if self.model is None:
            raise ValueError("Model belum ada. Jalankan .fit() dulu.")
        torch.save(self.model._model.state_dict(), path)
        return path

    # ----------------------
    # Load model pretrained
    # ----------------------
    def load_pretrained(self, data: pd.DataFrame, model_path: str):
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))

        prov_col = "Provinsi" if "Provinsi" in data.columns else "Nama_Provinsi"

        train_data = data[data["Tahun"] <= 2022]
        val_data = data[data["Tahun"] == 2023]
        test_data = data[data["Tahun"] == 2024]

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

        # Build ulang arsitektur
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [512, 256, 64]],
            drop_out=0.5,
            optimizer="Adadelta",
            optimizer_params={
                "scheduler": "MultiStepLR",
                "scheduler_milestones": [500, 1000, 1500],
                "scheduler_gamma": 0.8,
            },
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )
        self.model.add_graph()

        # Load weight
        state_dict = torch.load(model_path, map_location="cpu")
        self.model._model.load_state_dict(state_dict)

        return True

    # ----------------------
    # Prediksi
    # ----------------------
    def predict(self, dataset=None):
        if self.model is None:
            raise ValueError("Model belum diload.")
        if dataset is None:
            dataset = self.test_dataset
        if dataset is None:
            raise ValueError("Dataset test tidak ada.")
        y_pred = self.model.predict(dataset)
        return np.array(y_pred).flatten()

    # ----------------------
    # Ambil koefisien
    # ----------------------
    def get_coefs(self):
        if self.model is None:
            return None
        try:
            coef = self.model.getCoefs()
            if isinstance(coef, (np.ndarray, list)):
                coef = pd.DataFrame(coef)
            coef.columns = [*self.x_columns, "Intercept"]
            coef["Tahun"] = self.test_dataset.temporal
            coef["Provinsi"] = self.test_dataset.ids
            self.coef_df = coef
            return coef
        except Exception as e:
            print(f"[WARNING] Tidak bisa ambil koefisien: {e}")
            return None

    # ----------------------
    # Hasil evaluasi
    # ----------------------
    def result(self):
        if self.model is None:
            return {}
        try:
            return self.model.result()
        except:
            return {}
