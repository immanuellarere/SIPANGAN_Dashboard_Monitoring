# gtnnwr.py
import numpy as np
import pandas as pd

# Import modul GTNNWR dari package gnnwr
# ⚠️ Pastikan kamu sudah install / punya library gnnwr
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib


class GTNNWRWrapper:
    """
    Wrapper untuk model GTNNWR.
    Meng-handle preprocessing, pembagian data train/val/test,
    training, dan pengambilan hasil (termasuk koefisien variabel).
    """

    def __init__(self, x_columns, y_column="IKP"):
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.results = None

    def fit(self, data: pd.DataFrame):
        """
        Training model GTNNWR dengan dataset yang diberikan.

        Parameters
        ----------
        data : pd.DataFrame
            Data dengan kolom:
              - Tahun
              - Longitude, Latitude
              - target (default = IKP)
              - prediktor sesuai x_columns
        """
        # --- Preprocessing ---
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))

        # Split data berdasarkan Tahun
        train_data = data[data["Tahun"] <= 2022]
        val_data   = data[data["Tahun"] == 2023]
        test_data  = data[data["Tahun"] == 2024]

        # Init dataset untuk GTNNWR
        train_set, val_set, test_set = init_dataset_split(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            x_column=self.x_columns,
            y_column=self.y_column,
            spatial_column=['Longitude', 'Latitude'],
            temp_column=['Tahun'],
            id_column=['id'],
            use_model="gtnnwr",
            batch_size=1024,
            shuffle=False
        )

        # Hyperparameter optimizer
        optimizer_params = {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [1000, 2000, 3000, 4000],
            "scheduler_gamma": 0.8,
        }

        # --- Inisialisasi model ---
        self.model = GTNNWR_lib(
            train_set, val_set, test_set,
            [[3], [512, 256, 64]],   # hidden layers
            drop_out=0.5,
            optimizer="Adadelta",
            optimizer_params=optimizer_params,
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # Training
        self.model.add_graph()
        self.model.run(15000, 1000)

        # Ambil hasil dari model
        self.results = self.model.result()

        # --- Ambil koefisien variabel (jika ada di results) ---
        if "beta" in self.results:
            coefs = pd.DataFrame(self.results["beta"], columns=self.x_columns)
            coefs["id"] = data["id"].values
            coefs["Tahun"] = data["Tahun"].values
            coefs["Provinsi"] = data["Provinsi"].values if "Provinsi" in data.columns else data.index
            self.results["coefs"] = coefs

        return self.results
