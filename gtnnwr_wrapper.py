# gtnnwr_wrapper.py
import numpy as np
import pandas as pd

from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib


class GTNNWRWrapper:
    """
    Wrapper untuk model GTNNWR.
    - Preprocessing dataset
    - Split train/val/test berdasarkan waktu
    - Training GTNNWR
    - Mengambil hasil model termasuk koefisien variabel (beta)
    """

    def __init__(self, x_columns, y_column="Pred_IKP"):
        """
        Parameters
        ----------
        x_columns : list
            Kolom prediktor (misalnya semua kolom coef_*).
        y_column : str
            Target kolom (default = Pred_IKP).
        """
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.results = None

    def fit(self, data: pd.DataFrame):
        # --- Preprocessing ---
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["row_id"] = np.arange(len(data))  # ID unik per baris

        # Split data berdasarkan waktu
        train_data = data[data["waktu"] <= 2022]
        val_data   = data[data["waktu"] == 2023]
        test_data  = data[data["waktu"] == 2024]

        # Init dataset GTNNWR
        train_set, val_set, test_set = init_dataset_split(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            x_column=self.x_columns,
            y_column=self.y_column,
            spatial_column=[],           # dataset kamu tidak punya Longitude/Latitude
            temp_column=["waktu"],       # gunakan waktu sebagai dimensi temporal
            id_column=["row_id"],        # gunakan row_id sebagai ID unik
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

        # Inisialisasi model
        self.model = GTNNWR_lib(
            train_set, val_set, test_set,
            [[3], [512, 256, 64]],  # hidden layers
            drop_out=0.5,
            optimizer="Adadelta",
            optimizer_params=optimizer_params,
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # Training
        self.model.add_graph()
        self.model.run(15000, 1000)

        # Ambil hasil
        self.results = self.model.result()

        # --- Ambil koefisien variabel ---
        if "beta" in self.results:
            coefs = pd.DataFrame(self.results["beta"], columns=self.x_columns)
            coefs["row_id"] = data["row_id"].values
            coefs["waktu"] = data["waktu"].values
            coefs["ID"] = data["ID"].values if "ID" in data.columns else data.index
            self.results["coefs"] = coefs

        return self.results
