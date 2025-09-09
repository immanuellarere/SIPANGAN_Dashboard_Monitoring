import numpy as np
import pandas as pd

from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib


class GTNNWRWrapper:
    """
    Wrapper untuk model GTNNWR.
    - Preprocessing dataset
    - Split train/val/test berdasarkan Tahun
    - Training GTNNWR
    - Ambil hasil model termasuk koefisien variabel (beta)
    """

    def __init__(self, x_columns, y_column="IKP"):
        """
        Parameters
        ----------
        x_columns : list
            Kolom prediktor (misalnya variabel pertanian, bencana, OPD).
        y_column : str
            Kolom target (default = IKP).
        """
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.results = None

    def fit(self, data: pd.DataFrame):
        # --- Preprocessing ---
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))  # ID unik per baris

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
            spatial_column=["Longitude", "Latitude"],  # koordinat spasial
            temp_column=["Tahun"],                     # dimensi temporal
            id_column=["id"],                          # ID unik
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

        # Inisialisasi model GTNNWR
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
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

        # Hasil utama
        self.results = self.model.result()
        self.results["reg_result"] = self.model.reg_result

        # --- Ambil koefisien variabel ---
        if "beta" in self.results and self.results["beta"] is not None:
            try:
                coefs = pd.DataFrame(self.results["beta"], columns=self.x_columns)
                coefs["id"] = data["id"].values
                coefs["Tahun"] = data["Tahun"].values
                coefs["Provinsi"] = data["Provinsi"].values

                # Bentuk long format
                coefs_long = coefs.melt(
                    id_vars=["Provinsi", "Tahun", "id"],
                    value_vars=self.x_columns,
                    var_name="Variabel",
                    value_name="Koefisien"
                )

                self.results["coefs_long"] = coefs_long
            except Exception as e:
                print(f"Gagal memproses koefisien: {e}")
                self.results["coefs_long"] = pd.DataFrame()
        else:
            self.results["coefs_long"] = pd.DataFrame()

        return self.results
