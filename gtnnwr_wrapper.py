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
        self.results = {}

    def fit(self, data: pd.DataFrame, max_epoch=2000, print_step=200):
        # --- Preprocessing ---
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))  # ID unik per baris

        # Validasi kolom wajib
        for col in ["Provinsi", "Tahun", "Longitude", "Latitude"]:
            if col not in data.columns:
                raise ValueError(f"Dataset harus memiliki kolom '{col}'")

        # Split dataset (asumsi ada data 2022–2024)
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
            batch_size=256,    # kecilkan batch_size biar stabil
            shuffle=False
        )

        # Hyperparameter optimizer
        optimizer_params = {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [200, 400, 600, 800],
            "scheduler_gamma": 0.8,
        }

        # Inisialisasi model GTNNWR
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [128, 64]],   # hidden layers lebih ringan
            drop_out=0.0,       # matikan dropout biar tracing aman
            optimizer="Adadelta",
            optimizer_params=optimizer_params,
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # Training
        self.model.add_graph()
        self.model.run(max_epoch, print_step)

        # Ambil hasil
        raw_result = self.model.result()
        if raw_result is None:
            self.results = {}
        else:
            self.results = raw_result

        # Tambahkan reg_result kalau ada
        try:
            self.results["reg_result"] = self.model.reg_result
        except Exception:
            self.results["reg_result"] = {}

        # --- Ambil koefisien variabel ---
        if "beta" in self.results and self.results["beta"] is not None:
            try:
                coefs = pd.DataFrame(self.results["beta"], columns=self.x_columns)
                coefs["id"] = data["id"].values
                coefs["Tahun"] = data["Tahun"].values
                coefs["Provinsi"] = data["Provinsi"].values

                # Long format
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
            # fallback kalau beta kosong
            self.results["coefs_long"] = pd.DataFrame()

        return self.results
