import numpy as np
import pandas as pd
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib


class GTNNWRWrapper:
    """
    Wrapper untuk GTNNWR.
    Catatan: library gnnwr versi ini tidak mengembalikan koefisien (beta),
    jadi hanya hasil evaluasi model (R2, Loss, AIC, dsb) yang tersedia.
    """

    def __init__(self, x_columns, y_column="IKP"):
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.results = {}

    def fit(self, data, max_epoch=2000, print_step=200):
        # Preprocessing
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))

        # Split dataset
        train_data = data[data["Tahun"] <= 2022]
        val_data   = data[data["Tahun"] == 2023]
        test_data  = data[data["Tahun"] == 2024]

        # Init dataset
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
            batch_size=128,
            shuffle=False
        )

        optimizer_params = {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [200, 400, 600],
            "scheduler_gamma": 0.8,
        }

        # Init model
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], [128, 64]],
            drop_out=0.0,
            optimizer="Adadelta",
            optimizer_params=optimizer_params,
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # Training
        self.model.add_graph()
        self.model.run(max_epoch, print_step)

        # Ambil hasil evaluasi
        raw_result = self.model.result()
        self.results = raw_result if raw_result else {}

        return self.results
