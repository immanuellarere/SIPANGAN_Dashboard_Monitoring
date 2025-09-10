import torch
import numpy as np
import pandas as pd
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR as GTNNWR_lib


class GTNNWRWrapper:
    def __init__(self, x_columns, y_column="IKP"):
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.test_dataset = None
        self.results = {}

    def fit(self, data, max_epoch=2000, print_step=200):
        # Training biasa (kalau belum ada pretrained model)
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))

        # Split data
        train_data = data[data["Tahun"] <= 2022]
        val_data = data[data["Tahun"] == 2023]
        test_data = data[data["Tahun"] == 2024]

        train_dataset, val_dataset, test_dataset = init_dataset_split(
            train_data, val_data, test_data,
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

        self.model = GTNNWR_lib(
            train_dataset, val_dataset, test_dataset,
            [[3], [128, 64]],
            drop_out=0.5,
            optimizer="Adam",
            optimizer_params={},
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        self.model.add_graph()
        self.model.run(max_epoch, print_step)
        self.results = self.model.result()
        return self.results

    def load_pretrained(self, data, model_path):
        # Load model pretrained (pth)
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = np.arange(len(data))

        # Dataset split
        train_data = data[data["Tahun"] <= 2022]
        val_data = data[data["Tahun"] == 2023]
        test_data = data[data["Tahun"] == 2024]

        train_dataset, val_dataset, test_dataset = init_dataset_split(
            train_data, val_data, test_data,
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

        # Build model kosong
        self.model = GTNNWR_lib(
            train_dataset, val_dataset, test_dataset,
            [[3], [128, 64]],
            drop_out=0.5,
            optimizer="Adam",
            optimizer_params={},
            write_path="./gtnnwr_runs",
            model_name="GTNNWR_DSi"
        )

        # Load pretrained weights
        self.model.load_model(model_path)
        print(f"✅ Pretrained GTNNWR berhasil diload dari {model_path}")

    def predict(self, dataset=None):
        if dataset is None:
            dataset = self.test_dataset
        return np.array(self.model.predict(dataset)).flatten()

    def get_coefs(self):
        try:
            return self.model.reg_result()
        except:
            return None
