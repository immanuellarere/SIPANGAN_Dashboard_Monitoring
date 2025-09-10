import torch
from gnnwr.models import GTNNWR as GTNNWR_lib
from gnnwr.datasets import init_dataset_split

class GTNNWRWrapper:
    def __init__(self, x_columns, y_column="IKP"):
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None
        self.test_dataset = None

    def load_pretrained(self, data, model_path, hidden_layers=[512,256,64]):
        # ------------------------
        # Preprocessing
        # ------------------------
        data = data.rename(columns=lambda x: x.strip().replace(" ", "_"))
        data["id"] = range(len(data))

        # Split dataset
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

        # Bangun ulang arsitektur
        self.model = GTNNWR_lib(
            train_dataset,
            val_dataset,
            test_dataset,
            [[3], hidden_layers],
            drop_out=0.5,
            optimizer="Adadelta"
        )
        self.model.add_graph()

        # Load state_dict ke internal _model
        state_dict = torch.load(model_path, map_location="cpu")
        self.model._model.load_state_dict(state_dict)

        print("✅ Model pretrained berhasil dimuat")
        return self.model

    def predict(self, dataset=None):
        if dataset is None:
            dataset = self.test_dataset
        return self.model.predict(dataset)

    def get_coefs(self):
        try:
            coef_df = self.model.reg_result()
            return coef_df
        except:
            return None
