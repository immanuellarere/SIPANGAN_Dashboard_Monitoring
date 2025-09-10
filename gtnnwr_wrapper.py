import numpy as np
import pandas as pd
import torch


class GTNNWRWrapper:
    def __init__(self, y_column="IKP", prov_col="Provinsi"):
        self.x_columns = None
        self.y_column = [y_column]
        self.prov_col = prov_col
        self.model = None

    # ----------------------
    # Tentukan x_columns otomatis dari DataFrame
    # ----------------------
    def _set_x_columns(self, data: pd.DataFrame):
        exclude_cols = [self.prov_col, "Tahun", *self.y_column,
                        "Longitude", "Latitude", "id"]
        self.x_columns = [c for c in data.columns if c not in exclude_cols]
        if len(self.x_columns) == 0:
            raise ValueError("❌ Tidak ada kolom fitur (x_columns) yang valid ditemukan!")

    # ----------------------
    # Load model TorchScript
    # ----------------------
    def load(self, model_path: str):
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        print(f"✅ Model TorchScript berhasil diload dari {model_path}")
        return True

    # ----------------------
    # Prediksi dari DataFrame
    # ----------------------
    def predict(self, data: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model belum diload.")

        if self.x_columns is None:
            self._set_x_columns(data)

        # convert ke tensor
        x_input = torch.tensor(
            data[self.x_columns].values,
            dtype=torch.float32
        )

        # reshape -> (batch, 1, features) jika perlu
        if x_input.dim() == 2:
            x_input = x_input.unsqueeze(1)

        with torch.no_grad():
            y_pred = self.model(x_input)

        # masukkan hasil prediksi ke df
        data = data.copy()
        data["IKP_Prediksi"] = np.array(y_pred).flatten()
        return data

    # ----------------------
    # Ambil koefisien layer terakhir + merge ke provinsi/tahun
    # ----------------------
    def get_coefs(self, data: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model belum diload.")

        if self.x_columns is None:
            self._set_x_columns(data)

        coefs = {}
        for name, param in self.model.named_parameters():
            if "weight" in name and ("fc" in name or "linear" in name):
                coefs[name] = param.detach().cpu().numpy()

        if not coefs:
            print("[WARNING] Tidak ditemukan layer linear dengan weight")
            return None

        # ambil bobot terakhir
        last_layer_name = list(coefs.keys())[-1]
        coef_matrix = coefs[last_layer_name]

        # buat dataframe koefisien
        coef_df = pd.DataFrame(coef_matrix, columns=self.x_columns)
        coef_df["Intercept"] = 0.0

        # duplikasi sesuai jumlah baris dataset
        coef_df = pd.concat([coef_df]*len(data), ignore_index=True)
        coef_df[self.prov_col] = data[self.prov_col].values
        coef_df["Tahun"] = data["Tahun"].values

        return coef_df
