import numpy as np
import pandas as pd
import torch


class GTNNWRWrapper:
    def __init__(self, x_columns, y_column="IKP"):
        self.x_columns = x_columns
        self.y_column = [y_column]
        self.model = None

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

        # pastikan kolom input ada
        for col in self.x_columns:
            if col not in data.columns:
                raise ValueError(f"Kolom '{col}' tidak ada di DataFrame")

        # ambil fitur & convert ke tensor
        x_input = torch.tensor(
            data[self.x_columns].values,
            dtype=torch.float32
        )

        # prediksi
        with torch.no_grad():
            y_pred = self.model(x_input)

        return np.array(y_pred).flatten()

    # ----------------------
    # Ambil koefisien layer terakhir
    # ----------------------
    def get_coefs(self):
        if self.model is None:
            raise ValueError("Model belum diload.")

        coefs = {}
        for name, param in self.model.named_parameters():
            if "weight" in name and ("fc" in name or "linear" in name):
                coefs[name] = param.detach().cpu().numpy()

        if not coefs:
            print("[WARNING] Tidak ditemukan layer linear dengan weight")
            return None

        # ambil layer terakhir
        last_layer_name = list(coefs.keys())[-1]
        coef_matrix = coefs[last_layer_name]

        # buat DataFrame
        df = pd.DataFrame(coef_matrix, columns=self.x_columns)
        df["Intercept"] = 0.0  # intercept tidak otomatis tersimpan
        return df
