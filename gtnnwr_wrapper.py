import numpy as np
import pandas as pd
import torch


class GTNNWRWrapper:
    def __init__(self, prov_col="Provinsi"):
        self.x_columns = [
            'Skor_PPH', 'Luas_Panen', 'Produktivitas', 'Produksi',
            'Tanah_Longsor', 'Banjir', 'Kekeringan', 'Kebakaran', 'Cuaca',
            'OPD_Penggerek_Batang_Padi', 'OPD_Wereng_Batang_Coklat',
            'OPD_Tikus', 'OPD_Blas', 'OPD_Hwar_Daun', 'OPD_Tungro'
        ]
        self.y_column = ["IKP"]
        self.prov_col = prov_col
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

        # cek semua fitur ada di data
        for col in self.x_columns:
            if col not in data.columns:
                raise ValueError(f"❌ Kolom fitur '{col}' tidak ada di dataset!")

        # ambil input
        x_input = torch.tensor(
            data[self.x_columns].values,
            dtype=torch.float32
        )

        # pastikan dimensi cocok: (batch, 1, features)
        if x_input.dim() == 2:
            x_input = x_input.unsqueeze(1)

        with torch.no_grad():
            y_pred = self.model(x_input)

        data = data.copy()
        data["IKP_Prediksi"] = np.array(y_pred).flatten()
        return data

    # ----------------------
    # Ambil koefisien layer terakhir
    # ----------------------
    def get_coefs(self, data: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model belum diload.")

        coefs = {}
        for name, param in self.model.named_parameters():
            if "weight" in name and ("fc" in name or "linear" in name):
                coefs[name] = param.detach().cpu().numpy()

        if not coefs:
            print("[WARNING] Tidak ditemukan layer linear dengan weight")
            return None

        last_layer_name = list(coefs.keys())[-1]
        coef_matrix = coefs[last_layer_name]

        coef_df = pd.DataFrame(coef_matrix, columns=self.x_columns)
        coef_df["Intercept"] = 0.0

        coef_df = pd.concat([coef_df] * len(data), ignore_index=True)
        coef_df[self.prov_col] = data[self.prov_col].values
        coef_df["Tahun"] = data["Tahun"].values

        return coef_df
