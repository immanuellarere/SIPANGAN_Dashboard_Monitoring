import numpy as np
import pandas as pd
import torch


class GTNNWRWrapper:
    def __init__(self, prov_col="Provinsi"):
        # fitur utama (dari training)
        self.base_x_columns = [
            'Skor_PPH', 'Luas_Panen', 'Produktivitas', 'Produksi',
            'Tanah_Longsor', 'Banjir', 'Kekeringan', 'Kebakaran', 'Cuaca',
            'OPD_Penggerek_Batang_Padi', 'OPD_Wereng_Batang_Coklat',
            'OPD_Tikus', 'OPD_Blas', 'OPD_Hwar_Daun', 'OPD_Tungro'
        ]
        # fitur tambahan spatio-temporal
        self.extra_cols = ["Longitude", "Latitude", "Tahun"]

        # target
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
            raise ValueError("❌ Model belum diload.")

        # semua kolom input (urutan penting!)
        x_columns = self.base_x_columns + self.extra_cols

        # cek semua kolom ada
        for col in x_columns:
            if col not in data.columns:
                raise ValueError(f"❌ Kolom fitur '{col}' tidak ada di dataset!")

        # ambil input → convert ke tensor
        x_input = torch.tensor(
            data[x_columns].values,
            dtype=torch.float32
        )

        # pastikan dimensi sesuai (batch, 1, features)
        if x_input.dim() == 2:
            x_input = x_input.unsqueeze(1)

        # prediksi
        with torch.no_grad():
            y_pred = self.model(x_input)

        # hasil ke dataframe
        data = data.copy()
        data["IKP_Prediksi"] = np.array(y_pred).flatten()
        return data

    # ----------------------
    # Ambil koefisien (layer terakhir linear)
    # ----------------------
    def get_coefs(self, data: pd.DataFrame):
        if self.model is None:
            raise ValueError("❌ Model belum diload.")

        # kumpulkan semua weight linear/fc
        coefs = {}
        for name, param in self.model.named_parameters():
            if "weight" in name and ("fc" in name or "linear" in name):
                coefs[name] = param.detach().cpu().numpy()

        if not coefs:
            print("[⚠️ WARNING] Tidak ada layer linear dengan weight.")
            return None

        # ambil layer terakhir
        last_layer_name = list(coefs.keys())[-1]
        coef_matrix = coefs[last_layer_name]

        # fitur input asli
        x_columns = self.base_x_columns + self.extra_cols

        # cek dimensi sesuai
        if coef_matrix.shape[1] != len(x_columns):
            print(f"[⚠️ WARNING] Jumlah kolom fitur ({len(x_columns)}) ≠ bobot terakhir ({coef_matrix.shape[1]}).")
            # fallback: kasih nama generik
            coef_cols = [f"feat_{i}" for i in range(coef_matrix.shape[1])]
        else:
            coef_cols = x_columns

        # dataframe koefisien
        coef_df = pd.DataFrame(coef_matrix, columns=coef_cols)
        coef_df["Intercept"] = 0.0

        # gandakan supaya align dengan provinsi+tahun
        coef_df = pd.concat([coef_df] * len(data), ignore_index=True)
        coef_df[self.prov_col] = data[self.prov_col].values
        coef_df["Tahun"] = data["Tahun"].values

        return coef_df
