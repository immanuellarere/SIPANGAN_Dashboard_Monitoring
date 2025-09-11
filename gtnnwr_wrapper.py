import numpy as np
import pandas as pd
import torch
from gnnwr.models import GTNNWR
from gnnwr.datasets import init_dataset_split


class GTNNWRWrapper:
    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None, prov_col="Provinsi"):
        # ============================
        # DEFINISI FITUR
        # ============================
        # Model lama hanya pakai 16 fitur total
        # → 15 fitur base + 1 fitur temporal (Tahun)
        self.base_x_columns = [
            'Skor_PPH', 'Luas_Panen', 'Produktivitas', 'Produksi',
            'Tanah_Longsor', 'Banjir', 'Kekeringan', 'Kebakaran', 'Cuaca',
            'OPD_Penggerek_Batang_Padi', 'OPD_Wereng_Batang_Coklat',
            'OPD_Tikus', 'OPD_Blas', 'OPD_Hwar_Daun', 'OPD_Tungro'
        ]
        self.extra_cols = ["Tahun"]   # hanya Tahun supaya total = 16 fitur

        # Kolom target (output tunggal)
        self.y_column = ["IKP"]
        self.prov_col = prov_col

        # ============================
        # INISIALISASI MODEL
        # ============================
        if train_dataset is not None:
            self.model = GTNNWR(
                train_dataset, val_dataset, test_dataset,
                # hidden layers harus sama seperti model lama
                [[3], [512, 256, 64]],
                drop_out=0.5,
                optimizer="Adadelta",
                optimizer_params={
                    "scheduler": "MultiStepLR",
                    "scheduler_milestones": [1000, 2000, 3000, 4000],
                    "scheduler_gamma": 0.8
                },
                write_path="./gtnnwr_runs",
                model_name="GTNNWR_DSi"
            )._model

            # PASTIKAN layer output = 16 (sesuai model lama)
            in_features = self.model.fc.full3.layer.in_features
            self.model.fc.full3.layer = torch.nn.Linear(in_features, 16)

        else:
            self.model = None

    # ============================
    # LOAD BOBOT .pth
    # ============================
    def load(self, model_path: str):
        if self.model is None:
            raise ValueError("❌ Model belum diinisialisasi dengan dataset.")

        state_dict = torch.load(model_path, map_location="cpu")

        # strict=True karena kita sudah samakan arsitektur → tidak boleh mismatch
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        print(f"✅ Model lama (.pth) berhasil diload dari {model_path}")
        return True

    # ============================
    # PREDIKSI
    # ============================
    def predict(self, data: pd.DataFrame):
        # gunakan hanya 16 fitur (15 + Tahun)
        x_columns = self.base_x_columns + self.extra_cols
        x_input = torch.tensor(data[x_columns].values, dtype=torch.float32)

        if x_input.dim() == 2:
            x_input = x_input.unsqueeze(1)

        with torch.no_grad():
            y_pred = self.model(x_input)

        data = data.copy()
        data["IKP_Prediksi"] = np.array(y_pred).flatten()
        return data

    # ============================
    # AMBIL KOEFISIEN
    # ============================
    def get_coefs(self, data: pd.DataFrame):
        coefs = {}
        for name, param in self.model.named_parameters():
            if "weight" in name and ("fc" in name or "linear" in name):
                coefs[name] = param.detach().cpu().numpy()

        if not coefs:
            return None

        # ambil layer terakhir
        last_layer_name = list(coefs.keys())[-1]
        coef_matrix = coefs[last_layer_name]

        # pastikan kolom sesuai 16 fitur lama
        x_columns = self.base_x_columns + self.extra_cols
        coef_cols = (
            x_columns
            if coef_matrix.shape[1] == len(x_columns)
            else [f"feat_{i}" for i in range(coef_matrix.shape[1])]
        )

        coef_df = pd.DataFrame(coef_matrix, columns=coef_cols)
        coef_df["Intercept"] = 0.0

        # replikasi sesuai jumlah baris data
        coef_df = pd.concat([coef_df] * len(data), ignore_index=True)
        coef_df[self.prov_col] = data[self.prov_col].values
        coef_df["Tahun"] = data["Tahun"].values
        return coef_df
