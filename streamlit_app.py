import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import branca.colormap as cm
import torch

from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR

# --------------------------
# Konfigurasi Halaman
# --------------------------
st.set_page_config(page_title="SIPANGAN Dashboard Monitoring", layout="wide")
st.title("📊 SIPANGAN Dashboard Monitoring")
st.caption("Inference GTNNWR (.pt) dengan pipeline sama seperti training")

DATA_PATH = "datasec.xlsx"
MODEL_PATH = "gtnnwr_model.pt"   # file .pt hasil training

# --------------------------
# Load Dataset
# --------------------------
try:
    df = pd.read_excel(DATA_PATH, engine="openpyxl") if DATA_PATH.endswith("xlsx") else pd.read_csv(DATA_PATH)
    df = df.fillna(0)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    df["id"] = range(len(df))
except Exception as e:
    st.error(f"❌ Gagal membaca dataset: {e}")
    st.stop()

# Tentukan kolom provinsi
prov_col = "Provinsi" if "Provinsi" in df.columns else ("Nama_Provinsi" if "Nama_Provinsi" in df.columns else None)
if prov_col is None:
    st.error("❌ Dataset harus punya kolom 'Provinsi' atau 'Nama_Provinsi'")
    st.stop()

st.write("---")
st.subheader("🔍 Data Preview")
st.dataframe(df.head())

# --------------------------
# Definisi fitur sesuai training
# --------------------------
x_columns = [
    'Skor_PPH','Luas_Panen','Produktivitas','Produksi',
    'Tanah_Longsor','Banjir','Kekeringan','Kebakaran','Cuaca',
    'OPD_Penggerek_Batang_Padi','OPD_Wereng_Batang_Coklat',
    'OPD_Tikus','OPD_Blas','OPD_Hwar_Daun','OPD_Tungro'
]

# Split dataset seperti training
train_data = df[df["Tahun"] <= 2022].copy()
val_data   = df[df["Tahun"] == 2023].copy()
test_data  = df[df["Tahun"] == 2024].copy()

train_ds, val_ds, test_ds = init_dataset_split(
    train_data=train_data,
    val_data=val_data if len(val_data) else train_data,
    test_data=test_data if len(test_data) else train_data,
    x_column=x_columns,
    y_column=["IKP"],
    spatial_column=["Longitude","Latitude"],
    temp_column=["Tahun"],
    id_column=["id"],
    use_model="gtnnwr",
    batch_size=1024,
    shuffle=False
)

# --------------------------
# Load Model
# --------------------------
@st.cache_resource
def load_model():
    optim_params = {
        "scheduler":"MultiStepLR",
        "scheduler_milestones":[1000, 2000, 3000, 4000],
        "scheduler_gamma":0.8,
    }
    wrapper = GTNNWR(
        train_ds, val_ds, test_ds,
        [[3],[512,256,64]],  # arsitektur hidden layer
        drop_out=0.5,
        optimizer="Adadelta",
        optimizer_params=optim_params,
        write_path="./gtnnwr_runs",
        model_name="GTNNWR_DSi"
    )
    wrapper.add_graph()

    pretrained = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    wrapper._model.load_state_dict(pretrained.state_dict(), strict=False)
    wrapper._model.eval()
    return wrapper._model

try:
    model = load_model()
    st.success("✅ Model berhasil diload dari .pt (arsitektur + bobot)")
except Exception as e:
    st.error(f"❌ Gagal load model .pt: {e}")
    st.stop()

# --------------------------
# Bentuk Input Multi-Branch
# --------------------------
def dataset_to_branches(ds):
    """Ambil input (x_spatial, x_features) persis dari dataset"""
    xs = [ds[i][0] for i in range(len(ds))]   # setiap item: (spatial, features)
    x_spatial  = torch.stack([item[0].squeeze(0) for item in xs])   # [N,2]
    x_features = torch.stack([item[1].squeeze(0) for item in xs])   # [N,152]
    return x_spatial, x_features

# Ambil train+val+test lalu gabung
sp_train, ft_train = dataset_to_branches(train_ds)
sp_val,   ft_val   = dataset_to_branches(val_ds)
sp_test,  ft_test  = dataset_to_branches(test_ds)

x_spatial  = torch.cat([sp_train, sp_val, sp_test], dim=0).unsqueeze(1)   # [N,1,2]
x_features = torch.cat([ft_train, ft_val, ft_test], dim=0).unsqueeze(1)   # [N,1,152]

st.write("📐 Shape input spasial :", x_spatial.shape)
st.write("📐 Shape input fitur   :", x_features.shape)

# --------------------------
# Prediksi
# --------------------------
st.write("---")
st.subheader("🤖 Analisis GTNNWR (.pt)")

try:
    with torch.no_grad():
        # Masukkan ke model sebagai list of tensors
        y_pred = model([x_spatial, x_features])

    df_ordered = pd.concat([train_data, val_data, test_data], axis=0)
    df_ordered = df_ordered.sort_values("id")
    df_ordered["IKP_Prediksi"] = y_pred.cpu().numpy().flatten()[:len(df_ordered)]

    out = df.merge(df_ordered[["id","IKP_Prediksi"]], on="id", how="left")

    st.subheader("📑 Hasil Prediksi IKP")
    st.dataframe(out[["Tahun", prov_col, "IKP", "IKP_Prediksi"]])

    st.download_button(
        "📥 Download CSV Prediksi",
        out.to_csv(index=False).encode("utf-8"),
        file_name="prediksi_ikp.csv",
        mime="text/csv"
    )
except Exception as e:
    st.error(f"❌ Gagal menjalankan prediksi: {e}")
    st.info("💡 Coba cek kembali apakah jumlah fitur sudah tepat (harus 152).")
