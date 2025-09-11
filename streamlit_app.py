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
st.caption("Inference GTNNWR dari .pt dengan input multi-branch")

DATA_PATH = "datasec.xlsx"
MODEL_PATH = "gtnnwr_model.pt"

# --------------------------
# Load Data
# --------------------------
try:
    df = pd.read_excel(DATA_PATH, engine="openpyxl") if DATA_PATH.endswith("xlsx") else pd.read_csv(DATA_PATH)
    df = df.fillna(0)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    df["id"] = range(len(df))
except Exception as e:
    st.error(f"❌ Gagal membaca dataset: {e}")
    st.stop()

prov_col = "Provinsi" if "Provinsi" in df.columns else ("Nama_Provinsi" if "Nama_Provinsi" in df.columns else None)
if prov_col is None:
    st.error("❌ Dataset harus punya kolom 'Provinsi' atau 'Nama_Provinsi'")
    st.stop()

st.write("---")
st.subheader("🔍 Data Preview")
st.dataframe(df.head())

# --------------------------
# Definisi fitur
# --------------------------
x_columns = [
    'Skor_PPH','Luas_Panen','Produktivitas','Produksi',
    'Tanah_Longsor','Banjir','Kekeringan','Kebakaran','Cuaca',
    'OPD_Penggerek_Batang_Padi','OPD_Wereng_Batang_Coklat',
    'OPD_Tikus','OPD_Blas','OPD_Hwar_Daun','OPD_Tungro'
]

# --------------------------
# Split dataset persis training
# --------------------------
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
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ Model berhasil diload dari .pt (arsitektur + bobot)")
except Exception as e:
    st.error(f"❌ Gagal load model .pt: {e}")
    st.stop()

# --------------------------
# Bentuk input multi-branch
# --------------------------
def dataset_to_branches(ds):
    """Ambil input tuple (x_spatial, x_features)"""
    xs = [ds[i][0] for i in range(len(ds))]   # setiap item sudah tuple
    x_spatial = torch.stack([item[0].squeeze(0) for item in xs])   # [N, 2]
    x_features = torch.stack([item[1].squeeze(0) for item in xs]) # [N, F]
    return x_spatial, x_features

try:
    sp_train, ft_train = dataset_to_branches(train_ds)
    sp_val,   ft_val   = dataset_to_branches(val_ds)
    sp_test,  ft_test  = dataset_to_branches(test_ds)

    x_spatial = torch.cat([sp_train, sp_val, sp_test], dim=0).unsqueeze(1)   # [N,1,2]
    x_features = torch.cat([ft_train, ft_val, ft_test], dim=0).unsqueeze(1) # [N,1,F]

    st.write("📐 Shape input spasial:", x_spatial.shape)
    st.write("📐 Shape input fitur  :", x_features.shape)
except Exception as e:
    st.error(f"❌ Gagal menyiapkan input: {e}")
    st.stop()

# --------------------------
# Prediksi
# --------------------------
st.write("---")
st.subheader("🤖 Analisis GTNNWR (.pt)")

try:
    with torch.no_grad():
        y_pred = model((x_spatial, x_features))   # <== PENTING: tuple, bukan tensor!

    df_ordered = pd.concat([train_data, val_data, test_data], axis=0)
    df_ordered = df_ordered.sort_values("id")
    df_ordered["IKP_Prediksi"] = y_pred.numpy().flatten()[:len(df_ordered)]

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
