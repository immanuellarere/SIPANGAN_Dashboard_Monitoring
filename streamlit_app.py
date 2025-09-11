import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import branca.colormap as cm
import torch

# --------------------------
# Konfigurasi Halaman
# --------------------------
st.set_page_config(page_title="SIPANGAN Dashboard Monitoring", layout="wide")
st.title("📊 SIPANGAN Dashboard Monitoring")
st.caption("Inference GTNNWR (.pt) dengan 2 branch input (spasial + fitur)")

DATA_PATH = "datasec.xlsx"
MODEL_PATH = "gtnnwr_model.pt"

# --------------------------
# Load data
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
# Load model
# --------------------------
@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ Model berhasil diload dari .pt")
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

# --------------------------
# Siapkan input untuk 2 branch
# --------------------------
x_columns = [
    'Skor_PPH','Luas_Panen','Produktivitas','Produksi',
    'Tanah_Longsor','Banjir','Kekeringan','Kebakaran','Cuaca',
    'OPD_Penggerek_Batang_Padi','OPD_Wereng_Batang_Coklat',
    'OPD_Tikus','OPD_Blas','OPD_Hwar_Daun','OPD_Tungro'
]

try:
    # branch spasial → [N,1,2]
    x_spatial = torch.tensor(df[["Longitude","Latitude"]].values, dtype=torch.float32)
    if x_spatial.dim() == 2:
        x_spatial = x_spatial.unsqueeze(1)

    # branch fitur utama (16 fitur) → [N,1,16]
    x_features = torch.tensor(df[x_columns].values, dtype=torch.float32)
    if x_features.dim() == 2:
        x_features = x_features.unsqueeze(1)

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
        y_pred = model((x_spatial, x_features))   # <<<<<< KUNCI: tuple dua branch

    df_pred = df.copy()
    df_pred["IKP_Prediksi"] = y_pred.cpu().numpy().flatten()[:len(df)]

    st.subheader("📑 Hasil Prediksi IKP")
    st.dataframe(df_pred[["Tahun", prov_col, "IKP", "IKP_Prediksi"]])

    st.download_button(
        "📥 Download CSV Prediksi",
        df_pred.to_csv(index=False).encode("utf-8"),
        file_name="prediksi_ikp.csv",
        mime="text/csv"
    )
except Exception as e:
    st.error(f"❌ Gagal menjalankan prediksi: {e}")
