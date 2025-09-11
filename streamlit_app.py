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
# Konfigurasi halaman
# --------------------------
st.set_page_config(page_title="SIPANGAN Dashboard Monitoring", layout="wide")
st.title("📊 SIPANGAN Dashboard Monitoring")
st.caption("Inference GTNNWR (.pt) dengan pipeline sama seperti training")

DATA_PATH = "datasec.xlsx"
MODEL_PATH = "gtnnwr_model.pt"   # model hasil training (.pt)

# --------------------------
# Load dataset
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
# Definisi fitur (harus sama dengan training)
# --------------------------
x_columns = [
    'Skor_PPH','Luas_Panen','Produktivitas','Produksi',
    'Tanah_Longsor','Banjir','Kekeringan','Kebakaran','Cuaca',
    'OPD_Penggerek_Batang_Padi','OPD_Wereng_Batang_Coklat',
    'OPD_Tikus','OPD_Blas','OPD_Hwar_Daun','OPD_Tungro'
]

# Split data sama dengan training
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
# Load arsitektur + bobot
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
        [[3],[512,256,64]],  # hidden layers sama seperti training
        drop_out=0.5,
        optimizer="Adadelta",
        optimizer_params=optim_params,
        write_path="./gtnnwr_runs",
        model_name="GTNNWR_DSi"
    )
    wrapper.add_graph()

    pretrained = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    try:
        wrapper._model.load_state_dict(pretrained.state_dict(), strict=True)
    except Exception:
        wrapper._model.load_state_dict(pretrained.state_dict(), strict=False)
    wrapper._model.eval()
    return wrapper

try:
    gtnnwr = load_model()
    st.success("✅ Model berhasil diload dari .pt (arsitektur + bobot)")
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

# --------------------------
# Bentuk input (2 branch)
# --------------------------
def make_inputs(df):
    # branch spasial
    x_spatial = torch.tensor(df[["Longitude","Latitude"]].values, dtype=torch.float32)
    if x_spatial.dim() == 2:
        x_spatial = x_spatial.unsqueeze(1)   # [N,1,2]

    # branch fitur (indikator + tahun)
    x_features = torch.tensor(df[x_columns+["Tahun"]].values, dtype=torch.float32)
    if x_features.dim() == 2:
        x_features = x_features.unsqueeze(1) # [N,1,F]

    return x_spatial, x_features

x_spatial, x_features = make_inputs(df)
st.write("📐 Shape input spasial:", x_spatial.shape)
st.write("📐 Shape input fitur  :", x_features.shape)

# --------------------------
# Prediksi
# --------------------------
st.write("---")
st.subheader("🤖 Analisis GTNNWR (.pt)")

try:
    device = torch.device("cpu")
    x_spatial = x_spatial.to(device)
    x_features = x_features.to(device)

    with torch.no_grad():
        # 🚑 FIX: pakai list [ ] bukan tuple
        y_pred = gtnnwr._model([x_spatial, x_features])

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
    st.info("💡 Coba cek arsitektur model dengan `st.text(str(gtnnwr._model))` untuk memastikan input/output layer cocok.")
