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
st.caption("Monitoring Indeks Ketahanan Pangan (IKP) berbasis GTNNWR Pretrained (.pt)")

# --------------------------
# Load Dataset
# --------------------------
DATA_PATH = "datasec.xlsx"
MODEL_PATH = "gtnnwr_model.pt"   # full model .pt

try:
    df = pd.read_excel(DATA_PATH, engine="openpyxl") if DATA_PATH.endswith("xlsx") else pd.read_csv(DATA_PATH)
    df = df.fillna(0)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    df["id"] = range(len(df))
except Exception as e:
    st.error(f"❌ Gagal membaca dataset: {e}")
    st.stop()

# --------------------------
# Tentukan kolom Provinsi
# --------------------------
prov_col = "Provinsi" if "Provinsi" in df.columns else "Nama_Provinsi" if "Nama_Provinsi" in df.columns else None
if prov_col is None:
    st.error("❌ Dataset harus punya kolom 'Provinsi' atau 'Nama_Provinsi'")
    st.stop()

# --------------------------
# Preview Data
# --------------------------
st.write("---")
st.subheader("🔍 Data Preview")
st.dataframe(df.head())

# --------------------------
# Load Model .pt
# --------------------------
@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ Model berhasil diload dari .pt")
except Exception as e:
    st.error(f"❌ Gagal load model .pt: {e}")
    st.stop()

# --------------------------
# Peta IKP
# --------------------------
st.write("---")
st.subheader("🗺️ Peta Indonesia — IKP per Provinsi")

try:
    tahun_list = sorted(df["Tahun"].unique())
    tahun_peta = st.selectbox("Pilih Tahun untuk Peta", tahun_list)
    df_filtered = df[df["Tahun"] == tahun_peta].copy()

    url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
    gdf = gpd.read_file(url)
    gdf[prov_col] = gdf["Propinsi"].str.title()
    df_filtered[prov_col] = df_filtered[prov_col].str.title()

    ikp_min, ikp_max = df_filtered["IKP"].min(), df_filtered["IKP"].max()
    bins = [0, 37.61, 48.27, 57.11, 65.96, 74.40, max(100, ikp_max + 1)]
    colormap = cm.StepColormap(colors=['#4B0000', '#FF3333', '#FF9999',
                                       '#CCFF99', '#66CC66', '#006600'],
                               vmin=bins[0], vmax=bins[-1], index=bins,
                               caption="Indeks Ketahanan Pangan (IKP)")

    m = folium.Map(location=[-2.5, 118], zoom_start=5)
    gdf = gdf.merge(df_filtered[[prov_col, "IKP"]], on=prov_col, how="left")

    folium.GeoJson(
        gdf,
        style_function=lambda f: {"fillColor": colormap(f["properties"]["IKP"]) if f["properties"]["IKP"] else "gray",
                                  "fillOpacity": 0.7, "weight": 0.5, "color": "black"},
        tooltip=folium.GeoJsonTooltip(fields=[prov_col, "IKP"])
    ).add_to(m)

    colormap.add_to(m)
    st_folium(m, width=1000, height=600)

except Exception as e:
    st.error(f"❌ Gagal memuat peta: {e}")

# --------------------------
# Analisis GTNNWR (.pt)
# --------------------------
st.write("---")
st.subheader("🤖 Analisis GTNNWR (.pt)")

df_pred = None
try:
    # --- pastikan input sesuai dengan jumlah fitur yang dipakai saat training ---
    # ⚠️ ganti list fitur ini agar cocok dengan model Anda (kemarin terdeteksi 152 input!)
    x_columns = [c for c in df.columns if c not in ["IKP", prov_col, "id"]]

    x_input = torch.tensor(df[x_columns].values, dtype=torch.float32)
    if x_input.dim() == 2:
        x_input = x_input.unsqueeze(1)

    with torch.no_grad():
        y_pred = model(x_input)

    df_pred = df.copy()
    df_pred["IKP_Prediksi"] = y_pred.numpy().flatten()

    st.subheader("📑 Hasil Prediksi IKP")
    st.dataframe(df_pred[["Tahun", prov_col, "IKP", "IKP_Prediksi"]])
    st.download_button("📥 Download CSV Prediksi",
                       df_pred.to_csv(index=False).encode("utf-8"),
                       file_name="prediksi_ikp.csv", mime="text/csv")

except Exception as e:
    st.error(f"❌ Gagal menjalankan prediksi: {e}")

# --------------------------
# Detail Provinsi
# --------------------------
st.write("---")
st.subheader("📍 Detail Provinsi")

if df_pred is not None:
    prov = st.selectbox("Pilih Provinsi", df[prov_col].unique())
    prov_data = df_pred[df_pred[prov_col] == prov]
    st.write(f"### {prov} — IKP & Prediksi")
    st.dataframe(prov_data[["Tahun", "IKP", "IKP_Prediksi"]])
