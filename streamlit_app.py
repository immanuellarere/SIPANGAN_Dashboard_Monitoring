import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import branca.colormap as cm

from gtnnwr_wrapper import GTNNWRWrapper


# --------------------------
# Konfigurasi Halaman
# --------------------------
st.set_page_config(
    page_title="SIPANGAN Dashboard Monitoring",
    layout="wide"
)

st.title("📊 SIPANGAN Dashboard Monitoring")
st.caption("Monitoring Indeks Ketahanan Pangan (IKP) berbasis GTNNWR Pretrained (TorchScript .pt)")


# --------------------------
# Load Dataset
# --------------------------
DATA_PATH = "datasec.xlsx"
MODEL_PATH = "GTNNWR_DSi (1).pt"

try:
    if DATA_PATH.endswith(".csv"):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_excel(DATA_PATH, engine="openpyxl")

    # preprocessing dataset
    df = df.fillna(0)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    df["id"] = range(len(df))

except Exception as e:
    st.error(f"❌ Gagal membaca dataset: {e}")
    st.stop()


# --------------------------
# Tentukan kolom Provinsi
# --------------------------
if "Provinsi" in df.columns:
    prov_col = "Provinsi"
elif "Nama_Provinsi" in df.columns:
    prov_col = "Nama_Provinsi"
else:
    st.error("❌ Dataset harus punya kolom 'Provinsi' atau 'Nama_Provinsi'")
    st.stop()


# --------------------------
# Preview Data
# --------------------------
st.write("---")
st.subheader("🔍 Data Preview")
st.dataframe(df.head())


# --------------------------
# Peta IKP
# --------------------------
st.write("---")
st.subheader("🗺️ Peta Indonesia — IKP per Provinsi")

try:
    tahun_list = sorted(df["Tahun"].unique())
    tahun_peta = st.selectbox("Pilih Tahun untuk Peta", tahun_list)

    df_filtered = df[df["Tahun"] == tahun_peta].copy()

    # load geojson provinsi Indonesia
    url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
    gdf = gpd.read_file(url)

    gdf[prov_col] = gdf["Propinsi"].str.title()
    df_filtered[prov_col] = df_filtered[prov_col].str.title()

    ikp_min, ikp_max = df_filtered["IKP"].min(), df_filtered["IKP"].max()
    st.write(f"Range IKP ({tahun_peta}): {ikp_min:.2f} – {ikp_max:.2f}")

    bins = [0, 37.61, 48.27, 57.11, 65.96, 74.40, max(100, ikp_max + 1)]

    colormap = cm.StepColormap(
        colors=['#4B0000', '#FF3333', '#FF9999',
                '#CCFF99', '#66CC66', '#006600'],
        vmin=bins[0], vmax=bins[-1], index=bins,
        caption="Indeks Ketahanan Pangan (IKP)"
    )

    m = folium.Map(location=[-2.5, 118], zoom_start=5)
    gdf = gdf.merge(df_filtered[[prov_col, "IKP"]], on=prov_col, how="left")

    folium.GeoJson(
        gdf,
        style_function=lambda f: {
            "fillColor": colormap(f["properties"]["IKP"]) if f["properties"]["IKP"] else "gray",
            "fillOpacity": 0.7,
            "weight": 0.5,
            "color": "black"
        },
        tooltip=folium.GeoJsonTooltip(fields=[prov_col, "IKP"])
    ).add_to(m)

    colormap.add_to(m)
    st_folium(m, width=1000, height=600)

except Exception as e:
    st.error(f"❌ Gagal memuat peta: {e}")


# --------------------------
# Analisis GTNNWR
# --------------------------
st.write("---")
st.subheader("🤖 Analisis GTNNWR (Load Pretrained TorchScript)")

try:
    # load wrapper
    model = GTNNWRWrapper(prov_col=prov_col)
    model.load(MODEL_PATH)

    # prediksi
    df_pred = model.predict(df)

    # ambil koefisien
    coef_df = model.get_coefs(df_pred)
    if coef_df is not None:
        coef_filtered = coef_df[(coef_df["Tahun"] >= 2019) & (coef_df["Tahun"] <= 2024)]
        st.subheader("📑 Koefisien GTNNWR per Provinsi (2019–2024)")
        st.dataframe(coef_filtered)

        st.download_button(
            label="📥 Download Koefisien 2019–2024 (CSV)",
            data=coef_filtered.to_csv(index=False).encode("utf-8"),
            file_name="koefisien_2019_2024.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Tidak ada koefisien yang bisa ditampilkan.")

except Exception as e:
    st.error(f"❌ Gagal menjalankan GTNNWR: {e}")


# --------------------------
# Detail Provinsi
# --------------------------
st.write("---")
st.subheader("📍 Detail Provinsi")

try:
    prov = st.selectbox("Pilih Provinsi", df[prov_col].unique())
    prov_data = df_pred[df_pred[prov_col] == prov]

    st.write(f"### {prov} — IKP 2019–2024")
    st.dataframe(prov_data[["Tahun", "IKP", "IKP_Prediksi"]])

    if coef_df is not None:
        st.write(f"### {prov} — Koefisien GTNNWR 2019–2024")
        coef_prov = coef_df[coef_df[prov_col] == prov]
        st.dataframe(coef_prov)

except Exception:
    st.warning("⚠️ Data koefisien provinsi tidak tersedia.")
